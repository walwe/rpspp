from collections import Counter
from functools import partial
from itertools import groupby
from multiprocessing import Pool, cpu_count
from pathlib import Path
from random import sample

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

from .base import BaseDataLoader
import soundfile as sf


class Whited(BaseDataLoader):
    FS = 44_100
    F0 = 50

    HOUSE_COUNT = 9  # artificial number of houses as done bei WRG implementation

    # Appliances used by WRG https://github.com/sambaiga/WRG-NILM/blob/master/src/data/load_whited_data.py#L9
    USED_CATEGORIES = [
        'CFL', 'DrillingMachine', 'Fan', 'FlatIron', 'GameConsole', 'HairDryer', 'Iron', 'Kettle', 'LEDLight',
        'LightBulb', 'Massage', 'Microwave', 'Mixer', 'Monitor', 'PowerSupply', 'ShoeWarmer', 'Shredder',
        'SolderingIron', 'Toaster', 'VacuumCleaner', 'WaterHeater',
        'Monitor'
    ]  # monitor missing

    def __init__(self, path):
        super().__init__(path)
        self.label_encoder = LabelEncoder()

    def _process_file(self, file_name: Path, resize=True, align=True):
        category, name, region, kit, ts = self._parse_file_name(file_name)
        data, _ = sf.read(str(file_name))
        current, voltage = self._preprocess(data[:, 1], data[:, 0], self.OUTPUT_SIZE, self.FS, self.F0, resize, align)
        return current, voltage, category, name

    def _parse_file_name(self, file_name):
        category, name, region, kit, ts = file_name.stem.split("_")
        return category, name, region, kit, ts

    def get_features(self, resize=True, align=True):
        p = Pool(cpu_count())
        flac_files = self.path.glob("*.flac")
        # only load wanted categories
        flac_files = list(filter(
            lambda file_name: self._parse_file_name(file_name)[0] in self.USED_CATEGORIES, flac_files
        ))
        current, voltage, categories, names = zip(*p.starmap(
            partial(self._process_file),
            zip(flac_files, [resize]*len(flac_files), [align]*len(flac_files)))
                                           )
        assert len(current) == len(voltage) == len(categories) == len(names)
        return np.array(current), np.array(voltage), np.array(categories), np.array(names)

    def generate_groups(self, labels, names):
        # generate group label for each appliance model
        group_dict = {g: {nn[1]: i for i, nn in enumerate(n)}
                      for g, n in groupby(sorted(Counter(zip(labels, names))), lambda x: x[0])}
        group_id_range = list(range(max(len(t.values()) for t in group_dict.values())))

        # randomize group label in range(0,max_group_id)
        for g, t in group_dict.items():
            new_groups = sample(group_id_range, len(t.values()))
            for tt, i in t.items():
                t[tt] = new_groups[i]

        groups = [group_dict[l][n] for l, n in zip(labels, names)]

        return np.array(groups)

    def iter_split(self, resize=True, align=True, split='LOGO'):
        current, voltage, labels, names = self.get_features(resize, align)
        data = np.stack((current, voltage), axis=1)  # join channel first
        # groups = self._generate_groups(labels, self.HOUSE_COUNT)
        groups = self.generate_groups(labels, names)
        labels_encoded = self.label_encoder.fit_transform(labels)

        splitter_instance = self.SPLIT_CLS[split]()
        if split == 'LOGO':
            split_iter = splitter_instance.split(data, labels_encoded, groups)
        else:
            split_iter = splitter_instance.split(data, labels_encoded)

        for train_idx, test_idx in split_iter:
            yield data[train_idx], labels_encoded[train_idx], data[test_idx], labels_encoded[test_idx]
