from collections import Counter
from functools import partial
from itertools import groupby
from multiprocessing import Pool, cpu_count
from pathlib import Path
from random import sample
from typing import List

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from .base import BaseDataLoader
import numpy as np
import re
import soundfile as sf


class Cooll(BaseDataLoader):
    FS = 100_000
    F0 = 50
    NPTS = (FS // F0) * 20

    HOUSE_COUNT = 8  # artificial number of houses as done bei WRG implementation

    CONFIGS_REGEX = re.compile(r'^(?!#)(?P<key>.*)=(?P<value>.*)$')
    LABELS_REGEX = re.compile(r'^(?P<file_id>.*):\s+(?P<category>.*)_(?P<appliance>.*)_(?P<delay>.*)ms$')

    USED_CATEGORIES = [
        "Drill", "Fan", "Grinder", "Hair_drayer", "Hedge_trimmer",
        "Lamp", "Sander", "Saw", "Vacuum_cleaner"
    ]

    def __init__(self, path: Path):
        super().__init__(path)
        self.label_encoder = LabelEncoder()

    @property
    def flac_path(self):
        return self.path / 'data'

    @property
    def labels_file(self):
        return self.path / 'appliances_and_action_delays.txt'

    @property
    def configs_path(self):
        return self.path / 'configs'

    def _read_meta_file(self):
        """
        Read Meta files
        This parser ignores scenarios as there is only one.
        """
        meta = self._read_labels()

        for file_id in meta.keys():
            file_name = self.configs_path / f"scenario1_{file_id}.txt"
            _, file_id = file_name.stem.split("_")
            meta[file_id]['configs'] = {}
            for v in file_name.read_text().split('\n'):
                match = re.search(self.CONFIGS_REGEX, v)
                if match is not None:
                    meta[file_id]['configs'][match.group('key')] = int(match.group('value'))
        return meta

    def _read_labels(self):
        """
        Read labels from appliances_and_action_delays.txt
        :return:
        """
        data = self.labels_file.read_text().split('\n')
        labels = {}
        for l in data:
            match = re.search(self.LABELS_REGEX, l)
            if match is not None and match.group('category') in self.USED_CATEGORIES:
                labels[match.group('file_id')] = {
                    'category': match.group('category'),
                    'appliance': int(match.group('appliance')),
                    'delay': int(match.group('delay'))
                }
        return labels

    def _process_file(self, current_file_name: Path, voltage_file_name: Path, idx_on: List[int],
                      resize=True, align=True):
        current, _ = sf.read(str(current_file_name))
        voltage, _ = sf.read(str(voltage_file_name))
        current = current[idx_on - self.NPTS:idx_on + self.NPTS]
        voltage = voltage[idx_on - self.NPTS:idx_on + self.NPTS]

        try:
            current, voltage = self._preprocess(current, voltage, self.OUTPUT_SIZE, self.FS, self.F0, resize=resize, align=align)
        except:
            print(f"Something went wrong {voltage_file_name}")
        return current, voltage

    def _get_data(self, meta, resize=True, align=True):
        p = Pool(cpu_count())
        voltage_files = []
        current_files = []
        idx_on = []
        labels = []
        appliance_ids = []
        for file_id, m in meta.items():
            if file_id in ['0', '141']:  # skip 0 and 141
                continue
            voltage_files.append(self.flac_path / f"scenarioV1_{file_id}.flac")
            current_files.append(self.flac_path / f"scenarioC1_{file_id}.flac")
            configs = m['configs']

            idx_on.append(configs['tbna0'] * 100 + configs['ad1'])
            labels.append(m['category'])
            appliance_ids.append(m['appliance'])

        current, voltage = zip(*p.starmap(partial(self._process_file),
                                          zip(current_files, voltage_files, idx_on,
                                              [resize]*len(idx_on), [align]*len(idx_on))
                                          ))

        groups = self.randomize_groups(labels, appliance_ids)

        return np.array(current), np.array(voltage), np.array(labels), np.array(groups)

    def randomize_groups(self, labels, appliance_ids):
        # count groups
        group_dict = {k: [vv[1] for vv in v] for k, v in groupby(Counter(zip(labels, appliance_ids)), lambda x: x[0])}
        group_id_range = list(range(max(len(list(t)) for t in group_dict.values())))
        # randomize group label in range(0,max_group_id)
        for g, t in group_dict.items():
            group_dict[g] = sample(group_id_range, len(t))

        # assign new group labels
        groups = [group_dict[l][n - 1] for l, n in zip(labels, appliance_ids)]
        return groups

    def get_features(self, resize=True, align=True):
        meta = self._read_meta_file()
        current, voltage, labels, appliance_ids = self._get_data(meta, resize, align)
        assert len(current) == len(voltage) == len(labels) == len(appliance_ids)
        return current, voltage, labels, appliance_ids

    def iter_split(self, resize=True, align=True, split='LOGO'):
        current, voltage, labels, groups = self.get_features(resize, align)
        data = np.stack((current, voltage), axis=1)  # join channel first
        # groups = self._generate_groups(labels, appliance_ids,  self.HOUSE_COUNT)
        labels_encoded = self.label_encoder.fit_transform(labels)

        splitter_instance = self.SPLIT_CLS[split]()
        if split == 'LOGO':
            split_iter = splitter_instance.split(data, labels_encoded, groups)
        else:
            split_iter = splitter_instance.split(data, labels_encoded)
        for train_idx, test_idx in split_iter:
            if len(set(Counter(labels[test_idx]).keys()).difference(set(Counter(labels[train_idx]).keys()))) > 0:
                raise Exception("Test label not found in training set")
            yield data[train_idx], labels_encoded[train_idx], data[test_idx], labels_encoded[test_idx]
