import json
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import subprocess
from typing import List

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from .base import BaseDataLoader


class Plaid(BaseDataLoader):

    FS = 30_000
    F0 = 60
    NPTS = (FS // F0) * 20
    # OUTPUT_SIZE = 30  # Optimum

    def __init__(self, path):
        super().__init__(path)
        self.label_encoder = LabelEncoder()

    @property
    def csv_path(self) -> str:
        return self.path / 'CSV'

    @property
    def json_path(self) -> List[Path]:
        return [
            self.path / 'meta_2014.json',
            self.path / 'meta_2017.json'
        ]

    def _read_meta_file(self) -> dict:
        meta = []
        for j in self.json_path:
            with j.open() as f:
                meta += json.load(f)
        return meta

    def _read_csv_tail(self, file_name: Path, offset: int = NPTS):
        """https://stackoverflow.com/a/136280"""
        proc = subprocess.Popen(['tail', '-n', str(offset), str(file_name)], stdout=subprocess.PIPE)
        data = np.genfromtxt(proc.stdout, delimiter=',')
        return data

    def _process_file(self, file_name, resize=True, align=True):
        """Read and preprocess given file"""
        data = self._read_csv_tail(file_name)
        current, voltage = self._preprocess(data[:, 0], data[:, 1], self.OUTPUT_SIZE, self.FS, self.F0, resize, align)
        return current, voltage

    def _get_data(self, meta: dict, resize=True, align=True, multi_thread=True):
        """Read data files in parallel"""
        p = Pool(cpu_count() if multi_thread else 1)
        csv_files = [self.csv_path / f"{item['id']}.csv" for item in meta]
        current, voltage = zip(*p.starmap(partial(self._process_file),
                                      zip(csv_files, [resize]*len(csv_files), [align]*len(csv_files)))
                               )
        return np.array(current), np.array(voltage)

    def get_features(self, resize=True, align=True):
        meta = self._read_meta_file()

        current, voltage = self._get_data(meta, resize, align)
        houses = [v['meta']['location'] for v in meta]
        categories = [v['meta']['appliance']['type'] for v in meta]

        assert len(current) == len(voltage) == len(houses) == len(categories)

        return np.array(current), np.array(voltage), np.array(houses), np.array(categories)

    def iter_split(self, resize=True, align=True, amount_houses_test=1, split='LOGO'):
        current, voltage, houses, categories = self.get_features(resize, align)
        data = np.stack((current, voltage), axis=1)   # join channel first
        categories_encoded = self.label_encoder.fit_transform(categories)

        splitter_instance = self.SPLIT_CLS[split]()
        if split == 'LOGO':
            split_iter = splitter_instance.split(data, categories_encoded, houses)
        else:
            split_iter = splitter_instance.split(data, categories_encoded)
        for train_idx, test_idx in split_iter:
            yield data[train_idx], categories_encoded[train_idx], data[test_idx], categories_encoded[test_idx]
