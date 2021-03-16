from enum import Enum

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FloatDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    @property
    def num_outputs(self):
        return len(np.unique(self.labels))


class PhaseDataLoader:

    class Phase(Enum):
        TRAIN = "train"
        TEST = "test"

    def __init__(self, train: FloatDataset, test: FloatDataset, batch_size, shuffle, num_workers, sampler=None):
        self._data = {
            self.Phase.TRAIN: train,
            self.Phase.TEST: test
        }
        self._loaders = {
            self.Phase.TRAIN: DataLoader(train,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers,
                                         sampler=sampler
                                         ),
            self.Phase.TEST: DataLoader(test,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers
                                        )
        }

    @property
    def num_outputs(self):
        return max(v.num_outputs for v in self._data.values())

    def num_samples(self, phase):
        try:
            return len(self._data[phase])
        except KeyError:
            raise ValueError("Unknown Phase '{}".format(phase))

    def __getitem__(self, phase):
        try:
            return self._loaders[phase]
        except KeyError:
            raise ValueError("Unknown Phase '{}".format(phase))

    @property
    def num_inputs(self):
        return len(self._data[self.Phase.TRAIN][0][0][0])


def import_class(cl: str):
    """
    import given class name
    FROM https://stackoverflow.com/a/8255024

    :param cl: class FQN as string
    :return: class object
    """
    d = cl.rfind(".")
    classname = cl[d+1:len(cl)]
    m = __import__(cl[0:d], globals(), locals(), [classname])
    return getattr(m, classname)
