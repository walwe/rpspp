from collections import Counter
from pathlib import Path
from random import choices

from pyts.approximation import PiecewiseAggregateApproximation
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold


class AllFold:

    def split(self, X, y):
        assert len(X) == len(y)
        idx_range = np.arange(len(X))
        yield idx_range, idx_range


class BaseDataLoader:

    OUTPUT_SIZE = 48

    SPLIT_CLS = {
        'LOGO': LeaveOneGroupOut,
        'SKFOLD': StratifiedKFold,
        'ALL': AllFold
    }

    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def _preprocess(cls, current, voltage, output_size: int, fs: int, f0: int, resize=True, align=True):
        """Pre process current and voltage"""
        if align:
            c, v = cls.align_zero_crossing(current, voltage, fs // f0)
        else:
            c, v = current, voltage

        if resize:
            return cls._paa([c], output_size)[0], cls._paa([v], output_size)[0]
        else:
            return c, v

    @classmethod
    def zero_crossings(cls, data):
        """
        Find indices of all zero crossings
        https://stackoverflow.com/a/39537079
        """
        positive = data > 0
        return np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]

    @classmethod
    def get_zero_crossing(cls, data, period_length):
        zero_crossings = cls.zero_crossings(data)

        if len(zero_crossings) == 0:
            return

        if data[zero_crossings[0]] > data[zero_crossings[0] + 1]:
            zero_crossings = zero_crossings[1:]  # start on up hill slope

        if len(zero_crossings) % 2 == 1:
            zero_crossings = zero_crossings[:-1]   # we want even number of crossings

        if zero_crossings[-2] + period_length >= len(data):  # assure full period for last crossing
            zero_crossings = zero_crossings[:-2]

        return zero_crossings

    @classmethod
    def moving_average(cls, values, window):
        """https://gordoncluster.wordpress.com/2014/02/13/python-numpy-how-to-generate-moving-averages-efficiently-part-2/"""
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma

    @classmethod
    def align_zero_crossing(cls, current, voltage, period_length):
        zv = cls.get_zero_crossing(voltage, period_length)
        for crossing in reversed(zv[:-1]):
            start, stop = crossing, crossing+period_length
            if len(current[start:stop]) == len(voltage[start: stop]) == period_length:
                return current[start:stop], voltage[start: stop]

    @classmethod
    def _paa(cls, features, width):
        """Shorten features using PAA"""

        paa = PiecewiseAggregateApproximation(window_size=None, output_size=width, overlapping=False)
        return paa.transform(features)

    @classmethod
    def _generate_groups(cls, labels: np.array, appliance_ids: np.array, num_groups, min_samples_per_group=10):
        """Assign each label to a group"""
        label_count = Counter(labels)
        # calculate in how many groups the label will appear
        label_split_count = {k: min(v // min_samples_per_group, num_groups) for k, v in label_count.items()}
        groups = np.empty(len(labels), dtype=int)
        group_names = range(num_groups)
        for label_group, gc in label_split_count.items():
            idx = np.where(labels == label_group)[0]
            np.random.shuffle(idx)  # shuffle labels
            idx = np.array_split(idx, gc)  # split label group into gc-number of groups
            for idx, group in zip(idx, choices(group_names, k=gc)):
                groups[idx] = group  # randomly assign to groups
        return groups

    def iter_split(self, resize=True, align=True):
        raise NotImplementedError
