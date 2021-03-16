
from pathlib import Path
from pprint import pprint
from statistics import mean
from typing import Union

import numpy as np
import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix


class EvaluationLog:
    def __init__(self, verbose=True):
        self.d = {}
        self.im = {}
        self.verbose = verbose

    def precision_recall_fscore_support(self, section, y_true, y_pred, fold=None, average='weighted'):
        precision, recall, fscore, support = (v.tolist() if v is not None else None
                                              for v in precision_recall_fscore_support(y_true, y_pred, average=average))
        self.add(section, 'precision', precision, fold)
        self.add(section, 'recall', recall, fold)
        self.add(section, 'fscore', fscore, fold)
        self.add(section, 'support', support, fold)

    def f1(self, section, y_true, y_pred, fold=None, average='weighted'):
        f1 = f1_score(y_true, y_pred, average=average)
        if average is None:
            f1 = [float(f) for f in f1]
        else:
            f1 = float(f1)
        self.add(section, 'f1', f1, fold)

    def confusion_matrix(self, section, y_true, y_pred, fold=None, labels=None):
        if labels is None:
            labels = np.unique(y_true)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        self.add(section, 'confusion_matrix', cm.tolist(), fold)

    def add(self, section, score, value, fold=None):
        if self.verbose:
            print(section, score, value, fold)
        try:
            s = self.d[section]
        except KeyError:
            if fold is None:
                s = {}
            else:
                s = []
            self.d[section] = s
        if fold is None:
            s[score] = value
        else:
            if len(s) <= fold:
                s.append({})
            s[fold][score] = value

    def save_yaml(self, file_name: Union[str, Path]):
        if isinstance(file_name, str):
            file_name = Path(file_name)

        file_name.parent.mkdir(exist_ok=True, parents=True)
        with file_name.open('w') as f:
            yaml.dump(self.d, f)

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]):
        if isinstance(file_path, str):
            file_path = Path(file_path)

        instance = cls()
        with file_path.open() as f:
            instance.d = yaml.load(f, Loader=yaml.Loader)
        return instance

    def print(self):
        pprint(self.d)

    def mean(self, section, score):
        return mean([x[score] for x in self[section]])

    def __getitem__(self, section):
        return self.d[section]

    def __repr__(self):
        return f"Evaluation({self['meta']})"
