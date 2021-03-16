import torch
import numpy as np

from .model import CNN2DModel
from .._base import ClassifierMixin
from ..utils import train_model
from ...data.loader import PhaseDataLoader, FloatDataset
from .data import CoolData, WhitedData, PlaidData

torch.backends.cudnn.deterministic = True


class WRGClassifier(ClassifierMixin):

    def __init__(self, device='cuda', dataset=None):
        self._model = None
        self._device = device
        self.dataset = dataset
        self.batch_size = 16
        self._name = self.__class__.__name__
        self.nb_epochs = 100
        self._early_stop = False

    def preprocess(self, data):
        if self.dataset == "COOLL":
            data_cls = CoolData
        elif self.dataset == "PLAID":
            data_cls = PlaidData
        elif self.dataset == "WHITED":
            data_cls = WhitedData
        return np.concatenate(data_cls().transform(data[:, 0], data[:, 1]), axis=1)

    def fit(self, data, labels, test_data=None, test_label=None, log_dir=None):
        data = self.preprocess(data)
        test_data = self.preprocess(test_data)
        dataset = PhaseDataLoader(FloatDataset(data, labels),
                                  FloatDataset(test_data, test_label),
                                  self.batch_size,
                                  shuffle=True,
                                  num_workers=1
                                  )
        model = CNN2DModel(n_channels=2,
                           n_kernels=64,
                           n_layers=3,
                           emb_size=dataset.num_inputs,
                           dropout=0.25,
                           output_size=dataset.num_outputs).to(self._device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None

        best_model, test_acc = train_model(model,
                                           self._name,
                                           dataset,
                                           criterion,
                                           optimizer,
                                           scheduler,
                                           self._device,
                                           num_epochs=self.nb_epochs,
                                           log_dir=log_dir,
                                           early_stop=self._early_stop)
        self._model = best_model
        return best_model, test_acc

    def predict(self, data):
        preds = []
        data = self.preprocess(data)
        for d in data:
            d = torch.FloatTensor([d])
            d = d.to(self._device)
            o = self._model(d)
            _, p = torch.max(o, 1)
            preds.append(p.cpu().numpy())
        return np.array(preds).flatten()