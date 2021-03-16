import torch
from torchsummary import summary

from ...data.loader import FloatDataset, PhaseDataLoader
from .recurrence import RecurrenceLayer
from ..utils import train_model
from sklearn.preprocessing import StandardScaler
import numpy as np

from .._base import ClassifierMixin

torch.backends.cudnn.deterministic = True


class RPSPPClassifier(ClassifierMixin):

    def __init__(self, device='cuda', pretrained_state_dict_path=None):
        self._model = None
        self._device = device

        self.nb_epochs = 80
        self.batch_size = 16
        self._name = self.__class__.__name__
        self._early_stop = True
        self.scaler = None
        self.pretrained_state_dict_path = pretrained_state_dict_path

    def fit(self, data, labels, test_data=None, test_label=None, log_dir=None):

        self._fit_scaler(data)
        dataset = PhaseDataLoader(FloatDataset(self.scale(data), labels),
                                  FloatDataset(self.scale(test_data), test_label),
                                  self.batch_size,
                                  shuffle=True,
                                  num_workers=1
                                  )

        model = SpacialRecurrenceModel(dataset.num_inputs, dataset.num_outputs).to(self._device)
        if self.pretrained_state_dict_path is not None:
            self._load_pretrained(model)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
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

    def _load_pretrained(self, model):
        if self.pretrained_state_dict_path is None:
            print("Warning: Pretrained state dict not provided")
            return

        pretrained_state_dict = torch.load(self.pretrained_state_dict_path)
        model_dict = model.state_dict()

        del pretrained_state_dict['fc.weight']
        del pretrained_state_dict['fc.bias']

        for k, v in model_dict.items():
            v.requires_grad = False

        model_dict.update(pretrained_state_dict)
        model.load_state_dict(model_dict)

    def _fit_scaler(self, data):
        swapped_data = data.swapaxes(1, 2)  # channel last
        reshaped_data = swapped_data.reshape(swapped_data.shape[0] * swapped_data.shape[1], swapped_data.shape[2])
        if self.scaler is None:
            # Fit scaler
            self.scaler = StandardScaler()
            self.scaler.fit(reshaped_data)

    def scale(self, data):
        # return data
        if self.scaler is None:
            raise Exception("Call _fit_scaler() first")

        swapped_data = data.swapaxes(1, 2)  # TODO: There is a better way...
        reshaped_data = swapped_data.reshape(swapped_data.shape[0]*swapped_data.shape[1], swapped_data.shape[2])
        scaled_data = self.scaler.transform(reshaped_data).reshape(data.shape[0],
                                                                   data.shape[2],
                                                                   data.shape[1]).swapaxes(2, 1)

        return scaled_data

    def predict(self, data):
        preds = []
        for d in self.scale(data):
            d = torch.FloatTensor([d])
            d = d.to(self._device)
            o = self._model(d)
            _, p = torch.max(o, 1)
            preds.append(p.cpu().numpy())
        return np.array(preds).flatten()


class SpacialRecurrenceModel(torch.nn.Module):

    DEFAULT_POOL = (8, 4, 2, 1)

    def __init__(self, num_inputs, num_outputs, in_channels=2, pool_size=DEFAULT_POOL, kernel_size=3, stride=1):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.in_channels = in_channels
        self.pool_size = pool_size

        self.kernel_size = kernel_size
        self.stride = stride
        self.recurrence = RecurrenceLayer()

        layers_size = [self.in_channels, self.in_channels, 32, 32, 32]
        self.conv = torch.nn.Sequential(
            *[self._conv_block(in_c, out_c,
                               self.kernel_size,
                               stride=self.stride,
                               batch_norm=i > 0,
                               max_pool=i == 0
                               ) for i, (in_c, out_c) in
              enumerate(zip(layers_size, layers_size[1:]))]
        )

        self.pools = [torch.nn.AdaptiveMaxPool2d((p, p)) for p in pool_size]
        self.dropout2d = torch.nn.Dropout2d(0.2)

        self.fc = torch.nn.Linear(self._total_pool_size(layers_size), num_outputs)

    def _total_pool_size(self, layers_size):
        return sum(layers_size[-1] * p**2 for p in self.pool_size)

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding=1, batch_norm=False, max_pool=False):
        conv = [
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            torch.nn.ReLU()
        ]
        if batch_norm:
            conv.append(torch.nn.BatchNorm2d(out_channels))
        if max_pool:
            conv.append(torch.nn.MaxPool2d(3))
        return torch.nn.Sequential(*conv)

    def spatial_pooling(self, x):
        outs = []
        for pool in self.pools:
            out1 = pool(x)
            out1_shape = out1.shape
            sec_dim = out1_shape[-1] * out1_shape[-2] * out1_shape[-3]
            out1 = out1.contiguous().view(-1, sec_dim)
            outs.append(out1)

        return torch.cat(outs, 1)

    def apply_recurrence(self, x):
        x = torch.cat(
            [self.recurrence(x[:, i, :]) for i in range(x.shape[1])],
            dim=1
        )
        return x

    def forward(self, x):
        x = self.apply_recurrence(x)
        x = self.conv(x)
        x = self.dropout2d(x)
        x = self.spatial_pooling(x)
        x = self.fc(x)

        return x
