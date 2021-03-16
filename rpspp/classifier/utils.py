import copy
import torch
import logging

from torch.utils.tensorboard import SummaryWriter
from rpspp.data.loader import PhaseDataLoader
from tqdm import trange

DEFAULT_SEED = 4783957


def seed_torch(seed=DEFAULT_SEED):
    import random
    import numpy as np
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_model(model, name, data_loaders, criterion, optimizer, scheduler,
                device, num_epochs=24, log_dir=None, early_stop=None):

    early_stop = num_epochs if early_stop is None else early_stop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0
    epoch_loss = None

    writer = SummaryWriter(str(log_dir))
    epoch_iter = trange(num_epochs, desc=f"-- {name} #")

    for epoch in epoch_iter:
        test_losses = []

        # Each epoch has a training and validation phase
        for phase in PhaseDataLoader.Phase:
            if data_loaders.num_samples(phase) == 0:
                logging.info(f"Skipping Phase: {phase}. No samples available")
                continue
            is_train = phase == PhaseDataLoader.Phase.TRAIN
            if is_train:
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_f1_loss = 0

            dataset = data_loaders[phase]

            # Iterate over data.
            for inputs, labels in dataset:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(is_train):

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if is_train:  # phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_f1_loss += f1_loss(labels, preds)

            if is_train and scheduler is not None:
                scheduler.step(loss)

            epoch_loss = running_loss / data_loaders.num_samples(phase)
            epoch_acc = running_corrects.double() / data_loaders.num_samples(phase)
            epoch_f1 = running_f1_loss / data_loaders.num_samples(phase)

            writer.add_scalar(f'Loss/{phase.value}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase.value}', epoch_acc, epoch)
            writer.add_scalar(f'F1-Loss/{phase.value}', epoch_f1, epoch)

            if not is_train:
                test_losses.append(epoch_loss)

            epoch_iter.set_description(
                f'Epoch: {epoch} Phase: {phase.value} | '
                f'Best Test Acc: {best_acc:.4f} | '
                f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1-loss: {epoch_f1:.4f}',
                refresh=True
            )

            # deep copy the model
            if (not is_train) and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if (not is_train) and epoch_acc > best_f1:
                best_f1 = epoch_f1

        if early_stop is not None or len(test_losses) < early_stop:
            continue
        else:
            if all(epoch_loss >= x for x in test_losses[-early_stop:]):
                print(f'Early stopping after {epoch}, train loss not decreasing for {early_stop} epochs')
                break

    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, f'{best_acc:3.4f}'


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    """
    Calculate F1 score. Can work with gpu tensors
    FROM https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    The original implementation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    """
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1
