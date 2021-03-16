from datetime import datetime
import logging
from pathlib import Path

import click
from sklearn.metrics import f1_score

from .classifier.utils import seed_torch
from .data.base import BaseDataLoader
from .classifier.rpspp.classifier import RPSPPClassifier
from .classifier.wrg.classifier import WRGClassifier

CLASSIFIERS = {
    'WRG': WRGClassifier,
    'SPACIAL': RPSPPClassifier
}


def setup_logging(log_level):
    ext_logger = logging.getLogger("py.warnings")
    logging.captureWarnings(True)
    level = getattr(logging, log_level)
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(filename)s: %(message)s", level=level)
    if level <= logging.DEBUG:
        ext_logger.setLevel(logging.WARNING)


@click.group()
@click.option("-l", "--log-level", default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']))
def cli(log_level):
    setup_logging(log_level)
    seed_torch()


@cli.command()
@click.option("--log-path", default='logs')
@click.option("--plaid-path")
@click.option("--cooll-path")
@click.option("--whited-path")
@click.option("--verbose", is_flag=True)
@click.option("--paa-size", type=int, help="Output size of PAA")
@click.option("--device", type=str, default="cuda")
@click.option("--classifier", 'classifier_name', type=click.Choice(CLASSIFIERS.keys()), default="SPACIAL")
@click.option("--split", type=click.Choice(BaseDataLoader.SPLIT_CLS.keys()), default="LOGO")
@click.option("--pretrained-state-dict", 'pretrained_state_dict_path', type=str)
def device_identify(log_path, plaid_path, cooll_path, whited_path, verbose, paa_size, device, classifier_name, split, pretrained_state_dict_path):
    """Classification task"""

    import torch
    from .data.base import BaseDataLoader
    from .data.plaid import Plaid
    from .data.cooll import Cooll
    from .data.whited import Whited
    from .evaluation import EvaluationLog

    log_path = Path(log_path).absolute() / f"{datetime.now().isoformat()}"

    print("Log path: ", log_path)

    classifier_cls = CLASSIFIERS[classifier_name]

    datasets = []
    if cooll_path is not None:
        cooll_path = Path(cooll_path).absolute()
        datasets.append((Cooll(cooll_path), "COOLL"))
    if plaid_path is not None:
        plaid_path = Path(plaid_path).absolute()
        datasets.append((Plaid(plaid_path), "PLAID"))
    if whited_path is not None:
        whited_path = Path(whited_path).absolute()
        datasets.append((Whited(whited_path), "WHITED"))

    for it, dataset_name in datasets:
        print(f"------ Classifier: {classifier_cls.__name__} -------")
        print(f"------ Dataset: {dataset_name} -------")
        evaluation = EvaluationLog(verbose=verbose)
        file_name = f"{classifier_cls.__name__}.yaml"

        log_path_dataset = log_path / dataset_name

        if paa_size is not None:
            it.OUTPUT_SIZE = paa_size

        if classifier_cls == WRGClassifier:
            data_iterator = it.iter_split(resize=False, split=split)
        else:
            data_iterator = it.iter_split(split=split)

        evaluation.add('meta', 'dataset', dataset_name)
        evaluation.add('meta', 'split', split)
        evaluation.add('meta', 'classifier', classifier_name)
        for fold, (train_data, train_labels, test_data, test_labels) in enumerate(data_iterator):
            print("Train Shape:", train_data.shape, "Test Shape:", test_data.shape)
            print(f"------ Fold: {fold} -------")
            if classifier_cls == WRGClassifier:
                classifier = classifier_cls(device, dataset_name)
            else:
                classifier = classifier_cls(device, pretrained_state_dict_path)

            best_model, test_acc = classifier.fit(train_data, train_labels, test_data, test_labels, log_path_dataset / f"fold_{fold}")
            torch.save(best_model.state_dict(), log_path_dataset / f"fold_{fold}" / "model.pt")
            data = {'train': (train_data, train_labels), 'test': (test_data, test_labels)}

            for key, (e_data, e_labels) in data.items():
                pred_labels = classifier.predict(e_data)
                assert len(pred_labels) == len(e_labels) and len(e_labels) > 0
                evaluation.add('meta', 'dataset', dataset_name)
                evaluation.add('meta', 'labels', it.label_encoder.classes_.tolist())
                evaluation.add('meta', 'paa-size', BaseDataLoader.OUTPUT_SIZE)
                evaluation.f1(key, e_labels, pred_labels, fold, average='macro')
                evaluation.precision_recall_fscore_support(key, e_labels, pred_labels, fold, average=None)
                evaluation.add(key, 'acc', float((pred_labels == e_labels).sum()) / len(e_labels), fold)
                evaluation.confusion_matrix(key, e_labels, pred_labels, fold, list(range(len(it.label_encoder.classes_.tolist()))))
                evaluation.save_yaml(log_path_dataset / file_name)

        evaluation.print()


@cli.command()
@click.argument('file-paths', nargs=-1)
@click.option('--latex', is_flag=True)
@click.option('--cm', is_flag=True, help="Save confusion matrix")
@click.option('--boxplot', is_flag=True, help="Save boxplot matrix")
def parse_logs(file_paths, latex, cm, boxplot):
    from .evaluation import EvaluationLog
    import numpy as np

    evs = []
    for fp in file_paths:
        evs.append(EvaluationLog.from_yaml(Path(fp)))

    if cm:
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 13
        })

        for i, (path, e) in enumerate(zip(file_paths, evs)):
            combined = np.zeros((len(e['meta']['labels']), len(e['meta']['labels'])))
            for f in e['test']:
                combined += np.array(f['confusion_matrix'])
            cmn = combined.astype('float') / combined.sum(axis=1)[:, np.newaxis]
            cm_labels = [l.split('_')[0] for l in e['meta']['labels']]
            disp = ConfusionMatrixDisplay(confusion_matrix=cmn, display_labels=cm_labels)
            fig = plt.figure()
            disp.plot(cmap=plt.cm.viridis, colorbar=True,  values_format='.0f', xticks_rotation='vertical', include_values=False)
            fig.set_size_inches(6, 4)
            plt.tight_layout()
            plt.savefig(str(Path(path).parent / f'cm_test.pdf'), dpi=600, bbox_inches='tight', pad_inches=0)

    if boxplot:
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 13
        })
        for i, (path, e) in enumerate(zip(file_paths, evs)):
            combined = np.zeros((len(e['meta']['labels']), len(e['meta']['labels'])))
            for f in e['test']:
                combined += np.array(f['confusion_matrix'])
            # calculate from y_true and y_pred from confusion matrix
            y_true = []
            y_pred = []
            for i, c in enumerate(combined):
                y_true += [i] * int(c.sum())
                for j, x in enumerate(c):
                    if x > 0:
                        y_pred += [j] * int(x)
            labels = [l.replace("_", "\_") for l in e['meta']['labels']]
            f1_per_class = np.array(sorted(zip(
                labels,
                combined.sum(axis=1),
                f1_score(y_true, y_pred, average=None)
            ), key=lambda x: x[-1], reverse=False))

            y_pos = np.arange(f1_per_class.shape[0])
            fig, ax = plt.subplots()
            ax.barh(y_pos, f1_per_class[:, 2].astype(float), alpha=0.5)
            # add numbers
            for i, v in enumerate(f1_per_class[:, 2].astype(float)):
                ax.text(v + 0.05, i -.25, f"{v:.4f}")

            # draw f1 line
            f1 = f1_score(y_true, y_pred, average='macro')
            ax.axvline(x=f1)
            ax.text(f1, f1_per_class.shape[0] + 1, "macro-$F_1$", horizontalalignment="center")

            plt.yticks(y_pos, f1_per_class[:, 0])
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            # remove frame
            for s in ['top', 'bottom', 'right']:
                ax.spines[s].set_visible(False)

            fig.set_size_inches(4, 6)
            plt.tight_layout()
            plt.savefig(str(Path(path).parent / f'boxplot.pdf'), dpi=600, bbox_inches='tight', pad_inches=0)
            print(f1_per_class)
            print("macro F1: ", f1)

    for i, (fn, e) in enumerate(zip(file_paths, evs)):
        f1_scores = [f['f1'] for f in e['test']]
        if len(f1_scores) > 0 and isinstance(f1_scores[0], list):
            f1_mean = np.array([np.array(f1).mean() for f1 in f1_scores]).mean()
        else:
            f1_mean = np.array(f1_scores).mean()
        try:
            split = e['meta']['split']
        except KeyError:
            split = "Unkown"
        print(fn, e['meta']['dataset'], len(e['test']), e['meta']['paa-size'], split, f1_mean)
