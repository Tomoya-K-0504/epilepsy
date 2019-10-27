from __future__ import print_function, division

import argparse
import numpy as np
from ml.src.metrics import Metric
from eeglibrary.src.eeg_dataloader import set_dataloader
from ml.models.model_manager import model_manager_args, BaseModelManager
from eeglibrary.src.eeg_dataset import EEGDataSet
from eeglibrary.src.preprocessor import preprocess_args
from eeglibrary import eeg
from ml.tasks.train_manager import TrainManager, train_manager_args
import torch
from pathlib import Path
from src.const import BONN_LABELS
from sklearn.metrics import confusion_matrix


LABELS = ['none', 'interictal', 'ictal']


def train_args(parser):
    parser = train_manager_args(parser)
    parser = preprocess_args(parser)

    return parser


def label_func(path):
    return LABELS.index(BONN_LABELS[Path(path).parent.name[:-4]])


def load_func(path):
    pass


def voting(n_split, pred_list, label_list):
    assert len(pred_list) / n_split == float(len(pred_list) // n_split)
    voted_pred_list, voted_label_list = np.zeros(len(pred_list) // n_split), np.zeros(len(pred_list) // n_split)

    for i in range(len(pred_list) // n_split):
        voted_pred_list[i] = np.argmax(np.bincount(pred_list[i * n_split:(i + 1) * n_split]))
        voted_label_list[i] = np.argmax(np.bincount(label_list[i * n_split:(i + 1) * n_split]))

    print(confusion_matrix(voted_label_list, voted_pred_list))


def experiment(train_conf) -> float:

    dataset_cls = EEGDataSet
    set_dataloader_func = set_dataloader

    metrics = [
        Metric('loss', direction='minimize'),
        Metric('accuracy', direction='maximize', save_model=True),
        Metric('recall_1', direction='maximize'),
        Metric('recall_2', direction='maximize'),
        # Metric('far', direction='minimize')
    ]

    train_conf['class_names'] = list(range(len(LABELS)))
    train_manager = TrainManager(train_conf, load_func, label_func, dataset_cls, set_dataloader_func, metrics)

    train_manager.train()

    if train_conf['test']:
        pred_list, label_list = train_manager.test()
        voting(256, pred_list, label_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    train_conf = vars(train_args(parser).parse_args())
    assert train_conf['train_path'] != '' or train_conf['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'
    # returns loss or accuracy
    experiment(train_conf)
