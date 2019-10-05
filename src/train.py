from __future__ import print_function, division

import argparse
from ml.src.metrics import Metric
from eeglibrary.src.eeg_dataloader import set_dataloader
from ml.models.model_manager import model_manager_args, BaseModelManager
from eeglibrary.src.eeg_dataset import EEGDataSet
from eeglibrary.src.preprocessor import preprocess_args
from eeglibrary import eeg
import torch


LABELS = {'null': 0, 'seiz': 1, 'bckg': 2}


def train_args(parser):
    train_parser = parser.add_argument_group('train arguments')
    train_parser.add_argument('--only-model-test', action='store_true', help='Load learned model and not training')
    train_parser.add_argument('--test', action='store_true', help='Do testing')
    parser = preprocess_args(parser)
    parser = model_manager_args(parser)

    return parser


def label_func(path):
    return LABELS[path[-8:-4]]


def load_func(path):
    return torch.from_numpy(eeg.load_pkl(path).values.reshape(-1, ))


def train(train_conf) -> float:

    phases = ['train', 'val', 'test']

    class_names = list(set(LABELS.values()))
    metrics = [
        Metric('loss', direction='minimize'),
        Metric('accuracy', direction='maximize'),
        Metric('recall_1', direction='maximize', save_model=True),
        Metric('recall_2', direction='maximize'),
        Metric('far', direction='minimize')
    ]

    train_conf.update({
        'class_names': list(range(len(class_names))),
        'load_func': load_func,
        'label_func': label_func,
    })

    # dataset, dataloaderの作成
    dataloaders = {}
    for phase in phases:
        dataset = EEGDataSet(train_conf[f'{phase}_path'], train_conf)
        dataloaders[phase] = set_dataloader(dataset, phase, train_conf)

    # modelManagerをインスタンス化、trainの実行
    model_manager = BaseModelManager(class_names, train_conf, dataloaders, metrics)

    # モデルの学習を行う場合
    if not train_conf['only_model_test']:
        model_manager.train()

    if train_conf['only_model_test'] or train_conf['test']:
        model_manager.test()

    # if train_conf['train_manifest'] == 'all':
    #     for sub_name in subject_dir_names:
    #         args = arrange_paths(args, sub_name)
    #         train(args, class_names, label_func, metrics)
    # elif args.inference:
    #     pred_list, path_list = train(args, class_names, label_func, metrics)
    #     voting(args, pred_list, path_list)
    # else:
    #     train(args, class_names, label_func, metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    train_conf = vars(train_args(parser).parse_args())
    assert train_conf['train_path'] != '' or train_conf['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'
    # returns loss or accuracy
    train(train_conf)
