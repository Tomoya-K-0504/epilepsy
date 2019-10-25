import eeglibrary
import pyedflib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse
from eeglibrary import EEG
from src.const import BONN_LABELS


def annotate_args():
    parser = argparse.ArgumentParser(description='Annotation arguments')
    parser.add_argument('--data-dir', metavar='DIR',
                        help='Directory where data placed', default='/media/tomoya/SSD-PGU3/research/brain/Bonn_dataset')
    parser.add_argument('--duration', type=float,
                        help='duration of one splitted wave', default=10)
    # parser.add_argument('--annotate-method', type=int,
    #                     help='The way to annotate, 1 or 2', default=1)

    return parser


if __name__ == '__main__':

    args = annotate_args().parse_args()
    train_val_test = {label: [] for label in BONN_LABELS.values()}
    n_split = 16

    # txtファイルをそれぞれpklにする。
    for set_ in BONN_LABELS.keys():
        save_path = Path(args.data_dir) / f'{set_}_pkl'
        save_path.mkdir(exist_ok=True)

        for txt_file in (Path(args.data_dir) / set_).iterdir():
            with open(txt_file, 'r') as f:
                values = np.array(list(map(float, f.read().split('\n')[:-1])))
            for i in range(n_split):
                if not values.shape[0] == 4097:
                    print(values.shape[0])
                    exit()
                duration = values.shape[0] // n_split
                eeg = EEG(values[i * duration:(i + 1) * duration].reshape((1, -1)), channel_list=['0'], len_sec=duration/173.6, sr=173.6)
                eeg.to_pkl(f'{save_path}/{txt_file.name[:-4]}_{i * duration}.pkl')
                train_val_test[BONN_LABELS[set_]].append(f'{save_path}/{txt_file.name[:-4]}_{i * duration}.pkl')

    # Manifestを作成. 各ラベルの8割をtrain、それ以外をvalにする
    train_list = []
    [train_list.extend(train_val_test[label][:int(len(train_val_test[label]) * 0.8)]) for label in train_val_test.keys()]
    pd.DataFrame(train_list).to_csv(Path(args.data_dir) / 'train_manifest.csv', index=False, header=None)

    test_list = []
    [test_list.extend(train_val_test[label][int(len(train_val_test[label]) * 0.8):]) for label in train_val_test.keys()]
    pd.DataFrame(test_list).to_csv(Path(args.data_dir) / 'test_manifest.csv', index=False, header=None)

    train_mani = pd.read_csv(Path(args.data_dir) / 'train_manifest.csv', header=None)
    train_mani.iloc[:int(train_mani.shape[0] * 0.8), :].to_csv(Path(args.data_dir) / 'train_manifest.csv',
                                                               index=False, header=None)
    train_mani.iloc[int(train_mani.shape[0] * 0.8):, :].to_csv(Path(args.data_dir) / 'val_manifest.csv',
                                                               index=False, header=None)
