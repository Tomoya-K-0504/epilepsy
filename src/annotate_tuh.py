import eeglibrary
import pyedflib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from eeglibrary import EEG
from src.const import *
from src.args import annotate_args


def annotate_method_1(eeg, label_info, duration, save_folder):
    signals = np.copy(eeg.values)
    mask = np.ones(signals.shape, dtype=bool)
    paths = []

    # 発作やノイズがある場合
    for info in label_info:
        start, end, label = info

        # 10の倍数でmaskすることで、あとで10秒ごとの分割をそのまま行えるようにする
        # e.g. from 0.00001 to 10.0 into 10, from 10.00001 to 20.0 into 20
        mask_start, mask_end = (int((float(start) - 0.00001) / 10) + 1) * 10, (
                    int((float(end) - 0.00001) / 10) + 1) * 10
        if float(start) == 0.0:
            mask_start = 0
        mask[:, mask_start * eeg.sr:mask_end * eeg.sr] = False
        # 10秒分切り出す。足りなければ保存しない
        for start_sec in np.arange(float(start), float(end), duration):
            if start_sec + duration > float(end):
                break
            start_idx, end_idx = int(start_sec * eeg.sr), int((start_sec + duration) * eeg.sr)

            # ファイルに保存
            filename = '{}_{}_{}.pkl'.format(start_idx, end_idx, label)
            eeg.values = signals[:, start_idx:end_idx]
            eeg.to_pkl(save_folder / filename)
            paths.append(str(save_folder / filename))
        # print(start, end, label)

    null_array = np.array(np.ma.array(signals, mask=mask))
    assert null_array.shape[1] % eeg.sr == 0

    # 発作やノイズ以外の区間
    for start_idx in np.arange(0, null_array.shape[1], duration * eeg.sr):
        end_idx = start_idx + duration * eeg.sr
        if end_idx > null_array.shape[1]:
            break
        # ファイルに保存
        filename = '{}_{}_null.pkl'.format(start_idx, end_idx)
        eeg.values = null_array[:, start_idx:end_idx]
        if eeg.values.size < 32 * eeg.sr * duration:
            continue
        eeg.to_pkl(save_folder / filename)
        paths.append(str(save_folder / filename))

    return paths


def annotate_method_2(eeg, label_info, duration, save_folder):
    # label_info_list = [info.split() for info in label_info.split('\n')[2:-1]]
    signals = np.copy(eeg.values)
    paths = []

    for s_sec in np.arange(0, 400, duration):
        true_label = 'null'
        for info in label_info:
            start, end, label = info
            # true_labelを決定する。半分以上の時間を占めているラベルとする
            if s_sec >= start and s_sec + duration <= end:
                true_label = label
            elif s_sec >= start and s_sec + duration >= end and end - s_sec >= 5:
                true_label = label
            elif s_sec <= start and s_sec + duration <= end and s_sec + duration - start >= 5:
                true_label = label
            elif s_sec <= start and s_sec + duration >= end and end - start >= 5:
                true_label = label
        # print(s_sec, s_sec + duration, true_label)
        # ファイルに保存
        filename = '{}_{}_{}.pkl'.format(s_sec * eeg.sr, (s_sec + duration) * eeg.sr, true_label)
        eeg.values = np.copy(signals[:, s_sec * eeg.sr:(s_sec + duration) * eeg.sr])
        if eeg.values.size < 32 * eeg.sr * duration:
            continue
        eeg.to_pkl(save_folder / filename)
        paths.append(str(save_folder / filename))

    return paths


def annotate(label_path, args):
    edf_path = label_path.parent / str(label_path.name[:-7] + '.edf')

    with open(label_path, 'r') as f:
        label_info = f.read()
    label_info = [info.split()[:-1] for info in label_info.split('\n')[2:-1]]
    label_info = [(float(start), float(end), label) for start, end, label in label_info]

    edfreader = pyedflib.EdfReader(str(edf_path))

    save_folder = str(edf_path).split('/')
    save_folder[-3] = '_'.join([save_folder[-3], save_folder[-2], save_folder[-1].replace('.edf', '').split('_')[-1]])
    save_folder = save_folder[:-2]
    save_folder.insert(-2, 'method_{}_labeled'.format(args.annotate_method))
    save_folder = Path('/'.join(save_folder))
    save_folder.mkdir(exist_ok=True, parents=True)

    eeg = EEG.from_edf(edfreader)
    eeg.len_sec = args.duration
    # 値が入っていないチャンネルが存在するため、削除
    if ~np.all(~np.all(eeg.values == 0, axis=1)):
        mask_index = np.where(np.all(eeg.values == 0, axis=1))
        [eeg.channel_list.pop(i) for i in np.flip(mask_index)[0]]
        eeg.values = eeg.values[~np.all(eeg.values == 0, axis=1)]

    # すべて0のチャンネルが存在する場合、データとして使用しない
    if ~np.any(~np.all(eeg.values == 0, axis=1)):
        return []

    eeg.values = eeg.values[:32, :]
    eeg.channel_list = eeg.channel_list[:32]

    if args.annotate_method == 1:
        return annotate_method_1(eeg, label_info, args.duration, save_folder)
    elif args.annotate_method == 2:
        return annotate_method_2(eeg, label_info, args.duration, save_folder)
    else:
        raise NotImplementedError


def annotate_source_target(args):
    manifests = list(Path(args.data_dir).glob('*.csv'))
    for manifest in manifests:
        phase = manifest.name.replace('_manifest.csv', '')
        with open(manifest, 'r') as f:
            paths = f.readlines()
        pwise_paths = {}
        for path in paths:
            patient_id = path.split('/')[-2].split('_')[0]
            if not patient_id in pwise_paths.keys():
                pwise_paths[patient_id] = []
            pwise_paths[patient_id].append(path)
        
        source_patients = list(pwise_paths.keys())[:len(pwise_paths) * 8 // 10]
        target_patients = list(pwise_paths.keys())[len(pwise_paths) * 8 // 10:]
        for s_or_t, patients in zip(['source', 'target'], [source_patients, target_patients]):
            save_paths = []
            [save_paths.extend(pwise_paths[key]) for key in patients]
            with open(f'{args.data_dir}/{phase}_{s_or_t}.csv', 'w') as f:
                f.write(''.join(save_paths))


if __name__ == '__main__':

    args = annotate_args().parse_args()

    if args.source_target:
        annotate_source_target(args)
        exit()

    n_tqdm = sum([len(list(p.iterdir())) for p in Path(args.data_dir).iterdir() if p.name in ['train', 'dev_test']])
    print('There are {} folders to count by tdqm'.format(n_tqdm))
    for train_test_dir in [p for p in Path(args.data_dir).iterdir() if p.name in ['train', 'dev_test']]:
        paths = []
        for data_config_dir in [p for p in train_test_dir.iterdir() if p.is_dir()]:
            for tuh_id_dir in tqdm([p for p in data_config_dir.iterdir() if p.is_dir()]):
                for patient_dir in [p for p in tuh_id_dir.iterdir() if p.is_dir()]:
                    for record in [p for p in patient_dir.iterdir() if p.is_dir()]:
                        for label_path in record.glob('*.tse_bi'):
                            paths.extend(annotate(label_path, args))

        # Manifestを作成
        pd.DataFrame(paths).to_csv(Path(args.data_dir) / '{}_manifest.csv'.format(train_test_dir.name),
                                   index=False, header=None)
    train_mani = pd.read_csv(Path(args.data_dir) / 'train_manifest.csv', header=None)
    train_mani.iloc[:int(train_mani.shape[0] * 0.8), :].to_csv(Path(args.data_dir) / 'train_manifest.csv',
                                                               index=False, header=None)
    train_mani.iloc[int(train_mani.shape[0] * 0.8):, :].to_csv(Path(args.data_dir) / 'val_manifest.csv',
                                                               index=False, header=None)

