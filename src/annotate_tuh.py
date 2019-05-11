import eeglibrary
import pyedflib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.const import *
from src.args import annotate_args


def annotate_method_1(signals, label_info, sr, duration, save_folder):
    mask = np.ones(signals.shape, dtype=bool)
    # 発作やノイズがある場合
    for info in label_info:
        start, end, label = info

        # 10の倍数でmaskすることで、あとで10秒ごとの分割をそのまま行えるようにする
        # e.g. from 0.00001 to 10.0 into 10, from 10.00001 to 20.0 into 20
        mask_start, mask_end = (int((float(start) - 0.00001) / 10) + 1) * 10, (
                    int((float(end) - 0.00001) / 10) + 1) * 10
        if float(start) == 0.0:
            mask_start = 0
        mask[:, mask_start * sr:mask_end * sr] = False
        # 10秒分切り出す。足りなければ保存しない
        for start_sec in np.arange(float(start), float(end), duration):
            if start_sec + duration > float(end):
                break
            start_idx, end_idx = int(start_sec * sr), int((start_sec + duration) * sr)

            # ファイルに保存
            filename = '{}_{}_{}.npy'.format(start_idx, end_idx, label)
            np.save(save_folder / filename, signals[:, start_idx:end_idx])
        print(start, end, label)

    null_array = np.array(np.ma.array(signals, mask=mask))
    assert null_array.shape[1] % sr == 0

    # 発作やノイズ以外の区間
    for start_idx in np.arange(0, null_array.shape[1], duration * sr):
        end_idx = start_idx + duration * sr
        if end_idx > null_array.shape[1]:
            break
        # ファイルに保存
        filename = '{}_{}_null.npy'.format(start_idx, end_idx)
        np.save(save_folder / filename, null_array[:, start_idx:end_idx])


def annotate_method_2(signals, label_info, sr, duration, save_folder):
    # label_info_list = [info.split() for info in label_info.split('\n')[2:-1]]
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
        filename = '{}_{}_{}.npy'.format(s_sec * sr, (s_sec + duration) * sr, true_label)
        np.save(save_folder / filename, signals[s_sec * sr:(s_sec + duration) * sr])


def annotate(label_path, args):
    edf_path = label_path.parent / str(label_path.name[:-7] + '.edf')

    with open(label_path, 'r') as f:
        label_info = f.read()
    label_info = [info.split()[:-1] for info in label_info.split('\n')[2:-1]]
    label_info = [(float(start), float(end), label) for start, end, label in label_info]

    edfreader = pyedflib.EdfReader(str(edf_path))

    sr = edfreader.getSampleFrequencies()[0]

    pat_id = edfreader.patient.decode().split()[0]

    n = edfreader.signals_in_file
    signal_labels = edfreader.getSignalLabels()
    signals = np.zeros((n, edfreader.getNSamples()[0]))
    for i in np.arange(n):
        try:
            signals[i, :] = edfreader.readSignal(i)
        except ValueError as e:
            np.delete(signals, i, 0)
    duration = 10

    save_folder = str(edf_path).split('/')
    save_folder[-2] = save_folder[-2] + '_' + save_folder[-1].replace('.edf', '').split('_')[-1]
    save_folder.pop(-1)
    save_folder.insert(-3, 'labeled')
    save_folder = Path('/'.join(save_folder))
    save_folder.mkdir(exist_ok=True, parents=True)

    if args.annotate_method == 1:
        annotate_method_1(signals, label_info, sr, duration, save_folder)
    elif args.annotate_method == 2:
        annotate_method_2(signals, label_info, sr, duration, save_folder)
    else:
        raise NotImplementedError


if __name__ == '__main__':

    args = annotate_args().parse_args()

    tuh_id_dirs = [p for p in Path(args.data_dir).iterdir() if p.is_dir()]
    for tuh_id_dir in tqdm(tuh_id_dirs):
        patient_dirs = [p for p in tuh_id_dir.iterdir() if p.is_dir()]
        for patient_dir in patient_dirs:
            records = [p for p in patient_dir.iterdir() if p.is_dir()]
            for record in records:
                for label_path in record.glob('*.tse_bi'):
                    annotate(label_path, args)
