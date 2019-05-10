import eeglibrary
import pyedflib
import numpy as np
from pathlib import Path
from src.const import *
from src.args import annotate_args


def annotate(edf_path, label_path, args):

    with open(label_path, 'r') as f:
        label_info = f.read()

    edfreader = pyedflib.EdfReader(str(edf_path))

    sr = edfreader.getSampleFrequencies()[0]

    pat_id = edfreader.patient.decode().split()[0]

    n = edfreader.signals_in_file
    signal_labels = edfreader.getSignalLabels()
    sigbufs = np.zeros((n, edfreader.getNSamples()[0]))
    for i in np.arange(n):
        try:
            sigbufs[i, :] = edfreader.readSignal(i)
        except ValueError as e:
            np.delete(sigbufs, i, 0)
    duration = 10
    save_folder = Path(str(edf_path.parent).replace('data', 'labeled_data'))
    save_folder.mkdir(exist_ok=True, parents=True)

    mask = np.ones(sigbufs.shape, dtype=bool)
    # 発作やノイズがある場合
    for info in label_info.split('\n')[2:-1]:
        start, end, label, _ = info.split()

        # 10の倍数でmaskすることで、あとで10秒ごとの分割をそのまま行えるようにする
        # e.g. from 0.00001 to 10.0 into 10, from 10.00001 to 20.0 into 20
        mask_start, mask_end = (int((float(start)-0.00001) / 10) + 1) * 10, (int((float(end)-0.00001) / 10) + 1) * 10
        if float(start) == 0.0:
            mask_start = 0
        mask[:, mask_start*sr:mask_end*sr] = False
        # 10秒分切り出す。足りなければ保存しない
        for start_sec in np.arange(float(start), float(end), duration):
            if start_sec + duration > float(end):
                break
            start_idx, end_idx = int(start_sec*sr), int((start_sec+duration)*sr)

            # ファイルに保存
            filename = '{}_{}_{}.npy'.format(start_idx, end_idx, label)
            np.save(save_folder / filename, sigbufs[:, start_idx:end_idx])
        print(start, end, label)

    null_array = np.array(np.ma.array(sigbufs, mask=mask))
    assert null_array.shape[1] % sr == 0

    # 発作やノイズ以外の区間
    for start_idx in np.arange(0, null_array.shape[1], duration*sr):
        end_idx = start_idx + duration*sr
        if end_idx > null_array.shape[1]:
            break
        # ファイルに保存
        filename = '{}_{}_null.npy'.format(start_idx, end_idx)
        np.save(save_folder / filename, null_array[:, start_idx:end_idx])


if __name__ == '__main__':

    args = annotate_args().parse_args()

    tuh_id_dirs = [p for p in Path(args.patients_dir).iterdir() if p.is_dir()]
    for tuh_id_dir in tuh_id_dirs:
        patient_dirs = [p for p in tuh_id_dir.iterdir() if p.is_dir()]
        for patient_dir in patient_dirs:
            records = [p for p in patient_dir.iterdir() if p.is_dir()]
            for record in records:
                for label_path in record.glob('*.tse_bi'):
                    edf_path = label_path.parent / str(label_path.name[:-7]+'.edf')
                    annotate(edf_path, label_path, args)
