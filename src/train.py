from __future__ import print_function, division

from eeglibrary.src import train
from eeglibrary.src import Metric
from eeglibrary.utils import train_args
from src.utils import class_names


def label_func(path):
    return path[-8:-4]


if __name__ == '__main__':
    args = train_args().parse_args()
    metrics = [
        Metric('loss', initial_value=10000, inequality='less', save_model=True),
        Metric('confusion_matrix', initial_value=0, inequality='more')]
    train(args, class_names, label_func, metrics)

