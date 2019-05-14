from __future__ import print_function, division

from eeglibrary.src import train
from eeglibrary.utils import train_args
from src.utils import class_names


def label_func(path):
    return path[-8:-4]


if __name__ == '__main__':
    args = train_args().parse_args()
    train(args, class_names, label_func)
