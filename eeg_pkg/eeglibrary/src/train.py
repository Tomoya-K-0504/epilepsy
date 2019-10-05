from __future__ import print_function, division

import time
from pathlib import Path

import numpy as np
import torch
# from wrapper.models import adda
from eeglibrary.src import test
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder


def train_model(model, inputs, labels, phase, optimizer, criterion, type='nn', classes=None):
    if 'nn' in type:
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, 1)
    else:
        inputs, labels = inputs.data.numpy(), labels.data.numpy()
        if phase == 'train':
            model.partial_fit(inputs, labels)
        preds = model.predict(inputs)
        enc = OneHotEncoder(handle_unknown='ignore').fit(np.array([0, 1, 2]).reshape(-1, 1))
        # logloss of skearn is reverse argment order compared with pytorch criterion
        loss = criterion(labels, enc.transform(preds.reshape(-1, 1)).toarray(), labels=classes)

    return preds, loss


def save_model(model, model_path, numpy):
    if numpy:
        model.save_model(model_path)
    else:
        torch.save(model.state_dict(), model_path)


def update_by_epoch(args, metrics, phase, model, numpy, optimizer):
    for metric in metrics:
        best_flag = metric.average_meter[phase].update_best()
        # save model

        if metric.save_model and best_flag and phase == 'val':
            print("Found better validated model, saving to %s" % args.model_path)
            save_model(model, args.model_path, numpy)

        # reset epoch average meter
        metric.average_meter[phase].reset()

    # anneal lr
    if phase == 'train' and (not numpy):
        param_groups = optimizer.param_groups
        for g in param_groups:
            g['lr'] = g['lr'] / args.learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))


def record_log(logger, phase, metrics, epoch):
    values = {}
    for metric in metrics:
        values[phase + '_' + metric.name] = metric.average_meter[phase].average
    logger.update(epoch, values)


def train(args, class_names, label_func, metrics):
    init_seed(args)
    Path(args.model_path).parent.mkdir(exist_ok=True, parents=True)

    if args.tensorboard:
        tensorboard_logger = TensorBoardLogger(args.log_id, args.log_dir, args.log_params)

    start_epoch, start_iter, optim_state = 0, 0, None
    # far; False alarm rate = 1 - specificity
    best_loss, best_far = {}, {}
    for phase in ['train', 'val']:
        best_loss[phase], best_far[phase] = 1000, 1.0

    # init setting
    classes = [i for i in range(len(class_names))]
    device = init_device(args)
    eeg_conf = set_eeg_conf(args)
    model = set_model(args, classes, eeg_conf, device)
    dataloaders = {phase: set_dataloader(args, eeg_conf, class_names, phase, label_func, device='cpu')
                   for phase in ['train', 'val']}

    if 'nn' in args.model_name:
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=args.lr)
        args.weight = list(map(float, args.loss_weight.split('-')))
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.weight).to(device))
        numpy = False
    else:
        optimizer = None
        criterion = log_loss
        numpy = True

    batch_time = AverageMeter()
    execute_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()

        for phase in ['train', 'val']:
            print('\n{} phase started.'.format(phase))

            epoch_preds = torch.empty((len(dataloaders[phase])*args.batch_size, 1), dtype=torch.int64, device=device)
            epoch_labels = torch.empty((len(dataloaders[phase])*args.batch_size, 1), dtype=torch.int64, device=device)

            if numpy:
                epoch_preds, epoch_labels = epoch_preds.data.numpy(), epoch_labels.data.numpy()

            start_time = time.time()
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                # break
                data_load_time = time.time() - start_time
                # print('data loading time', data_load_time)

                # feature scaling
                if args.scaling:
                    inputs = (inputs - 100).div(600)

                preds, loss_value = train_model(model, inputs, labels, phase, optimizer, criterion, args.model_name,
                                                classes)

                epoch_preds[i * args.batch_size:(i + 1) * args.batch_size, 0] = preds
                epoch_labels[i * args.batch_size:(i + 1) * args.batch_size, 0] = labels

                # save loss and recall in one batch
                for metric in metrics:
                    metric.update(phase, loss_value, inputs.size(0), preds, labels, classes, numpy)

                if not args.silent:
                    print('Epoch: [{0}][{1}/{2}]'.format(epoch, i+1, len(dataloaders[phase])), end='\t')
                    print('Time {batch_time.value:.3f}'.format(batch_time=batch_time), end='\t')
                    for metric in metrics:
                        print('{} {:.3f}'.format(metric.name, metric.average_meter[phase].value), end='\t')
                    print('')

                # measure elapsed time
                batch_time.update(time.time() - start_time)
                start_time = time.time()

            if args.tensorboard:
                record_log(tensorboard_logger, phase, metrics, epoch)
            update_by_epoch(args, metrics, phase, model, numpy, optimizer)

    if args.adda:
        adda(args, model, eeg_conf, label_func, class_names, criterion, device,
             source_manifest=args.train_manifest, target_manifest=args.val_manifest)

    if args.silent:
        print(best_loss['val'].item())
    else:
        print('execution time was {}'.format(time.time() - execute_time))

    if args.test:
        # test phase
        test(args, model, eeg_conf, label_func, class_names, numpy, device)

    if args.inference:
        # inference phase
        return inference(args, model, eeg_conf, numpy, device)


if __name__ == '__main__':
    # This is for kaggle seizure prediction data

    from wrapper.src.metrics import Metric
    metrics = [
        Metric('loss', initial_value=10000, inequality='less', save_model=True),
        Metric('accuracy', initial_value=0, inequality='more'),
    ]

    class_names = ['interictal', 'preictal']

    def load_func():
        pass

    def label_func(path):
        return path.split('/')[-2].split('_')[2]

    data_conf = {
        'load_func': load_func,
        'labels': class_names,
        'label_func': label_func,
    }

    args = train_args().parse_args()

    train(args, class_names, lambda x: x, metrics)
