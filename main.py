import os
import torch
import numpy as np
import pandas as pd

from dataset import Dataset
from trainer import Trainer
from models import SegNet

if __name__=='__main__':

    # Params
    params = {
        'root_dir': '/home/fabricio/datasets/GTA-V-SID/500x500',
        'window_size': (250, 250),
        'cache': True,
        'bs': 8,
        'n_classes': 2,
        'cpu': None,
        'precision' : 'full',
        'optimizer_params': {
            'optimizer': 'SGD',
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0005
        },
        'lrs_params': {
            'type': 'multi',
            'milestones': [25, 35, 45],
            'gamma': 0.1
        },
        'weights': '',
        'maximum_epochs': 20,
    }

    params['weights'] = torch.ones(params['n_classes']) 

    image_dir = os.path.join(params['root_dir'], 'slice')
    label_dir = os.path.join(params['root_dir'], 'label')

    # Load image and label files from .txt
    train_images = pd.read_table('train_images.txt',header=None).values
    train_images = [os.path.join(image_dir, f[0]) for f in train_images]
    train_labels = pd.read_table('train_labels.txt',header=None).values
    train_labels = [os.path.join(label_dir, f[0]) for f in train_labels]

    test_images = pd.read_table('test_images.txt',header=None).values
    test_images = [os.path.join(image_dir, f[0]) for f in test_images]
    test_labels = pd.read_table('test_labels.txt',header=None).values
    test_labels = [os.path.join(label_dir, f[0]) for f in test_labels]

    # Create train and test sets
    train_dataset = Dataset(train_images, train_labels, window_size = params['window_size'])
    test_dataset = Dataset(test_images, test_labels, window_size = params['window_size'])

    # Load dataset classes in pytorch dataloader handler object
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = params['bs'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = params['bs'])

    # Load network model in cuda (gpu)
    model = SegNet(in_channels = 3, out_channels = params['n_classes'])
    model.cuda()


    loader = {
        "train": train_loader,
        "test": test_loader,
    }

    trainer = Trainer(model, loader, params)
    print(trainer.test(stride = 16, output_masks = True))
    for epoch in range(params['maximum_epochs']):
        train_metric = trainer.train()
    trainer.save()
    print(trainer.test(stride = 16, output_masks = True))