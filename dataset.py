import torch
import os
import cv2
import random
import numpy as np
from skimage import io
from utils import convert_from_color, get_random_pos

EXTS = ['tif', 'tiff', 'jpg', 'png']


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_files, label_files, window_size, n_channels = 3, cache = True):

        self.data_files = data_files
        self.label_files = label_files
        self.window_size = window_size
        self.n_channels = n_channels
        self.cache = cache

        # If a directory is passed as input (data_files), load the path of the images from the directory.
        if not isinstance(self.data_files, list):
        
            assert os.path.isdir(self.data_files)

            # Loads and keeps only files with the allowed extensions (images).
            data_dir = self.data_files 
            all_files = os.listdir(data_dir)
            all_files = [f for f in all_files if os.path.splitext(f)[1] in EXTS]

            self.data_files = sorted([os.path.join(data_dir, f) for f in all_files])
        
        # If a directory is passed as input (data_files), load the path of the images from the directory.
        if not isinstance(self.label_files, list):

            assert os.path.isdir(self.label_files)
            
            label_dir = self.label_files
            all_files = os.listdir(label_dir)
            all_files = [f for f in all_files if os.path.splitext(f)[1] in EXTS]

            self.label_files = sorted([os.path.join(label_dir, f) for f in all_files])
        

        # Raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        

        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}
    


    def __len__(self):
        # Since random patches are extracted from the original images, the default epoch size is usually larger than len(self.data_files)
        # I set it to a maximum of 10,000 samples, but you can set an arbitrary number
        return 10000

    
    # Return data_files and label_files
    def get_dataset(self):
        return self.data_files, self.label_files
    

    def __getitem__(self, i):
        
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        
        else:
            # Data is normalized in [0, 1]
            if self.n_channels == 3:
                print(f"RGB -> ${self.data_files[random_idx]}")
                # image = cv2.cvtColor(cv2.imread(self.data_files[random_idx]), cv2.COLOR_BGR2RGB)
                # data = 1/255 * np.asarray(image.transpose((2,0,1)), dtype='float32')
                data = 1/255 * np.asarray(cv2.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            else:
                data = 1/255 * np.asarray(cv2.imread(self.data_files[random_idx]), dtype='float32')
                data = np.stack((data,)*3, axis=0) # if an single-channel image is passed, repeat that channel 3 times.
            
            if self.cache:
                self.data_cache_[random_idx] = data


        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]

        else: 
            # Labels are converted from RGB to their numeric values
            print(f"LABEL -> ${self.label_files[random_idx]}")
            label = np.asarray(convert_from_color(cv2.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, self.window_size)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))