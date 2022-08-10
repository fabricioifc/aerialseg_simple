import os
import numpy as np
import cv2
import skimage.io as io
import torch
import matplotlib.pyplot as plt
from utils import convert_from_color, convert_to_color, get_random_pos

def check_torch():
    print(torch.__version__)
    print(torch.cuda.is_available())
    
def show_image(path='D:\\Projetos\\split\\saida\\images\\000000063.tif'):
    assert os.path.exists(path), True
    image = cv2.imread(path)
    print(image.shape)
    
    data = 1/255 * np.asarray(image.transpose((2,0,1)), dtype='float32')
    x1, x2, y1, y2 = get_random_pos(data, (256, 256))
    data_p = data[:, x1:x2,y1:y2]
    print(data_p.shape)
    print(data_p.transpose((2,1,0)).shape)
    
    plt.imshow(torch.from_numpy(data_p.transpose((2,1,0))))
    plt.show()

if __name__ == '__main__':
    check_torch()
    show_image()