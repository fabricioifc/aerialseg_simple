
import numpy as np
import random
import torch
import itertools

from sklearn.metrics import confusion_matrix

#### IMAGE UTILS ######

# ISPRS color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}


""" Numeric labels to RGB-color encoding """
def convert_to_color(arr_2d, palette=palette):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d



""" RGB-color encoding to grayscale labels """
def convert_from_color(arr_3d, palette=invert_palette):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def get_random_pos(img, window_shape):
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2



""" Slide a window_shape window across the image with a stride of step """
def sliding_window(top, step=10, window_size=(20,20)):
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
            


""" Count the number of windows in an image """
def count_sliding_window(top, step=10, window_size=(20,20)):
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c




#### TRAINER UTILS ####
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import torch.optim as optim


"""  Make optimizer routine"""
def make_optimizer(args, net):
    trainable = filter(lambda x: x.requires_grad, net.parameters()) # Only the parameters that requires gradient are passed to the optimizer

    if args['optimizer'] == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args['momentum']}
    elif args['optimizer'] == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args['beta1'], args['beta2']),
            'eps': args['epsilon']
        }
    elif args['optimizer'] == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args['epsilon']}

    kwargs['lr'] = args['lr']
    kwargs['weight_decay'] = args['weight_decay']
    
    return optimizer_function(trainable, **kwargs)



""" Make scheduler routine """
def make_scheduler(args, optimizer):
    if args['type'] == 'multi':
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=args['milestones'],
            gamma=args['gamma']
        )
    else:
        scheduler = lrs.StepLR(
            optimizer,
            step_size=args['lr_decay'],
            gamma=args['gamma']
        )
    return scheduler



""" 2D version of the cross entropy loss """
def CrossEntropy2d(input, target, weight=None, size_average=True):
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))



""" Acurracy metric formulation """
def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size



""" Browse an iterator by chunk of n elements """
def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def calculate_cm(predictions, labels, label_values = None, normalize = None):
    return confusion_matrix(labels, predictions, label_values, normalize=normalize)


""" Global acurracy metric calculation """
def global_accuracy(predictions, labels):
    # Calculate confusion matrix
    cm = calculate_cm(predictions, labels)
    # Sum all values in main diagonal
    main_diagonal = sum([cm[i][i] for i in range(len(cm))])
    # return TP+TN / TP+TN+FN x 100%
    return 100 * float(main_diagonal) / np.sum(cm)
