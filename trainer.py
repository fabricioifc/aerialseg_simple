import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from utils import make_optimizer, make_scheduler, CrossEntropy2d, accuracy, sliding_window, grouper, convert_from_color, convert_to_color, global_accuracy


class Trainer():
    
    def __init__(self, net, loader, params, scheduler = True, cbkp = None,):
        
        self.net = net
        self.loader = loader
        self.params = params

        # Define an id to a trained model. Use the number of seconds since 1970
        time_ = str(time.time())
        time_ = time_.replace(".", "")
        self.model_id = time_

        # Create optimizer
        self.optimizer = make_optimizer(self.params['optimizer_params'], self.net)
        # Create scheduler
        self.scheduler = make_scheduler(self.params['lrs_params'], self.optimizer) if scheduler else None

        self.last_loss = 0.0
        self.last_epoch = 0.0

        # Load a previously model if it exists
        if cbkp is not None:
            self.load(cbkp)
    

    def load(self, path):
        
        # Check if model file exists
        assert os.path.exists(path), "{} cant be opened".format(path)
        
        checkpoint = torch.load(path)

        self.last_epoch = checkpoint['epoch']
        self.last_loss = checkpoint['loss']
        self.model_id = checkpoint['model_id']
        
        # Load model and optimizer params
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None:
            for _ in range(0, self.last_epoch): self.scheduler.step()
    


    def save(self, path = None):

        if path is None:
            path = './{}_model_final.pth.tar'.format(self.model_id)

        # Save current loss, epoch, model weights and optimizer params
        torch.save({
            'epoch': self.last_epoch,
            'loss': self.last_loss,
            'model_id': self.model_id,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    

    def prepare(self, l, volatile=False):
        
        device = torch.device('cpu' if self.params['cpu'] is not None else 'cuda') # Define run-device torch
        def _prepare(tensor):
            if self.params['precision'] == 'half': tensor = tensor.half() # Convert to half precision
            return tensor.to(device)
        return [_prepare(_l) for _l in l]
    



    def train(self):
        
        running_loss = 0.0
        print_each = 500 # Print statistics every 500 iterations

        # Weights for class balancing
        weight_cls = self.prepare([self.params['weights']])
        
        if self.scheduler is not None:
            epoch = self.scheduler.last_epoch
        else:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
            
        self.net.train()

        for batch_id, (inputs, labels) in enumerate(self.loader['train']):

            inputs, labels = self.prepare([inputs, labels]) # Prepare input and labels 
            self.optimizer.zero_grad() # Set the gradients to zero

            outputs = self.net(inputs) # Forward step 
            loss = CrossEntropy2d(outputs, labels, weight_cls[0]) # Calculate the loss function
            loss.backward() # Compute the gradients

            self.optimizer.step() # Update network weigths
        
            running_loss += loss.item()
            if batch_id % print_each == print_each - 1:
                targets = labels.data.cpu().numpy()[0] # Labels
                predictions = outputs.data.cpu().numpy()[0] # Extract network outputs
                predictions = np.argmax(predictions, axis=0) # Get predictions

                acc = accuracy(predictions, targets)

                print('Epoch {} Iter {} Running Loss: {:.4f} Actual Loss: {:.4f} Running Acc: {:.4f}'.format(epoch + 1, batch_id + 1, running_loss / print_each, loss.item(), acc))
                self.last_loss = running_loss
                
                running_loss = 0.0

        return self.last_loss




    def test(self, test_loader = None, stride = None, window_size = None, batch_size = None, output_masks = False):
    
        # Loading test parameters
        test_ld = test_loader if test_loader is not None else self.loader['test']
        assert test_ld is not None, "Test_loader can't be None"

        test_stride = stride if stride is not None else self.params['stride']
        assert test_stride is not None, "Stride not set"

        test_ws = window_size if window_size is not None else self.params['window_size']
        assert test_ws is not None, "Window size not set"

        bs = batch_size if batch_size is not None else self.params['bs']
        bs = bs if bs is not None else 1

        # Get all data in test data loader
        test_inputs, test_labels = test_ld.dataset.get_dataset()
        test_inputs_name = np.copy(test_inputs)
        test_inputs = np.array([1/255 * np.asarray(io.imread(inputs), dtype='float32') for inputs in test_inputs])
        test_labels = np.array([np.asarray(convert_from_color(io.imread(label)), dtype='int64') for label in test_labels])

        all_predictions = []
        all_labels = []

        self.net.eval()

        accuracy = 0.0

        with torch.no_grad():

            for batch_id, (inputs, label) in enumerate(zip(test_inputs, test_labels)):
                predictions = np.zeros(inputs.shape[:2] + (self.params['n_classes'],))

                for idx, coords in enumerate(grouper(bs, sliding_window(inputs, step=test_stride, window_size=test_ws))):
                    input_patches = [np.copy(inputs[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
                    input_patches = np.asarray(input_patches)

                    input_patches = self.prepare([torch.from_numpy(input_patches)])[0]

                    outputs = self.net(input_patches)
                    outputs = outputs.data.cpu().numpy()

                    for out, (x, y, w, h) in zip(outputs, coords):
                        out = out.transpose((1,2,0))
                        predictions[x:x+w, y:y+h] += out

                predictions = np.argmax(predictions, axis=-1)

                # Output masks in a file
                if output_masks:
                    plt.imsave(self.model_id + os.path.basename(test_inputs_name[batch_id]) + "_label.png", convert_to_color(predictions))
                
                all_predictions.append(predictions)
                all_labels.append(label)

            accuracy = global_accuracy(np.concatenate([p.ravel() for p in all_predictions]), np.concatenate([p.ravel() for p in all_labels]).ravel())
        return accuracy
    



