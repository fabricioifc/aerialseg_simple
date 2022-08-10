import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet, self).__init__()
        
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_1 = nn.Conv2d(self.n_features, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_1 = nn.Conv2d(self.n_features*2, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_1 = nn.Conv2d(self.n_features*4, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        self.conv5_1 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features*8)

        self.conv5_3_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_1_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features*8)

        self.conv4_3_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*4, self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*2, self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features, self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv1_2_D = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)

        # Decoder block 5
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))
        
        # Decoder block 4
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x