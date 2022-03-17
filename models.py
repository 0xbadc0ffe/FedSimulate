from __future__ import print_function, division

from typing import OrderedDict, Union, Optional, Callable, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


import ssl
ssl._create_default_https_context = ssl._create_unverified_context



def get_CIFARloaders(batch_size=128, batch_size_val=1000, data_transform: Union[str, bool]="RGB", ret_datasets=False) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    if data_transform:
        image_transforms_gray = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.47,), std=(0.251,)),
            ]
        )

        image_transforms_RGB = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.47,), std=(0.251,)),
            ]
        )
        if data_transform == "RGB":
            image_transforms = image_transforms_RGB
        elif data_transform == "GRAY":
            image_transforms = image_transforms_gray
    else:
        image_transforms = None

    train_dataset = datasets.CIFAR10(
            "data",
            train=True,
            download=True,
            transform=image_transforms
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )


    test_dataset = datasets.CIFAR10(
        "data",
        train=False,
        transform=image_transforms
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_val,
        shuffle=True,
    )
    
    if ret_datasets:
        return train_loader, test_loader, train_dataset, test_dataset
    
    else:
        return train_loader, test_loader        


# Specify the number of classes in CIFAR10
CIFAR10_output_size = 10  # there are 10 classes
CIFAR10_output_classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck')


class MLP(nn.Module):
    def __init__(
        self, input_size: int, input_channels: int, n_hidden: int, output_size: int
    ) -> None:
        """
        Simple MLP model

        :param input_size: number of pixels in the image
        :param input_channels: number of color channels in the image
        :param n_hidden: size of the hidden dimension to use
        :param output_size: expected size of the output
        """
        super().__init__()
        self.name= "MLP"
        self.network = nn.Sequential(
            nn.Linear(input_size * input_channels, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: batch of images with size [batch, 1, w, h]

        :returns: predictions with size [batch, output_size]
        """
        x = x.view(x.shape[0], -1)
        o = self.network(x)
        return o


class CNN(nn.Module):
    def __init__(
        self, input_size: int, input_channels: int, n_feature: int, output_size: int
    ) -> None:
        """
        Simple model that uses convolutions

        :param input_size: number of pixels in the image
        :param input_channels: number of color channels in the image
        :param n_feature: size of the hidden dimensions to use
        :param output_size: expected size of the output
        """
        super().__init__()
        self.name="CNN"
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=n_feature, kernel_size=3
        )
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=3)
        self.conv3 = nn.Conv2d(n_feature, n_feature, kernel_size=3)
        self.conv4 = nn.Conv2d(n_feature, n_feature, kernel_size=2)

        self.fc1 = nn.Linear(n_feature * 5 * 5, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, 
                x: torch.Tensor, 
                return_conv1: bool = False, 
                return_conv2: bool = False, 
                return_conv3: bool = False,
                return_conv4: bool = False
        ) -> torch.Tensor:
        """
        :param x: batch of images with size [batch, 1, w, h]
        :param return_conv1: if True return the feature maps of the first convolution
        :param return_conv2: if True return the feature maps of the second convolution
        :param return_conv3: if True return the feature maps of the third convolution

        :returns: predictions with size [batch, output_size]
        """
        x = self.conv1(x)
        if return_conv1:
            return x
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        if return_conv2:
            return x
        x = F.relu(x)

        # Not so easy to keep track of shapes... right?
        # An useful trick while debugging is to feed the model a fixed sample batch
        # and print the shape at each step, just to be sure that they match your expectations.

        # print(x.shape)

        x = self.conv3(x)
        if return_conv3:
            return x
        x = F.relu(x)
        #x = F.max_pool2d(x, kernel_size=2) # comment if add conv4

        
        x = self.conv4(x)
        if return_conv4:
            return x
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def permute_pixels(images: torch.Tensor, perm: Optional[torch.Tensor]) -> torch.Tensor:
    """ Permutes the pixel in each image in the batch

    :param images: a batch of images with shape [batch, channels, w, h]
    :param perm: a permutation with shape [w * h]

    :returns: the batch of images permuted according to perm
    """
    if perm is None:
        return images

    batch_size = images.shape[0]
    n_channels = images.shape[1]
    w = images.shape[2]
    h = images.shape[3]
    images = images.view(batch_size, n_channels, -1)
    images = images[..., perm]
    images = images.view(batch_size, n_channels, w, h)
    return images



