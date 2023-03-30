from __future__ import absolute_import
import argparse, random, copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from skimage import filters, transform
from skimage.io import imread
from skimage import img_as_ubyte
from typing import Tuple

import pandas as pd

from sagemaker_inference import decoder

def preprocess_image(image, radius=2, resize_dims = (50,70)):
    image = image.astype(np.uint8)
    blurred_image = filters.gaussian(image, radius, preserve_range=True)
    threshold = filters.threshold_otsu(blurred_image)
    binarized_image = blurred_image > threshold
    image_background_removed = np.where(binarized_image==False, 255-image, 0)
    image_background_removed = transform.resize(image_background_removed, resize_dims, mode='constant', anti_aliasing=True, preserve_range=True)
    image_background_removed = image_background_removed /255.0
    image_background_removed = np.pad(image_background_removed, (20, 20), 'constant', constant_values=(0,0))
    return image_background_removed

class SiameseNetwork(nn.Module):
    """
        modified version of an official pytorch example Siamese nn model using resnet18
        https://github.com/pytorch/examples/blob/main/siamese_network/main.py
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc = nn.Sequential(
            nn.Linear(2*self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1))
        self.sigmoid = nn.Sigmoid()
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        input1 = input1.view(-1, 1, 90, 110).float()
        input2 = input2.view(-1, 1, 90, 110).float()
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output


def model_fn(model_dir):
    device = torch.device('cpu')
    model = SiameseNetwork()
    with open(os.path.join(model_dir, 'siamese_network.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.eval()
    return model

def input_fn(input_data, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats.
    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type
    Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor,
        depending if cuda is available.
    """
    device = torch.device("cpu")
    np_array = decoder.decode(input_data, content_type)
    img1 = np_array[0]
    img2 = np_array[1]
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)
    tensor1 = torch.tensor(img1).unsqueeze(0)
    tensor2 = torch.tensor(img2).unsqueeze(0)
    return [tensor1.to(device), tensor2.to(device)] 

def predict_fn(data, model):
    """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
    Runs prediction on GPU if cuda is available.
    Args:
        data: input data (torch.Tensor) for prediction deserialized by input_fn
        model: PyTorch model loaded in memory by model_fn
    Returns: a prediction
    """
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_data1, input_data2 = data[0].to(device), data[1].to(device)
        model.eval()
        output = model(input_data1, input_data2).squeeze()
    return output