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

def preprocess_image(img,final_img_size,img_size =(170, 242),input_size =(150, 220)):
    img = img.astype(np.uint8)
    resized = resize_image(255 - normalize_image(img, final_img_size), img_size)
    if input_size is not None and input_size != img_size:
        cropped = crop_center(resized, input_size)
    else:
        cropped = resized
    return cropped

def normalize_image(img,final_img_size = (1000, 1400)):
    blur_radius = 2
    blurred_image = filters.gaussian(img, blur_radius, preserve_range=True)

    threshold = filters.threshold_otsu(img)

    binarized_image = blurred_image > threshold
    rows, columns = np.where(binarized_image == 0)
    row_com = int(rows.mean() - rows.min())
    column_com = int(columns.mean() - columns.min())

    cropped = img[rows.min(): rows.max(), columns.min(): columns.max()]

    img_rows, img_cols = cropped.shape
    max_rows, max_cols = final_img_size

    row_start = max_rows // 2 - row_com
    column_start = max_cols // 2 - column_com

    if img_rows > max_rows:
        row_start = 0
        difference = img_rows - max_rows
        crop_start = difference // 2
        cropped = cropped[crop_start:crop_start + max_rows, :]
        img_rows = max_rows
    else:
        extra_rows = (row_start + img_rows) - max_rows
        if extra_rows > 0:
            row_start = row_start - extra_rows
        if row_start < 0:
            row_start = 0

    if img_cols > max_cols:
        column_start = 0
        difference = img_cols - max_cols
        crop_start = difference // 2
        cropped = cropped[:, crop_start:crop_start + max_cols]
        img_cols = max_cols
    else:
        extra_columns = (column_start + img_cols) - max_cols
        if extra_columns > 0:
            column_start = column_start - extra_columns
        if column_start < 0:
            column_start = 0
    normalized_image = np.ones((max_rows, max_cols), dtype=np.uint8) * 255
    normalized_image[row_start:row_start + img_rows, column_start:column_start + img_cols] = cropped
    normalized_image[normalized_image > threshold] = 255

    return normalized_image


def resize_image(img,size):
    height, width = size
    width_ratio = float(img.shape[1]) / width
    height_ratio = float(img.shape[0]) / height
    if width_ratio > height_ratio:
        resize_height = height
        resize_width = int(round(img.shape[1] / height_ratio))
    else:
        resize_width = width
        resize_height = int(round(img.shape[0] / width_ratio))

    img = transform.resize(img, (resize_height, resize_width),
                           mode='constant', anti_aliasing=True, preserve_range=True)

    img = img.astype(np.uint8)
    
    if width_ratio > height_ratio:
        start = int(round((resize_width-width)/2.0))
        return img[:, start:start + width]
    else:
        start = int(round((resize_height-height)/2.0))
        return img[start:start + height, :]


def crop_center(img, size):
    img_shape = img.shape
    start_y = (img_shape[0] - size[0]) // 2
    start_x = (img_shape[1] - size[1]) // 2
    cropped = img[start_y: start_y + size[0], start_x:start_x + size[1]]
    return cropped

def load_image(path):
    return img_as_ubyte(imread(path, as_gray=True))

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

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
        input1 = input1.view(-1, 1, 150, 220).float().div(255)
        input2 = input2.view(-1, 1, 150, 220).float().div(255)

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
    img1 = preprocess_signature(img1, (1000, 1400), (256, 256))
    img2 = preprocess_signature(img2, (1000, 1400), (256, 256))
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