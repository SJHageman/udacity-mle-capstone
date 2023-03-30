import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_curve

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from skimage import filters, transform
from skimage.io import imread
from skimage import img_as_ubyte
from typing import Tuple

import pandas as pd


def load_image(fname):
    return img_as_ubyte(imread(fname, as_gray=True))

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

class SignatureDataset(Dataset):
    def __init__(self, category,  resize_dims = (50,70)):
        df = pd.read_csv(f'/opt/ml/input/data/meta/{category}_data.csv', header=0, names=['img_path_real', 'img_path_forged', 'label'])
        df['img_path_real'] = df['img_path_real'].apply(lambda x : f'/opt/ml/input/data/{category}/{x}')
        df['img_path_forged'] = df['img_path_forged'].apply(lambda x : f'/opt/ml/input/data/{category}/{x}')
        self.df  = df.sample(frac=0.1)
        self.real_fnames = self.df["img_path_real"].values
        self.forged_fnames = self.df["img_path_forged"].values
        self.labels = self.df["label"].values
        self.resize_dims=resize_dims

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self,index):
        img1 = load_image(self.real_fnames[index])
        img2 = load_image(self.forged_fnames[index])
        img1 = preprocess_image(img1, self.resize_dims)
        img2 = preprocess_image(img2, self.resize_dims)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return torch.tensor(img1), torch.tensor(img2), label.float()
    
    
def train(args, model, device, train_loader, optimizer, epoch, criterion):
    """
        modified version of an official pytorch example Siamese nn model using resnet18
        https://github.com/pytorch/examples/blob/main/siamese_network/main.py
    """
    model.train()
    for i, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print(f'Training epich {epoch}, completed batch {i}/{len(train_loader)}, loss = {loss.item():.6f}')


def test(model, device, test_loader, criterion):
    """
        modified version of an official pytorch example Siamese nn model using resnet18
        https://github.com/pytorch/examples/blob/main/siamese_network/main.py
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i,(images_1, images_2, targets) in enumerate(test_loader):
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss = test_loss + criterion(outputs, targets).sum().item()
    test_loss = test_loss/len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')


def main():
    parser = argparse.ArgumentParser(description='Siamese network for signature verification')
    parser.add_argument('--epochs', type=int, 
                        default=1, metavar='N',
                        help='number of epochs')
    parser.add_argument('--lr', type=float, 
                        default=1.0, metavar='LR',
                        help='learning rate')
    parser.add_argument('--gamma', type=float, 
                        default=0.7, metavar='M',
                        help='lr step gamma')
    parser.add_argument('--seed', type=int, default=614, metavar='S',
                        help='random seed')
    parser.add_argument('--train-batch-size', 
                        type=int, default=64, metavar='N',
                        help='batch size for training')
    parser.add_argument('--test-batch-size', 
                        type=int, default=1000, metavar='N',
                        help='batch size for testing')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='number of batches between out putting logs')
    parser.add_argument('--resize-rows', type=int, default=50, metavar='N',
                        help='number of rows in resized images')
    parser.add_argument('--resize-columns', type=int, default=70, metavar='N',
                        help='number of columns in resized image')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    train_dataset = SignatureDataset('train',  resize_dims = (args.resize_rows,args.resize_columns))
    test_dataset = SignatureDataset('test',  resize_dims = (args.resize_rows,args.resize_columns))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.train_batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.test_batch_size)

    model = SiameseNetwork().to(device)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    criterion = nn.BCELoss()

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        test(model, device, test_loader, criterion)
        scheduler.step()

    torch.save(model.state_dict(), "/opt/ml/model/siamese_network.pt")


if __name__ == '__main__':
    main()