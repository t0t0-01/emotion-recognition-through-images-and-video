from __future__ import division

"""
This code section allows us to test the model of our choice
on the testing set of a certain dataset
"""
import argparse
import time
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import datasets, transforms
from net_factory import GCN_mod
from utils import (AverageMeter, accuracy)

#import sys
#sys.path.append("D:\\Work\\ggg\\Gabor_CNN_FER")
parser = argparse.ArgumentParser(description='PyTorch GCN MNIST Training')
parser.add_argument('--gpu', default=1, type=int,
                    metavar='N', help='GPU device ID (default: 1)')
args = parser.parse_args()
use_cuda = (args.gpu > 0) and torch.cuda.is_available()


#Setting up the data transformation
transform =  transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(100),
        transforms.TenCrop(90),  
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),    
    ])

"""
Please revise the path for dataset
"""
#Loading the dataset
#Change DIR
data_dir = '.\\FER2013 Dataset\\'
image_datasets = datasets.ImageFolder(data_dir + 'test', transform)
test_loader = Data.DataLoader(dataset = image_datasets, batch_size = 32)

#Loading the model
model = GCN_mod(channel=4, lych = 10)
model.load_state_dict(torch.load('trained_RAF_10.pt', map_location=torch.device('cpu')))

#Choosing CPU or GPU
device = torch.device("cuda" if use_cuda else "cpu")
model = model.to(device)

#for every data point inside the specified folder, 
#predict and comapre y_pred with y_test to get the accuracy of the model
def test():
    model.eval()
    test_loss = AverageMeter()
    acc = AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            bs, ncrops, c,h,w = data.size()
            data = data.view(-1,c,h,w)
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.view(bs,ncrops,-1).mean(1)
            test_loss.update(F.cross_entropy(output, target, reduction='mean').item(), target.size(0))
            prec1, = accuracy(output, target) # test precison in one batch
            acc.update(prec1.item(), target.size(0))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss.avg, acc.avg))
    return acc.avg

st = time.time()
prediction= test()
epoch_time = time.time() - st
print('Accuracy is: %.2f%%' %prediction)
print('Test time:{:0.2f}s'.format(epoch_time))

print('\nFinished!')
