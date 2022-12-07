# -*- coding: utf-8 -*-

#Created on Mon Jun 22 13:58:32 2020
#@author: Dell

"""
This code section allows us to test the model of our choice
on one image at a time.
"""
import torch
from PIL import Image
from net_factory import GCN_mod
import matplotlib.pyplot as plt
from torchvision import transforms
import time

#Load the model
model = GCN_mod(channel=4, lych = 10)
model.load_state_dict(torch.load('trained_RAF_10.pt', map_location=torch.device('cpu')))
model.eval()

#Load the image
#Change directory
img = Image.open('.\\Images\\4 people.jpeg')
"""
img = Image.open('.\\Images\\Cropped - far.jpeg')
img = Image.open('.\\Images\\Cropped1.jpeg')
img = Image.open('.\\Images\\Cropped2.jpeg') 
img = Image.open('.\\Images\\Cropped3.jpeg') 
img = Image.open('.\\Images\\Cropped4.jpeg') 
"""
img = img.resize((90,90))


"""
Ten-crop test method
"""
#Transform the image
transform = transforms.Compose([
        transforms.Grayscale(1),        
        transforms.TenCrop(90),  
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),    
        ])
timg = transform(img)

#predict the class of the image
with torch.no_grad():
    score = model(timg)

score = score.mean(dim=0)
emotion = torch.argmax(score)

"""
recognition result
Anger:     0
Dsigust:   1
Fear:      2
Happy:     3
Netural:   4
Sadness:   5
Surprise:  6
"""
emo = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise'] #Added Line of code to decode the class
print("expression class is", emo[emotion])


