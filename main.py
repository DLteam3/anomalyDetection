# -*- coding: utf-8 -*-
"""
Created on Sat May 28 16:02:12 2022

@author: user
"""
#pip install opencv-python
#pip install timm
#%% wandb 설정
from __future__ import print_function
import wandb
wandb.init(project="DL_project", entity="jaemyoung")
#%% Package
import warnings
warnings.filterwarnings('ignore')
from glob import glob
import pandas as pd
import numpy as np 
from tqdm import tqdm
import cv2
import os
import timm
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import gc

#%% label 불러오기

train_png = sorted(glob('C:/Users/user/Documents/GitHub/DL_teamproject/dataset/train/*.png'))
test_png = sorted(glob('C:/Users/user/Documents/GitHub/DL_teamproject/dataset/test/*.png'))

train_y = pd.read_csv("C:/Users/user/Documents/GitHub/DL_teamproject/dataset/train_df.csv")
train_labels = train_y["label"]

# label : 88개
label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
train_labels = [label_unique[k] for k in train_labels]# train data에 label 씌우기

#%% image 불러오기
def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (256, 256)) #512*512 -> 사이즈를 줄여서 사용
    return img
train_imgs = [img_load(m) for m in tqdm(train_png)]
test_imgs = [img_load(n) for n in tqdm(test_png)]

#%% dataset 구축

class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.mode=='train':
            augmentation = random.randint(0,8)
            # RGB
            if augmentation<3:
                pass

            # Rotate 90'
            elif augmentation==3:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            # Rotate 270'
            elif augmentation==4:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Y flip
            elif augmentation==5:
                img = img[::-1].copy()

            # X flip
            elif augmentation==6:
                img = img[:,::-1].copy()

            # X, Y flip
            elif augmentation==7:
                img = img[::-1, ::-1, :].copy()
                

        img = transforms.ToTensor()(img)
        if self.mode=='test':
            pass
        
        label = self.labels[idx]
        return img, label
    
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=88) #efficientnet_b 1 2 3 4 5 6 7 
        
    def forward(self, x):
        x = self.model(x)
        return x

batch_size = 8
epochs = 10

# Train
train_dataset = Custom_dataset(np.array(train_imgs), np.array(train_labels), mode='train')
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# Test
test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"]*len(test_imgs)), mode='test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

#%% model 학습
def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

model = Network().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

best=0

wandb.init(project='DL_project', entity='jaemyoung', name='efficientnet_b0', config={'batch_size':batch_size, 'epochs':epochs, 'learning_rate':0.001})
for epoch in range(epochs):
    torch.cuda.empty_cache() #캐시 삭제
    start=time.time()
    train_loss = 0
    train_pred=[]
    train_y=[]
    model.train()
    for batch in (train_loader):
        optimizer.zero_grad()
        x = torch.tensor(batch[0], dtype=torch.float32) #device=device)
        y = torch.tensor(batch[1], dtype=torch.long) #device=device)
        with torch.cuda.amp.autocast():
            pred = model(x)
        loss = criterion(pred, y)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()/len(train_loader)
        train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
        train_y += y.detach().cpu().numpy().tolist()
        
    
    train_f1 = score_function(train_y, train_pred)

    TIME = time.time() - start
    print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
    print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
    wandb.log({'Epoch':'%01d' % (epoch + 1), 'train_loss':train_loss,'Train Accuracy':train_f1})
wandb.finish() 
#%% inference
model.eval()
f_pred = []

with torch.no_grad():
    for batch in (test_loader):
        x = torch.tensor(batch[0], dtype = torch.float32, device = device)
        with torch.cuda.amp.autocast():
            pred = model(x)
        f_pred.extend(pred.argmax(1).detach().cpu().numpy().tolist())

label_decoder = {val:key for key, val in label_unique.items()}

f_result = [label_decoder[result] for result in f_pred]

#%% 제출

submission = pd.read_csv("C:/Users/user/Documents/GitHub/DL_teamproject/dataset/sample_submission.csv")

submission["label"] = f_result

submission.to_csv("C:/Users/user/Documents/GitHub/DL_teamproject/dataset/submission(1).csv",index=False)
