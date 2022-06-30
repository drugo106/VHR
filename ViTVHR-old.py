import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
import sys 
from torchinfo import summary
from scipy import signal
import pyVHR as vhr
import pickle
from typing import Optional
import math

from TorchLossComputer import TorchLossComputer
from TorchLossComputerCPU import TorchLossComputerCPU
from PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp


        
PATCH_SIZE = 16
EMBED_DIM = PATCH_SIZE * PATCH_SIZE * 3
NUM_PATCHES = 100
IMG_SIZE = PATCH_SIZE * NUM_PATCHES
HEADS = 12
BLOCKS = 12
BATCH = 300
LENGTH = 160



def chunks(lst, bvp, stride = 5, window = 160):
    tmp = []
    tmp_bvp = []
    for i in range(0, len(lst)-window, stride):
        tmp.append(torch.as_tensor(lst[i:i + window]))
        tmp_bvp.append(torch.as_tensor(bvp[i:i + window]))
    if tmp[-1].shape[0] != tmp[0].shape[0]:
        return(torch.stack(tmp[:-1]),torch.stack(tmp_bvp[:-1]))
    return(torch.stack(tmp),torch.stack(tmp_bvp))
        
#webs,train_bvp = chunks(webs,train_bvp,160)

def load_data(tot):
    min_len = 0
    train_video = []
    train_bvp = []
    val_video = []
    val_bvp = []
    test_video = []
    test_bvp = []
    for idx in range(0,tot):
        with open('/var/datasets/PURE_webs/'+str(idx)+'-WEBS-'+str(PATCH_SIZE), 'rb') as f:
            #print('/var/datasets/PURE_webs/'+str(idx)+'-WEBS-'+str(PATCH_SIZE))
            (webs,labels) = pickle.load(f)
            if idx < ((tot/10)*9)-1:   #9 : 0 : 1
                train_video.append(webs)
                train_bvp.append(labels)
            #elif idx < (tot/10)*8+(tot/10):
                #print(2)
                #val_video.append(webs)
                #val_bvp.append(labels)
            else:
                #print(3)
                test_video.append(webs)
                test_bvp.append(labels)
            if min_len==0 or len(webs)<min_len:
                min_len = len(webs)
    for i in range(0,len(train_video)):
        train_video[i],train_bvp[i] = chunks(train_video[i][:min_len-1], train_bvp[i][:min_len-1])
    #for i in range(0,len(val_video)):
    #    val_video[i],val_bvp[i] = chunks(val_video[i][:min_len-1], val_bvp[i][:min_len-1],160)
    for i in range(0,len(test_video)):
        test_video[i],test_bvp[i] = chunks(test_video[i][:min_len-1], test_bvp[i][:min_len-1])
    return (train_video, train_bvp), (val_video, val_bvp), (test_video, test_bvp)



def save_ckp(model, optimizer, epoch, loss, iteration, path="."):
    checkpoint = {
        'epoch': epoch + 1,
        'iteration': iteration + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    f_path = path + '/checkpoint.pt'
    torch.save(checkpoint, f_path)
    
def load_ckp(model, optimizer, path='./checkpoint.pt'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'], checkpoint['iteration']


class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; i>0, 1- loss                                                                      
    def __init__(self):                                                           
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):       # all variable operation    
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])
            sum_y = torch.sum(labels[i])
            sum_xy = torch.sum(preds[i]*labels[i])
            sum_x2 = torch.sum(torch.pow(preds[i],2))
            sum_y2 = torch.sum(torch.pow(labels[i],2))
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss += 1 - pearson
        loss = loss/preds.shape[0]
        return loss


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def train(model, optimizer, trainloader, epoch_start=0, iter_start=0):
    criterion_reg = nn.MSELoss()
    criterion_L1loss = nn.L1Loss()
    criterion_class = nn.CrossEntropyLoss()
    criterion_Pearson = Neg_Pearson()
    
    a_start = 0.1
    b_start = 1.0
    exp_a = 0.5
    exp_b = 5.0
    
    for epoch in range(epoch_start, epochs):
        print("\nStarting epoch", epoch+1)
        current_loss = 0.0
        loss = 0.0
        loss_rPPG_avg = AvgrageMeter()
        loss_peak_avg = AvgrageMeter()
        loss_kl_avg_test = AvgrageMeter()
        loss_bvp_mae = AvgrageMeter()
        
        model.train()
        
        for i, data in enumerate(trainloader,0):
            if i >= iter_start: 
                iter_start = 0
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                optimizer.zero_grad()
                rPPG = model(inputs,0.2)   
                rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)
                #loss
                loss_rPPG = criterion_Pearson(rPPG, targets)
                fre_loss = 0.0
                kl_loss = 0.0
                train_mae = 0.0
                for bb in range(BATCH):
                    loss_distribution_kl, fre_loss_temp, train_mae_temp = TorchLossComputerCPU.cross_entropy_power_spectrum_DLDL_softmax2(
                        rPPG[bb], torch.mean(targets[bb].float()), 30, std=1.0)  # std=1.1
                    fre_loss = fre_loss + fre_loss_temp
                    kl_loss = kl_loss + loss_distribution_kl
                    train_mae = train_mae + train_mae_temp
                fre_loss = fre_loss/inputs.shape[0]
                kl_loss = kl_loss/inputs.shape[0]
                train_mae = train_mae/inputs.shape[0]
                if epoch >25:
                    a = 0.05
                    b = 5.0
                else:
                    a = a_start*math.pow(exp_a, epoch/25.0)
                    b = b_start*math.pow(exp_b, epoch/25.0)
            
                a = 0.1
                #b = 1.0
            
                loss =  a*loss_rPPG + b*(fre_loss+kl_loss)
                
                loss.backward()
                optimizer.step()

                n = inputs.size(0)
                loss_rPPG_avg.update(loss_rPPG.data, n)
                loss_peak_avg.update(fre_loss.data, n)
                loss_kl_avg_test.update(kl_loss.data, n)
                loss_bvp_mae.update(train_mae, n)
                

                sys.stdout.write('\r')
                sys.stdout.write(f"Iteration {i+1}, {loss/((i+1)*inputs.shape[0]):1.5f}, {loss_rPPG_avg.avg:1.5f}, {loss_peak_avg.avg:1.5f}, {loss_kl_avg_test.avg:1.5f}")
                sys.stdout.flush()

                save_ckp(model, optimizer, loss, epoch, i)
                
                '''
                outputs = torch.mean(outputs,1)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                sys.stdout.write('\r')
                sys.stdout.write(f"Iteration {i}, Loss:  {current_loss/BATCH:1.5f}")
                sys.stdout.flush()
                if i % 10 == 0:
                    last_loss = current_loss
                    current_loss = 0.0'''

    return model, loss, loss_rPPG_avg, loss_peak_avg, loss_kl_avg_test, loss_bvp_mae


#vhr.plot.VisualizeParams.renderer = 'notebook'  # or 'notebook'

dataset_name = 'pure'           
video_DIR = '/var/datasets/VHR1/'  
BVP_DIR = '/var/datasets/VHR1/'    
dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR=video_DIR, BVPdataDIR=BVP_DIR)
allvideo = dataset.videoFilenames

(train_video,train_bvp),(val_video,val_bvp),(test_video,test_bvp) = load_data(7)
train_video, train_bvp = torch.cat(train_video[:]), torch.cat(train_bvp[:])
#val_video, val_bvp     = torch.cat(val_video[:]), torch.cat(val_bvp[:])
test_video, test_bvp   = torch.cat(test_video[:]), torch.cat(test_bvp[:])

print(train_video.shape, test_video.shape)


BATCH = 4
#train_video = train_video.permute(0,4,1,2,3)
#val_video = val_video.permute(0,4,1,2,3)
#test_video = test_video.permute(0,4,1,2,3)
print(train_video.shape,train_bvp.shape)
dataset = torch.utils.data.TensorDataset(train_video.permute(0,4,1,2,3),torch.as_tensor(train_bvp))
trainloader = torch.utils.data.DataLoader(dataset,batch_size=BATCH, shuffle=True, num_workers=1)

model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160,160,160), patches=(4,16,16), dim=160, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
print(summary(model))

optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.L1Loss()
criterion_Pearson = Neg_Pearson() 

loss = 0.0
epochs = 1



model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160,160,160), patches=(4,16,16), dim=160, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
#model = model.cuda()
model.train() 
print("iterations per epoch: ",len(trainloader)) 
print("epochs: {0}\nStart".format(epochs)) 
train(model,optimizer,trainloader, 0,0) 
print('\nTraining process has finished.')

