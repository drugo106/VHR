import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import sys 
from torchinfo import summary
from scipy import signal
from scipy.stats import zscore
import pyVHR as vhr
import pickle
from typing import Optional
import math
import os
import random

from time import gmtime, strftime

#import torchvision
#from torch.utils.tensorboard import SummaryWriter
#from torchvision import datasets, transforms


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

#vhr.plot.VisualizeParams.renderer = 'notebook'  # or 'notebook'

dataset_name = 'pure'           
video_DIR = '/var/data/VHR1/'  
BVP_DIR = '/var/datasets/VHR1/'    

dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR=video_DIR, BVPdataDIR=BVP_DIR)
allvideo = dataset.videoFilenames


def chunks(lst, bvp, stride = 5, window = 160):
    tmp = []
    tmp_bvp = []
    for i in range(0, len(lst)-window, stride):
        tmp.append(torch.as_tensor(lst[i:i + window]))
        tmp_bvp.append(torch.as_tensor(bvp[i:i + window]))
    if tmp[-1].shape[0] != tmp[0].shape[0]:
        return(torch.stack(tmp[:-1]),torch.stack(tmp_bvp[:-1]))
    return(torch.stack(tmp),torch.stack(tmp_bvp))


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
        #with open('/var/datasets/VIPL-HR-V1_webs/'+str(idx)+'-WEBS-'+str(PATCH_SIZE), 'rb') as f:

            #print('/var/datasets/PURE_webs/'+str(idx)+'-WEBS-'+str(PATCH_SIZE))
            (webs,labels) = pickle.load(f)
            if idx < ((tot/10)*9)-1:   #9 : 0 : 1
                train_video.append(webs)
                train_bvp.append(zscore(labels))
            #elif idx < (tot/10)*8+(tot/10):
                #print(2)
                #val_video.append(webs)
                #val_bvp.append(labels)
            else:
                #print(3)
                test_video.append(webs)
                test_bvp.append(zscore(labels))
            if min_len==0 or len(webs)<min_len:
                min_len = len(webs)
    for i in range(0,len(train_video)):
        train_video[i],train_bvp[i] = chunks(train_video[i][:min_len-1], train_bvp[i][:min_len-1])
    #for i in range(0,len(val_video)):
    #    val_video[i],val_bvp[i] = chunks(val_video[i][:min_len-1], val_bvp[i][:min_len-1],160)
    for i in range(0,len(test_video)):
        test_video[i],test_bvp[i] = chunks(test_video[i][:min_len-1], test_bvp[i][:min_len-1])
    return (train_video, train_bvp), (val_video, val_bvp), (test_video, test_bvp)



def save_ckp(model, optimizer, loss, epoch, iteration, path=".",rPPG=None):
    checkpoint = {
        'epoch': epoch + 1,
        'iteration': iteration + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'rPPG': rPPG
    }
    f_path = path + '/'+ str(epoch +1) + 'checkpoint.pt'
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
        for i in range(labels.shape[0]):
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
    
    #writer = SummaryWriter()
    
    path_log = 'LOG'
    isExists = os.path.exists(path_log)
    if not isExists:
        os.makedirs(path_log)
    log_file = open(path_log+'/LOG_log.txt', 'w')

    checkpoint_dir = "models/" + strftime("%Y-%m-%d %H:%M:%S", gmtime())
    os.mkdir(checkpoint_dir)
    
    for epoch in range(epoch_start, epochs):
        print("\nStarting epoch", epoch+1)
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
                inputs, targets = inputs.float().cuda(), targets.float().cuda()
                inputs_resized = torch.zeros((4,3,160,128,128)).float().cuda()
                for j in range(0, inputs.shape[0]):
                    inputs_resized[j] = torch.from_numpy(np.resize(inputs[j].cpu().numpy(),(3,160,128,128)))

                
                optimizer.zero_grad()
                rPPG_nonorm = model(inputs_resized,0.2)
                rPPG = (rPPG_nonorm-torch.mean(rPPG_nonorm)) /(torch.std(rPPG_nonorm))
                #loss
                
                
                loss = criterion_reg
                loss_rPPG = criterion_Pearson(rPPG, targets)
                fre_loss = 0.0
                kl_loss = 0.0
                train_mae = 0.0
                for bb in range(inputs.shape[0]):
                    loss_distribution_kl, fre_loss_temp, train_mae_temp = TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(
                        rPPG[bb], torch.mean(targets[bb].float()), 30, std=1.0) 
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
            
                loss =  a*loss_rPPG + b*(fre_loss)#+kl_loss)
                #loss =  loss_rPPG
                """
                if(targets.shape[0]!=rPPG.shape[0]):
                    rPPG = torch.split(rPPG,[targets.shape[0],rPPG.shape[0] - targets.shape[0]])[0]

                #loss = torch.mean(torch.abs(rPPG - targets))  #MAE
                loss = torch.mean(torch.pow((rPPG - targets),2))  #MSE
                #loss = torch.sqrt(torch.mean(torch.pow((rPPG - targets),2)))  #RMSE
                #loss = torch.mean(nn.CosineSimilarity()(rPPG,targets))
                #loss = criterion_reg(rPPG,targets)
                #loss = loss_rPPG
                #loss = criterion_Pearson(rPPG, targets)
                """
                loss.backward()
                optimizer.step()
                n = inputs.size(0)
                loss_rPPG_avg.update(loss_rPPG.data, n)
                loss_peak_avg.update(fre_loss.data, n)
                loss_kl_avg_test.update(kl_loss.data, n)
                loss_bvp_mae.update(train_mae, n)
            
                
                sys.stdout.write('\r')
                sys.stdout.write(f"Iteration {i+1}, loss= {loss:1.5f}, NegPearson= {loss_rPPG_avg.avg:1.5f}, kl= {loss_kl_avg_test.avg:1.5f}, fre_CEloss= {loss_peak_avg.avg:1.5f}")
                sys.stdout.flush()
                
                #writer.add_scalar('Epoch '+str(epoch)+' Loss/train', loss,i)
                
                if( i%50 == 0):
                    save_ckp(model, optimizer, loss, epoch, i, path=checkpoint_dir, rPPG=rPPG_nonorm)
                    log_file.write("\n")        
                    log_file.write(f"Epoch {epoch+1}, Iteration {i+1}, loss= {loss/((i+1)*inputs.shape[0]):1.5f}, NegPearson= {loss_rPPG_avg.avg:1.5f}, kl= {loss_kl_avg_test.avg:1.5f}, fre_CEloss= {loss_peak_avg.avg:1.5f}")
                    log_file.write("\n")
                    log_file.write("\n")
                    log_file.flush()
                

    return model, loss, loss_rPPG_avg, loss_peak_avg, loss_kl_avg_test, loss_bvp_mae

def test_chunk1(model):
    print(test_video.shape,test_bvp.shape)
    dataset = torch.utils.data.TensorDataset(test_video.permute(0,4,1,2,3),test_bvp)
    testloader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=1)
    model = model.cuda()
    model.eval()

    r=[random.randint(0,len(testloader)),random.randint(0,len(testloader)),random.randint(0,len(testloader))]
    r=r+r
    print(r)
    
    rPPG = []
    gt = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            if i in r:
                inputs, targets = data
                inputs, targets = inputs.float().cuda(), targets.float().cuda()
            
                inputs = torch.from_numpy(np.resize(inputs.cpu().numpy(),(1,3,160,128,128))).cuda()

                rPPG.append((model(inputs, 2.0).cpu()[0]))
                gt.append(targets.cpu()[0]/torch.std(targets.cpu()[0]))
    return (rPPG,gt,r)


def plot_result(rPPG,gt,title,r):
    fig = plt.figure(figsize=(1, 3))
    fig.suptitle(title)
    for i in range(3):
        print(torch.mean(torch.pow((rPPG[i] - gt[i]),2)))  #MSE
        ax = fig.add_subplot(1, 3, i+1)
        ax.plot(rPPG[i][10:],label="rPPG_chunck_"+str(r[i]))
        ax.plot(gt[i],label="GT_chunck_"+str(r[i]))
        ax.set_xlabel('frames')
        ax.legend()
    plt.show()

    

def test(model,optimizer,path='./checkpoint.pt',title=''):
    model,_,_,_= load_ckp(model,optimizer,path=path)
    (rPPG,gt,r) = test_chunk1(model)
    for i in range(0,len(rPPG)): rPPG[i] = (rPPG[i]-torch.mean(rPPG[i]))/(torch.std(rPPG[i]))
    plot_result(rPPG,gt,title,r)
    
(train_video,train_bvp),(val_video,val_bvp),(test_video,test_bvp) = load_data(7)

train_video, train_bvp = torch.cat(train_video[:]), torch.cat(train_bvp[:])
#val_video, val_bvp     = torch.cat(val_video[:]), torch.cat(val_bvp[:])
test_video, test_bvp   = torch.cat(test_video[:]), torch.cat(test_bvp[:])


BATCH = 4
#train_video = train_video.permute(0,4,1,2,3)
#val_video = val_video.permute(0,4,1,2,3)
#test_video = test_video.permute(0,4,1,2,3)
print(train_video.shape,train_bvp.shape)
dataset = torch.utils.data.TensorDataset(train_video.permute(0,4,1,2,3),torch.as_tensor(train_bvp))
trainloader = torch.utils.data.DataLoader(dataset,batch_size=BATCH, shuffle=True, num_workers=1)

model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160,128,128), patches=(4,4,4), dim=96, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
print(summary(model))

optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.L1Loss()
criterion_Pearson = Neg_Pearson() 

loss = 0.0
epochs = 10


model = model.cuda()
model.train() 
print("iterations per epoch: ",len(trainloader)) 
print("epochs: {0}\nStart".format(epochs)) 
train(model,optimizer,trainloader, 0,0) 
print('\nTraining process has finished.')


