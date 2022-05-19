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
 
from TorchLossComputer import TorchLossComputer
from TorchLossComputerCPU import TorchLossComputerCPU

PATCH_SIZE = 16
EMBED_DIM = PATCH_SIZE * PATCH_SIZE * 3
NUM_PATCHES = 100
IMG_SIZE = PATCH_SIZE * NUM_PATCHES
HEADS = 12
BLOCKS = 12
BATCH = 300
LENGTH = 160

vhr.plot.VisualizeParams.renderer = 'notebook'  # or 'notebook'

dataset_name = 'pure'           
video_DIR = '/var/datasets/VHR1/'  
BVP_DIR = '/var/datasets/VHR1/'    

dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR=video_DIR, BVPdataDIR=BVP_DIR)
allvideo = dataset.videoFilenames

"""
Adapted from https://github.com/lukemelas/simple-bert
"""
import math

import pdb

'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal




def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)



class MultiHeadedSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""
    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()
        
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
            #nn.ELU(),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.BatchNorm3d(dim),
            #nn.ELU(),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),  
            #nn.BatchNorm3d(dim),
            #nn.ELU(),
        )
        
        #self.proj_q = nn.Linear(dim, dim)
        #self.proj_k = nn.Linear(dim, dim)
        #self.proj_v = nn.Linear(dim, dim)
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, gra_sharp):    # [B, 4*4*40, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        [B, P, C]=x.shape
        x = x.transpose(1, 2).view(B, C, P//8, 4, 2)      # [B, dim, 40, 4, 4]
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        k = k.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        v = v.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / gra_sharp

        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h, scores




class PositionWiseFeedForward_ST(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        
        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):    # [B, 4*4*40, 128]
        [B, P, C]=x.shape
        #x = x.transpose(1, 2).view(B, C, 40, 4, 4)      # [B, dim, 40, 4, 4]
        x = x.transpose(1, 2).view(B, C, P//8, 4, 2)      # [B, dim, 40, 4, 4]
        x = self.fc1(x)		              # x [B, ff_dim, 40, 4, 4]
        x = self.STConv(x)		          # x [B, ff_dim, 40, 4, 4]
        x = self.fc2(x)		              # x [B, dim, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        
        return x
        
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        #return self.fc2(F.gelu(self.fc1(x)))





class Block_ST_TDC_gra_sharp(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, Score


class Transformer_ST_TDC_gra_sharp(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp(dim, num_heads, ff_dim, dropout, theta) for _ in range(num_layers)])

    def forward(self, x, gra_sharp):
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score

"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

'''from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import math

from .transformer_layer import Transformer_ST_TDC_gra_sharp

import pdb'''


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal




# stem_3DCNN + ST-ViT with local Depthwise Spatio-Temporal MLP
class ViT_ST_ST_Compact3_TDC_gra_sharp(nn.Module):

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.2,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        #positional_embedding: str = '1d',
        in_channels: int = 3, 
        frame: int = 160,
        theta: float = 0.2,
        image_size: Optional[int] = None,
    ):
        super().__init__()

        
        self.image_size = image_size  
        self.frame = frame  
        self.dim = dim              

        # Image and patch sizes
        t, h, w = as_tuple(image_size)  # tube sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40
        gt, gh, gw = t//ft, h // fh, w // fw  # number of patches
        seq_len = gh * gw * gt

        # Patch embedding    [4x16x16]conv
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))
        
        # Transformer
        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer3 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        
        
        
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim//4, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(dim//4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim//4, dim//2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim//2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        #self.normLast = nn.LayerNorm(dim, eps=1e-6)
        
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(8,1,1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(5,1,1)),
            nn.Conv3d(dim, dim//2, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim//2),
            nn.ELU(),
        )
 
        self.ConvBlockLast = nn.Conv1d(dim//2, 1, 1,stride=1, padding=0)
        
        
        # Initialize weights
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)


    def forward(self, x, gra_sharp):
        b, c, t, fh, fw = x.shape
        
        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)  # [B, 64, 160, 64, 64]
        x = self.patch_embedding(x)  # [B, 64, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 40*4*4, 64]
        
        Trans_features, Score1 =  self.transformer1(x, gra_sharp)  # [B, 4*4*40, 64]
        Trans_features2, Score2 =  self.transformer2(Trans_features, gra_sharp)  # [B, 4*4*40, 64]
        Trans_features3, Score3 =  self.transformer3(Trans_features2, gra_sharp)  # [B, 4*4*40, 64]
        
        
        #Trans_features3 = self.normLast(Trans_features3)
        
        # upsampling heads
        #features_last = Trans_features3.transpose(1, 2).view(b, self.dim, 40, 4, 4) # [B, 64, 40, 4, 4]
        features_last = Trans_features3.transpose(1, 2).view(b, self.dim, 4, 5,2) # [B, 64, 40, 4, 4]
        
        features_last = self.upsample(features_last)		    # x [B, 64, 7*7, 80]
        features_last = self.upsample2(features_last)		    # x [B, 32, 7*7, 160]

        features_last = torch.mean(features_last,3)     # x [B, 32, 160, 4]  
        features_last = torch.mean(features_last,3)     # x [B, 32, 160]    
        rPPG = self.ConvBlockLast(features_last)    # x [B, 1, 160]
        
        #pdb.set_trace()
        
        rPPG = rPPG.squeeze(1)
        
        #return rPPG, Score1, Score2, Score3
        return rPPG


def chunks(lst,bvp, n):
    tmp = []
    tmp_bvp = []
    for i in range(0, len(lst), n):
        tmp.append(torch.as_tensor(lst[i:i + n]))
        tmp_bvp.append(torch.as_tensor(bvp[i:i + n]))
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
            #print('/var/datasets/PURE_webs/'+str(idx)+'-WEBS-'+str(PATCH_SIZE))
            (webs,labels) = pickle.load(f)
            if idx < (tot/10)*9:   #9 : 0 : 1
                #print(1)
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
        train_video[i],train_bvp[i] = chunks(train_video[i][:min_len-1], train_bvp[i][:min_len-1],160)
    #for i in range(0,len(val_video)):
    #    val_video[i],val_bvp[i] = chunks(val_video[i][:min_len-1], val_bvp[i][:min_len-1],160)
    for i in range(0,len(test_video)):
        test_video[i],test_bvp[i] = chunks(test_video[i][:min_len-1], test_bvp[i][:min_len-1],160)
    return (train_video, train_bvp), (val_video, val_bvp), (test_video, test_bvp)


(train_video,train_bvp),(val_video,val_bvp),(test_video,test_bvp) = load_data(len(allvideo))

train_video, train_bvp = torch.cat(train_video[:]), torch.cat(train_bvp[:])
#val_video, val_bvp     = torch.cat(val_video[:]), torch.cat(val_bvp[:])
test_video, test_bvp   = torch.cat(test_video[:]), torch.cat(test_bvp[:])


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

def train(model, optimizer, trainloader, epoch_start=0, iter_start=0):
    a_start = 0.1
    b_start = 1.0
    exp_a = 0.5
    exp_b = 5.0
    for epoch in range(epoch_start, epochs):
        print("\nStarting epoch", epoch+1)
        current_loss = 0.0
        loss = 0.0

        for i, data in enumerate(trainloader,0):
            sys.stdout.write('\r')
            sys.stdout.write(f"Iteration {i+1}, {loss/BATCH:1.5f}")
            sys.stdout.flush()
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
                for bb in range(inputs.shape[0]):
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

    return model, loss

BATCH = 4
#train_video = train_video.permute(0,4,1,2,3)
#val_video = val_video.permute(0,4,1,2,3)
#test_video = test_video.permute(0,4,1,2,3)
print(test_video.shape,train_bvp.shape)
dataset = torch.utils.data.TensorDataset(train_video.permute(0,4,1,2,3),torch.as_tensor(train_bvp))
trainloader = torch.utils.data.DataLoader(dataset,batch_size=BATCH, shuffle=True, num_workers=1)

model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160,160,160), patches=(4,16,16), dim=160, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
print(summary(model))

optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.L1Loss()
criterion_Pearson = Neg_Pearson() 

loss = 0.0
epochs = 25


#model.cuda()
model.train() 
print("iterations per epoch: ",len(trainloader)) 
print("epochs: {0}\nStart".format(epochs)) 
train(model,optimizer,trainloader, 0,0) 
print('\nTraining process has finished.')

