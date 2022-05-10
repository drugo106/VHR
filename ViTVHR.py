import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
import pyVHR as vhr
import numpy as np
from torchinfo import summary
import sys

#ViT MODEL

PATCH_SIZE = 15
EMBED_DIM = PATCH_SIZE * PATCH_SIZE * 3
HEADS = 5
BLOCKS = 12
IMG_SIZE = PATCH_SIZE * 10


class PatchEmbed(nn.Module):
  def __init__(self, img_size, patch_size, in_chans=3, embed_dim=EMBED_DIM):
    super().__init__()
    self.img_size = img_size
    self.patch_size = patch_size
    self.n_patches = (img_size // patch_size) ** 2
    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

  def forward(self, x):
    x = torch.movedim(x,3,1)
    x = self.proj(x)       # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
    x = x.flatten(2)        # (n_samples, embed_dim, n_patches)
    x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

    return x

class Attention(nn.Module):


  def __init__(self, dim, n_heads=HEADS, qkv_bias=True, attn_p=0., proj_p=0.):
    super().__init__()
    self.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads
    self.scale = self.head_dim ** -0.5

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_p)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_p)

  def forward(self, x):
    n_samples, n_tokens, dim = x.shape

    if dim != self.dim:
      raise ValueError

    qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
    qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches + 1, head_dim)

    # compute att matrices
    q, k, v = qkv[0], qkv[1], qkv[2]
    k_t = k.transpose(-2, -1)   # (n_samples, n_heads, head_dim, n_patches + 1)
    dp = (q @ k_t) * self.scale

    attn = dp.softmax(dim=-1)   # (n_samples, n_heads, n_patches + 1, n_patches + 1)
    attn = self.attn_drop(attn)

    # compute weigthed avg
    weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
    weighted_avg = weighted_avg.transpose(1, 2)  # (n_samples, n_patches + 1, n_heads, head_dim)
    weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

    # linear projection
    x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
    x = self.proj_drop(x)        # (n_samples, n_patches + 1, dim)

    return x

class MLP(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, p=0.):
      super().__init__()
      self.fc1 = nn.Linear(in_features, hidden_features)
      self.act = nn.GELU()
      self.fc2 = nn.Linear(hidden_features, out_features)
      self.drop = nn.Dropout(p)

    def forward(self, x):
      x = self.fc1(x)   # (n_samples, n_patches + 1, hidden_features)
      x = self.act(x)   # (n_samples, n_patches + 1, hidden_features)
      x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
      x = self.fc2(x)   # (n_samples, n_patches + 1, hidden_features)
      x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)

      return x

class Block(nn.Module):
    
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
      super().__init__()
      self.norm1 = nn.LayerNorm(dim, eps=1e-6)
      self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
      self.norm2 = nn.LayerNorm(dim, eps=1e-6)
      hidden_features = int(dim * mlp_ratio)
      self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim)

    def forward(self, x):
     
      
      x = x + self.attn(self.norm1(x))
      x = x + self.mlp(self.norm2(x))

      return x

class VisionTransformer(nn.Module):
  
  def __init__(
          self,
          img_size=IMG_SIZE,
          patch_size=PATCH_SIZE,
          in_chans=3,
          n_classes=1000,
          embed_dim=EMBED_DIM,
          depth=BLOCKS,
          n_heads=HEADS,
          mlp_ratio=4.,
          qkv_bias=True,
          p=0.,
          attn_p=0.,
  ):
    super().__init__()

    self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
    )
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
    )
    self.pos_drop = nn.Dropout(p=p)

    self.blocks = nn.ModuleList(
        [
            Block(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                p=p,
                attn_p=attn_p,
            )
            for _ in range(depth)
        ]
    )

    self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
    self.head = nn.Linear(embed_dim, n_classes)

  def forward(self, x):
    n_samples = x.shape[0]
    x = self.patch_embed(x)  #(n_samples, n_patches, embed_dim)
    
    
    cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
    x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
    x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
    x = self.pos_drop(x)

    for block in self.blocks:
      x = block(x)

    x = self.norm(x)

    cls_token_final = x[:, 0]  # just the CLS token
    #x = self.head(cls_token_final)
    x = x.type(torch.DoubleTensor)
    
    x = torch.mean(x,2)
    x = torch.mean(x,1)
    
    return x


#SET DATASET
print("\n###### Load dataset ######")
vhr.plot.VisualizeParams.renderer = 'notebook'  # or 'notebook'
    
dataset_name = 'pure'           
video_DIR = '/var/datasets/VHR1/'  
BVP_DIR = '/var/datasets/VHR1/'    

dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR=video_DIR, BVPdataDIR=BVP_DIR)
allvideo = dataset.videoFilenames


wsize = 8          
video_idx = 0      
fname = dataset.getSigFilename(video_idx)
sigGT = dataset.readSigfile(fname)
test_bvp = [sigGT.data[0][i] for i in range(0,len(sigGT.data[0])-1,2)]
#bpmGT, timesGT = sigGT.getBPM(wsize)
videoFileName = dataset.getVideoFilename(video_idx)
print('Video processed name: ', videoFileName)
fps = vhr.extraction.get_fps(videoFileName)
print('Video frame rate:     ',fps)

#vhr.plot.display_video(videoFileName)

#SKIN EXTRACTION AND VISUALIZATION
def patches_extraction():
    sig_extractor = vhr.extraction.SignalProcessing()
    sig_extractor.set_skin_extractor(vhr.extraction.SkinExtractionConvexHull())
    seconds = 0
    sig_extractor.set_total_frames(seconds*fps)

    vhr.extraction.SkinProcessingParams.RGB_LOW_TH = 2
    vhr.extraction.SkinProcessingParams.RGB_HIGH_TH = 254
    vhr.extraction.SignalProcessingParams.RGB_LOW_TH = 2
    vhr.extraction.SignalProcessingParams.RGB_HIGH_TH = 254

    sig_extractor.set_visualize_skin_and_landmarks(
          visualize_skin=True, 
          visualize_landmarks=True, 
          visualize_landmarks_number=True, 
          visualize_patch=True)

    landmarks = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, \
             58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, \
             118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188, 193, 197, 201, 205, 206, 207, \
             210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, \
             284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, \
             346, 361, 363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430, 432, 436]

    print('Num landmarks: ', len(landmarks))
    #vhr.plot.visualize_landmarks_list(landmarks_list=landmarks)
    sig_extractor.set_landmarks(landmarks)

    #PATCHES EXTRACTION

    sig_extractor.set_square_patches_side(PATCH_SIZE+0.0)
    patch_sig = sig_extractor.extract_patches(videoFileName, "squares", "mean")
    print('Size: (#frames, #landmarks, #channels) = ', patch_sig.shape)

    visualize_patches_coll = sig_extractor.get_visualize_patches()
    print('Number of frames processed: ',len(visualize_patches_coll))
    #vhr.plot.interactive_image_plot(visualize_patches_coll,1.0)

    patches = sig_extractor.patches
    print("patches: ", len(patches), len(patches[0]))
    #return torch.as_tensor(patches)
    return patches

print("\n###### Patches extraction ######")
patches = patches_extraction()

print("\n###### Creating webs ######")
#for each frame concatenate patches in one image
webs = []

for f_p in patches[:len(test_bvp)]:
    tmp=np.concatenate(f_p[0:10])
    for i in range(10,len(f_p),10):
        r=np.concatenate(f_p[i:i+10])
        tmp=np.concatenate((tmp,r),axis=1)
    if webs==[]:
        webs = torch.unsqueeze(torch.as_tensor(tmp),0)
    else:
        webs=torch.cat((webs,torch.unsqueeze(torch.as_tensor(tmp),0)))

#inputs = torch.as_tensor(inputs)
print(webs.shape)


#PREPARE DATASET
BATCH_SIZE = 300
print(webs.shape,torch.as_tensor(test_bvp).shape)
dataset = torch.utils.data.TensorDataset(webs,torch.as_tensor(test_bvp))
trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

#TRAIN MODEL
print("\n###### Training ######")
v = VisionTransformer()
print(summary(v))

optimizer = torch.optim.Adam(v.parameters(), lr=1e-4)
loss_function = nn.L1Loss()
last_loss = 0.0
epochs = 5
print("epochs: {0}\nStart".format(epochs))
for epoch in range(0, epochs):
    print("\nStarting epoch", epoch+1)
    current_loss = 0.0
    last_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        sys.stdout.write('\r')
        sys.stdout.write(f"Iteration {i}, ")
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        #targets = targets.reshape((targets.shape[0], 1))
        optimizer.zero_grad()
        outputs = v(inputs)        
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
        sys.stdout.write(f"Loss:  {current_loss/BATCH_SIZE:1.5f}")
        sys.stdout.flush()
        if i % 10 == 0:
            last_loss = current_loss
            current_loss = 0.0

print('Training process has finished.')


with open('MODEL', 'wb') as f:
    pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('LOSS', 'wb') as f:
    pickle.dump(last_loss, f, protocol=pickle.HIGHEST_PROTOCOL)




