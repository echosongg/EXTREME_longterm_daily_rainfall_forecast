import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import time
from os import mkdir
from os.path import join, isdir
from datetime import date
from datetime import datetime
from torch.utils.data import random_split, DataLoader
import properscoring as ps
from datetime import date, timedelta
from torchvision import transforms

from utils import Huber, ACCESS_AWAP_GAN, RMSE, MAE

import cv2

### USER PARAMS ###
START_TIME = date(1990, 1, 1)
END_TIME = date(2005, 12, 31)

EXP_NAME = "/scratch/iu60/xs5813/DESRGAN_YAO"
VERSION = "original_DESRGAN"
UPSCALE = 8  # upscaling factor

NB_BATCH = 18  # mini-batch
NB_Iteration = 10
PATCH_SIZE = 256  # Training patch size
NB_THREADS = 36

START_ITER = 0  # Set 0 for from scratch, else will load saved params and trains further


L_ADV = 1e-3  # Scaling params for the Adv loss
L_FM = 1  # Scaling params for the feature matching loss
L_LPIPS = 1e-3  # Scaling params for the LPIPS loss

LR_G = 1e-5  # Learning rate for the generator
LR_D = 1e-5  # Learning rate for the discriminator
# LR_D = 3e-5

best_avg_rmses = 1

def write_log(log):
    print(log)
    if not os.path.exists("./save/"):
        os.mkdir("./save/")
    if not os.path.exists("./save/" + VERSION + "/"):
        os.mkdir("./save/" + VERSION + "/")
    my_log_file = open("./save/" + VERSION + '/train_9.txt', 'a')
    #     log="Train for batch %d,data loading time cost %f s"%(batch,start-time.time())
    my_log_file.write(log + '\n')
    my_log_file.close()
    return


### Generator ###
## ESRGAN for x8
import RRDBNet_arch as arch

model_G = arch.RRDBNetx4x2(1, 1, 64, 23, gc=32).cuda()

if torch.cuda.device_count() > 1:
    write_log("!!!Let's use" + str(torch.cuda.device_count()) + "GPUs!")
    model_G = nn.DataParallel(model_G, range(torch.cuda.device_count()))


### U-Net Discriminator ###
# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, wide=True,
                 preactivation=True, activation=nn.LeakyReLU(0.1, inplace=False), downsample=nn.AvgPool2d(2, stride=2)):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

        self.bn1 = self.which_bn(self.hidden_channels)
        self.bn2 = self.which_bn(out_channels)


    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = self.activation(x)
        else:
            h = x
        h = self.bn1(self.conv1(h))
        # h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h  # + self.shortcut(x)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, activation=nn.LeakyReLU(0.1, inplace=False),
                 upsample=nn.Upsample(scale_factor=2, mode='nearest')):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(out_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x):
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            # x = self.upsample(x)
        h = self.bn1(self.conv1(h))
        # h = self.activation(self.bn2(h))
        # h = self.conv2(h)
        # if self.learnable_sc:
        #     x = self.conv_sc(x)
        return h  # + x


class UnetD(torch.nn.Module):
    def __init__(self):
        super(UnetD, self).__init__()

        self.enc_b1 = DBlock(1, 64, preactivation=False)
        self.enc_b2 = DBlock(64, 128)
        self.enc_b3 = DBlock(128, 192)
        self.enc_b4 = DBlock(192, 256)
        self.enc_b5 = DBlock(256, 320)
        self.enc_b6 = DBlock(320, 384)
        # 这里320，384是否与图像大小有关？
        self.enc_out = nn.Conv2d(384, 1, kernel_size=1, padding=0)

        self.dec_b1 = GBlock(384, 320)
        self.dec_b2 = GBlock(320 * 2, 256)
        self.dec_b3 = GBlock(256 * 2, 192)
        self.dec_b4 = GBlock(192 * 2, 128)
        self.dec_b5 = GBlock(128 * 2, 64)
        self.dec_b6 = GBlock(64 * 2, 32)

        self.dec_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                # print(classname)
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        e1 = self.enc_b1(x)
        e2 = self.enc_b2(e1)
        e3 = self.enc_b3(e2)
        e4 = self.enc_b4(e3)
        e5 = self.enc_b5(e4)
        e6 = self.enc_b6(e5)
        e_out = self.enc_out(F.leaky_relu(e6, 0.1))

        d1 = self.dec_b1(e6)
        d2 = self.dec_b2(torch.cat([d1, e5], 1))
        d3 = self.dec_b3(torch.cat([d2, e4], 1))
        d4 = self.dec_b4(torch.cat([d3, e3], 1))
        d5 = self.dec_b5(torch.cat([d4, e2], 1))
        d6 = self.dec_b6(torch.cat([d5, e1], 1))

        d_out = self.dec_out(F.leaky_relu(d6, 0.1))

        return e_out, d_out, [e1, e2, e3, e4, e5, e6], [d1, d2, d3, d4, d5, d6]


model_D = UnetD().cuda()

## Optimizers
params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
opt_G = optim.Adam(params_G, lr=LR_G)

params_D = list(filter(lambda p: p.requires_grad, model_D.parameters()))
opt_D = optim.Adam(params_D, lr=LR_D)
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
## Load saved params
if START_ITER > 0:
    lm = torch.load('{}/checkpoint/v{}/model_G_i{:06d}{}.pth'.format(EXP_NAME, str(VERSION), START_ITER,current_time))
    model_G.load_state_dict(lm.state_dict(), strict=True)

lr_transforms = transforms.Compose([
    transforms.ToTensor()
])

hr_transforms = transforms.Compose([
    transforms.ToTensor()
])
def date_range(start_date, end_date):
    """This function takes a start date and an end date as datetime date objects.
    It returns a list of dates for each date in order starting at the first date and ending with the last date"""
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

def get_initial_date(rootdir):
    '''
    This function is used to extract the date that we plan to use in training
    '''
    _dates = []
    for date in dates:
        access_path = rootdir + "e09/da_pr_" + date.strftime("%Y%m%d") + "_e09.nc"
        if os.path.exists(access_path):
            _dates.append(date)
    return _dates
file_ACCESS_dir = "/scratch/iu60/xs5813/Processed_data_train/"
dates = date_range(START_TIME, END_TIME)
initial_dates = get_initial_date(file_ACCESS_dir)
generator1 = torch.Generator().manual_seed(42)
train_dates, val_dates = random_split(initial_dates, [int(len(initial_dates) * 0.8), len(initial_dates) - int(len(initial_dates) * 0.8)], generator = generator1)

train_data = ACCESS_AWAP_GAN(train_dates, START_TIME, END_TIME, lr_transform=lr_transforms, hr_transform=hr_transforms)
val_data = ACCESS_AWAP_GAN(val_dates, START_TIME, END_TIME, lr_transform=lr_transforms, hr_transform=hr_transforms)
write_log("Training Dataset length: " + str(len(train_data)))
write_log("Validation Dataset length: " + str(len(val_data)))
len_dataset = len(train_data) + len(val_data)
print(len_dataset)
# train_data, val_data = random_split(data_set, [int(len(data_set) * 0.8), len(data_set) - int(len(data_set) * 0.8)])

#######################################################################
train_dataloders = DataLoader(train_data, batch_size=NB_BATCH, shuffle=False, num_workers=NB_THREADS)
val_dataloders = DataLoader(val_data, batch_size=NB_BATCH, shuffle=False, num_workers=NB_THREADS)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Prepare directories
if not isdir('{}'.format(EXP_NAME)):
    mkdir('{}'.format(EXP_NAME))
if not isdir('{}/checkpoint'.format(EXP_NAME)):
    mkdir('{}/checkpoint'.format(EXP_NAME))
if not isdir('{}/result'.format(EXP_NAME)):
    mkdir('{}/result'.format(EXP_NAME))
if not isdir('{}/checkpoint/v{}'.format(EXP_NAME, str(VERSION))):
    mkdir('{}/checkpoint/v{}'.format(EXP_NAME, str(VERSION)))
if not isdir('{}/result/v{}'.format(EXP_NAME, str(VERSION))):
    mkdir('{}/result/v{}'.format(EXP_NAME, str(VERSION)))

## Some preparations
print('===> Training start')
l_accum = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
l_accum_n = 0.
dT = 0.
rT = 0.
n_mix = 0
accum_samples = 0

def SaveCheckpoint(i, model_G=None, model_D=None, opt_G=None, opt_D=None, best=False):
    str_best = ''
    if best:
        str_best = '_best'

    # 获取当前的日期和时间
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    i = i + START_ITER + 1
    torch.save(model_G, '{}/checkpoint/v{}/model_G_i{:06d}_{}.pth'.format(EXP_NAME, str(VERSION), i,current_time))
    torch.save(opt_G, '{}/checkpoint/v{}/opt_G_i{:06d}_{}_no_log1p.pth'.format(EXP_NAME, str(VERSION), i,current_time))
    write_log("Checkpoint saved with timestamp " + current_time)



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

print('model_version is: ', VERSION)
print('train batch size is ', NB_BATCH)
### TRAINING
for itera in range(NB_Iteration):
    print("epoch:",itera)
    batch = 0
    for batch, (lr, hr, _, _, _, _) in enumerate(train_dataloders):

        model_G.train()
        model_D.train()

        # crop size 576
        # 688和880是什么？是(110,86)的8倍？ 那raw_data的size是(110,86)/1.5 = (73, 57)?
        np.random.seed(batch)
        #hh = np.random.randint(0, 688 - PATCH_SIZE + 1)
        #hw = np.random.randint(0, 880 - PATCH_SIZE + 1)
        hh = np.random.randint(0, 408 - PATCH_SIZE + 1)
        hw = np.random.randint(0, 264 - PATCH_SIZE + 1)
        # crop the patch
        hr = hr[:, :, hh:(hh + PATCH_SIZE), hw:(hw + PATCH_SIZE)]
        lr = lr[:, :, int(hh / UPSCALE):int((hh + PATCH_SIZE) / UPSCALE),
             int(hw / UPSCALE):int((hw + PATCH_SIZE) / UPSCALE)]
        ## TRAIN D
        st = time.time()

        batch_L = Variable(lr).cuda()
        batch_H = Variable(hr).cuda()

        dT += time.time() - st

        st = time.time()
        opt_G.zero_grad()
        opt_D.zero_grad()

        # G
        # Generated image G(Il)  batch_H(Ig)
        batch_S = model_G(batch_L).detach()

        ###在这个地方加上我的loss

        # D
        # e_S = Denc (G(Il))  e_H = Denc(Ig)
        e_S, d_S, _, _ = model_D(batch_S)
        e_H, d_H, _, _ = model_D(batch_H)

        # D Loss, for encoder end and decoder end
        loss_D_Enc_S = torch.nn.ReLU()(1.0 + e_S).mean()
        loss_D_Enc_H = torch.nn.ReLU()(1.0 - e_H).mean()

        # loss_D_Enc_H = (-1.0) * torch.log(torch.nn.ReLU()(e_H)).mean()
        # loss_D_Enc_S = (-1.0) * torch.log(torch.nn.ReLU()(1.0 - e_S)).mean()  


        loss_D_Dec_S = torch.nn.ReLU()(1.0 + d_S).mean()
        loss_D_Dec_H = torch.nn.ReLU()(1.0 - d_H).mean()

        # loss_D_Dec_S = (-1.0) * torch.log(torch.nn.ReLU()(d_H)).mean()
        # loss_D_Dec_H = (-1.0) * torch.log(torch.nn.ReLU()(1.0 - d_S)).mean()  
        # Firstly E[Encoder(High-res)] + E[Decoder[High-res]]
        loss_D = loss_D_Enc_H + loss_D_Dec_H

        # CutMix for consistency loss
        batch_S_CutMix = batch_S.clone()

        # probability of doing cutmix
        p_mix = batch / 100000
        if p_mix > 0.5:
            p_mix = 0.5

        if torch.rand(1) <= p_mix:
            n_mix += 1
            r_mix = torch.rand(1)  # real/fake ratio

            bbx1, bby1, bbx2, bby2 = rand_bbox(batch_S_CutMix.size(), r_mix)
            batch_S_CutMix[:, :, bbx1:bbx2, bby1:bby2] = batch_H[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            r_mix = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_S_CutMix.size()[-1] * batch_S_CutMix.size()[-2]))

            e_mix, d_mix, _, _ = model_D(batch_S_CutMix)

            loss_D_Enc_S = torch.nn.ReLU()(1.0 + e_mix).mean()
            loss_D_Dec_S = torch.nn.ReLU()(1.0 + d_mix).mean()

            d_S[:, :, bbx1:bbx2, bby1:bby2] = d_H[:, :, bbx1:bbx2, bby1:bby2]
            loss_D_Cons = F.mse_loss(d_mix, d_S)

            loss_D += loss_D_Cons
            l_accum[5] += torch.mean(loss_D_Cons).item()

        loss_D += loss_D_Enc_S + loss_D_Dec_S

        # Update
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(params_D, 0.1)
        opt_D.step()
        rT += time.time() - st

        # for monitoring
        l_accum[0] += loss_D.item()
        l_accum[1] += torch.mean(e_H).item()
        l_accum[2] += torch.mean(e_S).item()
        l_accum[3] += torch.mean(d_H).item()
        l_accum[4] += torch.mean(d_S).item()

        ## TRAIN G
        st = time.time()

        batch_H = Variable(hr).cuda()
        batch_L = Variable(lr).cuda()

        batch_L = torch.clamp(batch_L, 0, 1)

        dT += time.time() - st

        st = time.time()
        opt_G.zero_grad()
        opt_D.zero_grad()

        batch_S = model_G(batch_L)

        # Pixel loss
        loss_Pixel = Huber(batch_S, batch_H)
        loss_G = loss_Pixel

        # GAN losses
        e_S, d_S, e_Ss, d_Ss = model_D(batch_S)
        _, _, e_Hs, d_Hs = model_D(batch_H)

        loss_Advs = []
        loss_Advs += [torch.nn.ReLU()(1.0 - e_S).mean() * L_ADV]
        loss_Advs += [torch.nn.ReLU()(1.0 - d_S).mean() * L_ADV]
        loss_Adv = torch.mean(torch.stack(loss_Advs))
        loss_G += loss_Adv
        l_accum[9] += loss_Adv.item()

        # Update
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(params_G, 0.1)
        opt_G.step()
        rT += time.time() - st

        # For monitoring
        l_accum[6] += loss_Pixel.item()
        l_accum[10] += loss_G.item()
        accum_samples += NB_BATCH

    # Show information
    write_log(
        "{} {} | Iter:{:6d}, Sample:{:6d}, D:{:.2e}, DEnc(1/-1):{:.2f}/{:.2f}, DDec(1/-1):{:.2f}/{:.2f}, "
        "nMix:{:2d}, Dcons:{:.2e}, GPixel:{:.2e}, GLPIPS:{:.2e}, GFM:{:.2e}, GAdv:{:.2e}, G:{:.2e}, dT:{:.4f}, huber:{:.4f}"
        "rT:{:.4f}".format(
            EXP_NAME, VERSION, batch, accum_samples, l_accum[0] / len_dataset, l_accum[1] / len_dataset,
                                                     l_accum[2] / len_dataset, l_accum[3] / len_dataset,
                                                     l_accum[4] / len_dataset, n_mix, l_accum[5] / (n_mix + 1e-12),
            (l_accum[6] - l_accum[7] - l_accum[8] - l_accum[9]) / len_dataset, l_accum[7] / len_dataset,
                                                     l_accum[8] / len_dataset, l_accum[9] / len_dataset,
            l_accum[10] / len_dataset, dT / len_dataset, rT / len_dataset,loss_Pixel))
    l_accum = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    l_accum_n = 0.
    n_mix = 0
    dT = 0.
    rT = 0.

    ## Save models per iteration
    SaveCheckpoint(itera, model_G, model_D, opt_G, opt_D, best=False)

    ## Validate per Iteration
    with torch.no_grad():
        model_G.eval()

        # Test for validation nc
        rmses = []
        lpips = []
        maes = []

        # Load test image
        for i_val, (lr_val, hr_val, _, _, _, _) in enumerate(val_dataloders):
            batch_H = np.asarray(hr_val).astype(np.float32)  # HxWxC
            batch_L = np.asarray(lr_val).astype(np.float32)


            # Data
            # batch_H = Variable(torch.from_numpy(val_H)).cuda()
            batch_L = Variable(torch.from_numpy(batch_L)).cuda()
            batch_L = torch.clamp(batch_L, 0., 1.)
            # batch_L = torch.clamp(torch.round(batch_L * 255) / 255.0, 0, 1)

            # Forward
            batch_Out = model_G(batch_L)

            # Output
            batch_Out = batch_Out.cpu().data.numpy()
            batch_Out = np.clip(batch_Out, 0., 1.)  # BXCxHxW 16*1*688*880
            batch_Out = np.squeeze(batch_Out, axis=1)  # 16*688*880
            batch_Out = batch_Out.transpose(1, 2, 0) # 688*880*16
            batch_Out = cv2.resize(batch_Out, (267, 413), interpolation=cv2.INTER_CUBIC)
            if len(batch_Out.shape) == 2:
                batch_Out = batch_Out.reshape(batch_Out.shape[0], batch_Out.shape[1], 1)
            # batch_Out = np.transpose(batch_Out, [2, 0, 1])
            batch_Out = batch_Out.transpose(2, 0, 1) # 3 * 413 * 267

            # RMSE
            img_gt = np.squeeze(batch_H, axis=1)
            img_gt = np.expm1(img_gt * 7)
            img_target = np.expm1(batch_Out * 7)
            rmses.append(RMSE(img_gt, img_target, 0))
            maes.append(MAE(img_gt, img_target, 0))
    avg_rmse = np.mean(np.asarray(rmses))
    avg_mae = np.mean(np.asarray(maes))
    write_log('AVG RMSE: Validation: {:.4f}, AVG MAE: Validation: {:.4f}'.format(avg_rmse, avg_mae))

    #write_log('AVG RMSE: Validation: {}'.format(np.mean(np.asarray(rmses))))

    # Save best model
    if np.mean(np.asarray(rmses)) < best_avg_rmses:
        best_avg_rmses = np.mean(np.asarray(rmses))
        SaveCheckpoint(itera, model_G, model_D, opt_G, opt_D, best=True)