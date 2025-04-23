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
import pretrain_m_arch as G_arch
import RRDBNet_arch as pre_arch

from utils2 import Huber, ACCESS_AWAP_GAN, RMSE,  MAE, log_loss2, generate_sample,generate_3_channels, CRPS_from_distribution

import cv2

### USER PARAMS ###
START_TIME = date(1990, 1, 1)
END_TIME = date(2005, 12, 31)

EXP_NAME = "/scratch/iu60/xs5813/EXTREME"
VERSION = "version_0"
UPSCALE  = 8 # upscaling factor

NB_BATCH = 18  # mini-batch
NB_Iteration = 20
# 训练策略相关参数
STAGE1_EPOCHS = 10  # 训练 `new_output_conv` 的前几轮
STAGE2_EPOCHS = NB_Iteration - STAGE1_EPOCHS  # 训练前面层
PATCH_SIZE = 256  # Training patch size
NB_THREADS = 36

START_ITER = 0  # Set 0 for from scratch, else will load saved params and trains further

L_ADV = 1e-3  # Scaling params for the Adv loss
L_FM = 1  # Scaling params for the feature matching loss
L_LPIPS = 1e-3  # Scaling params for the LPIPS loss

LR_G = 1e-5  # Learning rate for the generator
# LR_D = 3e-5

best_avg_rmses = 1

def write_log(log):
    print(log)
    if not os.path.exists("./save/"):
        os.mkdir("./save/")
    if not os.path.exists("./save/" + VERSION + "/"):
        os.mkdir("./save/" + VERSION + "/")
    my_log_file = open("./save/" + VERSION + '/train_prgan_3.txt', 'a')
    #     log="Train for batch %d,data loading time cost %f s"%(batch,start-time.time())
    my_log_file.write(log + '\n')
    my_log_file.close()
    return


### Generator ###
## ESRGAN for x8
#pretrain model path
OSAVE_PREFIX = "/scratch/iu60/xs5813/DESRGAN_YAO/"
OMODEL_PREFIX = OSAVE_PREFIX + "checkpoint/voriginal_DESRGAN/"

#summer: model_G_name = "model_G_i000004_20241218-214333"
model_G_name = "model_G_i000003_20250416-161818"
#d_bug = "/scratch/iu60/xs5813/DESRGAN_ORIGINAL/DESRGAN/checkpoint/vTestRefactored/model_G_i000004_best_20240219-114819.pth"
model_path = OMODEL_PREFIX +  "/" + model_G_name + ".pth"
model_G = G_arch.ModifiedRRDBNet(model_path, 1, 3, 64, 23, gc=32).cuda()
# 如果 model_G 是 DataParallel，访问 .module 才能看到实际的模型
real_model_G = model_G.module if isinstance(model_G, nn.DataParallel) else model_G

if torch.cuda.device_count() > 1:
    write_log("!!!Let's use" + str(torch.cuda.device_count()) + "GPUs!")
    model_G = nn.DataParallel(model_G, range(torch.cuda.device_count()))

print("pretrain")

## Optimizers
# 冻结所有层，除 `new_output_conv` 外
for name, param in real_model_G.named_parameters():
    param.requires_grad = False if "new_output_conv" not in name else True

# **定义优化器**
opt_G_stage1 = optim.Adam(real_model_G.new_output_conv.parameters(), lr=LR_G)  # 只优化最后一层
opt_G_stage2 = optim.Adam([p for name, p in real_model_G.named_parameters() if "new_output_conv" not in name], lr=LR_G)  # 只优化前面的层


current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
## Load saved params
if START_ITER > 0:
    lm = torch.load('{}/checkpoint/v{}/model_G_i{:06d}.pth'.format(EXP_NAME, str(VERSION), START_ITER,current_time))
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
file_ACCESS_dir = "/scratch/iu60/xs5813/Processed_data_q/"
dates = date_range(START_TIME, END_TIME)
initial_dates = get_initial_date(file_ACCESS_dir)
generator1 = torch.Generator().manual_seed(42)
train_dates, val_dates = random_split(initial_dates, [int(len(initial_dates) * 0.8), len(initial_dates) - int(len(initial_dates) * 0.8)], generator = generator1)

train_data = ACCESS_AWAP_GAN(train_dates, START_TIME, END_TIME, lr_transform=lr_transforms, hr_transform=hr_transforms)
val_data = ACCESS_AWAP_GAN(val_dates, START_TIME, END_TIME, lr_transform=lr_transforms, hr_transform=hr_transforms,validation = True)
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
    """
    Save the model and optimizer params.

    Args:
        i: current iteration
        model_G: generator model
        model_D: discriminator model
        opt_G: generator optimizer
        opt_D: discriminator optimizer
        best: whether to save the best model
    """
    str_best = ''
    if best:
        str_best = '_best'

    # 获取当前的日期和时间
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    i = i + START_ITER + 1

    torch.save(model_G, '{}/checkpoint/v{}/model_G_i{:06d}_{}_with_huber.pth'.format(EXP_NAME, str(VERSION), i,current_time))
    torch.save(optimizer, '{}/checkpoint/v{}/opt_G_i{:06d}_{}.pth'.format(EXP_NAME, str(VERSION), i,current_time))
    write_log("Checkpoint saved with timestamp " + current_time)

print('model_version is: ', VERSION)
print('train batch size is ', NB_BATCH)
### TRAINING
for itera in range(NB_Iteration):
    print("epoch:",itera)

    # **确定当前训练阶段**
    if itera < STAGE1_EPOCHS:
        optimizer = opt_G_stage1  # 第一阶段
        write_log("Training Stage 1: Only new_output_conv")
    else:
        if itera == STAGE1_EPOCHS:  # **切换到第二阶段**
            for name, param in real_model_G.named_parameters():
                param.requires_grad = True if "new_output_conv" not in name else False
        optimizer = opt_G_stage2  # 第二阶段
        write_log("Training Stage 2: Fine-tuning front layers")

    for batch, (lr, hr, _, _, _, _) in enumerate(train_dataloders):

        model_G.train()

        # crop size 576
        # 688和880是什么？是(110,86)的8倍？ 那raw_data的size是(110,86)/1.5 = (73, 57)?
        np.random.seed(batch)
        #hh = np.random.randint(0, 688 - PATCH_SIZE + 1)
        #hw = np.random.randint(0, 880 - PATCH_SIZE + 1)
        hh = np.random.randint(0, 257 - PATCH_SIZE + 1)
        hw = np.random.randint(0, 257 - PATCH_SIZE + 1)
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
        optimizer.zero_grad()

        # G
        # Generated image G(Il)  batch_H(Ig)

        ###在这个地方加上我的loss
        ## TRAIN G
        st = time.time()

        batch_H = Variable(hr).cuda()
        batch_L = Variable(lr).cuda()

        batch_L = torch.clamp(batch_L, 0, 1)

        dT += time.time() - st

        st = time.time()
        optimizer.zero_grad()

        output = model_G(batch_L)
        loss_Log = log_loss2(batch_H, output)
        batch_S = generate_sample(output)

        # Pixel loss
        loss_Pixel = Huber(batch_S, batch_H)
        #loss_G = loss_Pixel
        loss_G = 0
        loss_G += 1e-4*loss_Log #+ L_CRPS * crps

        # Update
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(real_model_G.parameters(), 0.1)
        optimizer.step()
        rT += time.time() - st
        print(f"Huber Loss: {loss_Pixel.item()}")
        print(f"Log Loss: {loss_Log .item()}")
        # For monitoring
        l_accum[6] += loss_Pixel.item()
        l_accum[10] += loss_G.item()
        accum_samples += NB_BATCH

    # Show information
    write_log(
        "{} {} | Iter:{:6d}, Sample:{:6d}, D:{:.2e}, DEnc(1/-1):{:.2f}/{:.2f}, DDec(1/-1):{:.2f}/{:.2f}, "
        "nMix:{:2d}, Dcons:{:.2e}, GPixel:{:.2e}, GLPIPS:{:.2e}, GFM:{:.2e}, GAdv:{:.2e}, G:{:.2e}, dT:{:.4f}, "
        "rT:{:.4f}, Huber_loss{:.6f}, Log_loss:{:.6f}".format(
            EXP_NAME, VERSION, batch, accum_samples, l_accum[0] / len_dataset, l_accum[1] / len_dataset,
                                                     l_accum[2] / len_dataset, l_accum[3] / len_dataset,
                                                     l_accum[4] / len_dataset, n_mix, l_accum[5] / (n_mix + 1e-12),
            (l_accum[6] - l_accum[7] - l_accum[8] - l_accum[9]) / len_dataset, l_accum[7] / len_dataset,
                                                     l_accum[8] / len_dataset, l_accum[9] / len_dataset,
            l_accum[10] / len_dataset, dT / len_dataset, rT / len_dataset, loss_Pixel.item(), loss_Log .item()))
    l_accum = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    l_accum_n = 0.
    n_mix = 0
    dT = 0.
    rT = 0.

    ## Save models per iteration
    SaveCheckpoint(itera, model_G, optimizer, best=False)

    ## Validate per Iteration
    with torch.no_grad():
        model_G.eval()

        # Test for validation nc
        rmses = []
        lpips = []
        maes = []
        crps = []

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
            output = model_G(batch_L)
            batch_Out = generate_sample(output)
            p_pred, alpha_pred, beta_pred = generate_3_channels(output)
            # Output
            def resize_out(batch_Out):
                batch_Out = batch_Out.cpu().data.numpy()
                batch_Out = np.clip(batch_Out, 0., 1.)  # BXCxHxW 16*1*688*880
                batch_Out = np.squeeze(batch_Out, axis=1)  # 16*688*880
                batch_Out = batch_Out.transpose(1, 2, 0) # 688*880*16

                batch_Out = cv2.resize(batch_Out, (257, 257), interpolation=cv2.INTER_CUBIC)
                if len(batch_Out.shape) == 2:
                    batch_Out = batch_Out.reshape(batch_Out.shape[0], batch_Out.shape[1], 1)
                # batch_Out = np.transpose(batch_Out, [2, 0, 1])
                batch_Out = batch_Out.transpose(2, 0, 1) # 3 * 413 * 267
                return batch_Out

            batch_Out = resize_out(batch_Out)
            p_pred = resize_out(p_pred)
            alpha_pred = resize_out(alpha_pred)
            beta_pred = resize_out(beta_pred)

            # RMSE
            img_gt = np.squeeze(batch_H, axis=1)
            img_gt = np.expm1(img_gt * 7)
            print("img_gt",img_gt)
            img_target = np.expm1(batch_Out * 7)
            rmses.append(RMSE(img_gt, img_target, 0))
            maes.append(MAE(img_gt, img_target, 0))
            crps.append(CRPS_from_distribution(p_pred, alpha_pred, beta_pred, img_gt))
    avg_rmse = np.mean(np.asarray(rmses))
    avg_mae = np.mean(np.asarray(maes))
    avg_crps = np.mean(np.asarray(crps))
    write_log('AVG RMSE: Validation: {:.4f}, AVG MAE: Validation: {:.4f}, AVG CRPS: Validation: {:.4f}, merge metric: Validation: {:.4f}'.format(avg_rmse, avg_mae, avg_crps, avg_mae+1.3*avg_crps))
#原尺度，去掉log1p
# validation mask ocean data, 防止数据不好的地方影响到我的model
# extreme
# probability 是要在当月的尺度算
    # Save best model
    if np.mean(np.asarray(rmses)) < best_avg_rmses:
        best_avg_rmses = np.mean(np.asarray(rmses))
        SaveCheckpoint(itera, model_G, optimizer, best=True)

# 先固定前面的weight，训练几轮最后一层，然后再调节他前面几层
