# Imports 
'''
如果你的目标是使用这个GAN进行降雨预测，你的训练循环可能看起来是这样的：

生成器阶段：生成器接受低分辨率的气象数据，输出降雨概率和伽马分布参数的估计值。
判别器阶段：判别器接受生成器的输出和真实的高分辨率降雨数据，然后被训练以区分两者。
损失计算和优化：计算生成器和判别器的损失，并更新它们的参数。
记住，对于GANs的训练，平衡生成器和判别器的性能是至关重要的。
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time
from os import mkdir
from os.path import isdir
from torch.utils.data import random_split, DataLoader
from datetime import date
from utils import Huber, ACCESS_AWAP_GAN, RMSE, MAE
import cv2
import RRDBNet_arch as G_arch
import UNET_arch as D_arch

### USER PARAMS ###
START_TIME = date(1990, 1, 1)
END_TIME = date(2001, 12, 31)

SAVE_PREFIX = "/scratch/iu60/xs5813/DESRGAN_ORIGINAL/"
EXP_NAME = "DESRGAN"
VERSION = "TestRefactored"
UPSCALE = 8  # upscaling factor 40km->5km

NB_BATCH = 3  # mini-batch
NB_Iteration = 10 # Number of iterations (epochs)
#PATCH_SIZE = 576  # Training patch size
PATCH_SIZE = 256
NB_THREADS = 36

START_ITER = 0  # Set 0 for from scratch, else will load saved params and trains further

L_ADV = 1e-3  # Scaling params for the Adv loss
L_FM = 1  # Scaling params for the feature matching loss
L_LPIPS = 1e-3  # Scaling params for the LPIPS loss

LR_G = 1e-5  # Learning rate for the generator
LR_D = 1e-5  # Learning rate for the discriminator

def prepare_directories():
    """
        Set up the log file and checkpoint/result folders.
    """

    # Set up log file
    if not os.path.exists("./save/"):
        os.mkdir("./save/")
    if not os.path.exists("./save/" + VERSION + "/"):
        os.mkdir("./save/" + VERSION + "/")

    # Set up checkpoint and result folder
    if not isdir('{}'.format(SAVE_PREFIX + EXP_NAME)):
        mkdir('{}'.format(SAVE_PREFIX + EXP_NAME))
    if not isdir('{}/checkpoint'.format(SAVE_PREFIX + EXP_NAME)):
        mkdir('{}/checkpoint'.format(SAVE_PREFIX + EXP_NAME))
    if not isdir('{}/result'.format(SAVE_PREFIX + EXP_NAME)):
        mkdir('{}/result'.format(SAVE_PREFIX + EXP_NAME))
    if not isdir('{}/checkpoint/v{}'.format(SAVE_PREFIX + EXP_NAME, str(VERSION))):
        mkdir('{}/checkpoint/v{}'.format(SAVE_PREFIX + EXP_NAME, str(VERSION)))
    if not isdir('{}/result/v{}'.format(SAVE_PREFIX + EXP_NAME, str(VERSION))):
        mkdir('{}/result/v{}'.format(SAVE_PREFIX + EXP_NAME, str(VERSION)))

def write_log(log):
    """
        Write log to the log file.

        Args:
            log: log to be written
    """
    print(log)
    my_log_file = open("./save/" + VERSION + '/train.txt', 'a')
    my_log_file.write(log + '\n')
    my_log_file.close()

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

    i = i + START_ITER + 1

    torch.save(model_G, '{}/checkpoint/v{}/model_G_i{:06d}{}.pth'.format(SAVE_PREFIX + EXP_NAME, str(VERSION), i, str_best))
    torch.save(model_D, '{}/checkpoint/v{}/model_D_i{:06d}{}.pth'.format(SAVE_PREFIX + EXP_NAME, str(VERSION), i, str_best))

    torch.save(opt_G, '{}/checkpoint/v{}/opt_G_i{:06d}{}.pth'.format(SAVE_PREFIX + EXP_NAME, str(VERSION), i, str_best))
    torch.save(opt_D, '{}/checkpoint/v{}/opt_D_i{:06d}{}.pth'.format(SAVE_PREFIX + EXP_NAME, str(VERSION), i, str_best))
    write_log("Checkpoint saved")

def rand_bbox(size, lam):
    """
    Generate random bounding box for CutMix regularization.

    Args:
        size: (N, C, H, W)
        lam: lambda value from beta distribution.

    Returns:
        Bounding box.
    """
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

def cutmix(batch_S_CutMix, batch_H, d_S, d_H):
    """
        Generate random bounding box for CutMix regularization and perform CutMix.

        Args:
            batch_S_CutMix: batch of S images
            batch_H: batch of H images
            d_S: discriminator output for S images
            d_H: discriminator output for H images

        Returns:
            Cutmix loss

    """

    print("Performing CutMix")

    r_mix = torch.rand(1)  # real/fake ratio
    # adjust lambda to exactly match pixel ratio
    # r_mix = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_S_CutMix.size()[-1] * batch_S_CutMix.size()[-2])) # Is this necessary?

    bbx1, bby1, bbx2, bby2 = rand_bbox(batch_S_CutMix.size(), r_mix)
    batch_S_CutMix[:, :, bbx1:bbx2, bby1:bby2] = batch_H[:, :, bbx1:bbx2, bby1:bby2]

    e_mix, d_mix, _, _ = model_D(batch_S_CutMix)

    loss_D_Enc_S = torch.nn.ReLU()(1.0 + e_mix).mean()
    loss_D_Dec_S = torch.nn.ReLU()(1.0 + d_mix).mean()

    d_S[:, :, bbx1:bbx2, bby1:bby2] = d_H[:, :, bbx1:bbx2, bby1:bby2]
    loss_D_Cons = F.mse_loss(d_mix, d_S)

    return loss_D_Cons + loss_D_Enc_S + loss_D_Dec_S

def get_patches(batch, lr, hr, patch_size, scaling_factor):

    np.random.seed(batch)
    # 确保随机坐标加上补丁尺寸不会超出边界
    max_hh = min(lr.shape[2] * scaling_factor, hr.shape[2]) - patch_size
    max_hw = min(lr.shape[3] * scaling_factor, hr.shape[3]) - patch_size

    hh = np.random.randint(0, max_hh + 1)
    hw = np.random.randint(0, max_hw + 1)

    hr = hr[:, :, hh:(hh + patch_size), hw:(hw + patch_size)]
    lr = lr[:, :, int(hh / scaling_factor):int((hh + patch_size) / scaling_factor), int(hw / scaling_factor):int((hw + patch_size) / scaling_factor)]

    return lr, hr
''''''
def generate_sample(rain_prob, gamma_shape, gamma_scale):
    # 生成一个样本估计值
    rain_sample = torch.zeros_like(rain_prob)
    rain_mask = torch.bernoulli(rain_prob)  # 根据降雨概率生成一个掩码
    rain_sample[rain_mask.bool()] = torch.distributions.Gamma(gamma_shape, gamma_scale).sample()[rain_mask.bool()]
    return rain_sample

def discriminator_loss(model_G, model_D, batch_L, batch_H):
    """
        Train the discriminator on the batch of lr and hr images.

        Args:
            model_G: generator model
            model_D: discriminator model
            batch_L: batch of lr images
            batch_H: batch of hr images
        
        Returns:
            loss_D: discriminator loss
    """
    # Get super-resolved batch (G(lr) ~ hr)
    #batch_S = model_G(batch_L)
    # 获取生成器的输出
    rain_prob, gamma_shape, gamma_scale = model_G(batch_L)
    # 生成样本估计值
    batch_S = generate_sample(rain_prob, gamma_shape, gamma_scale)
    # Run the hr, and predicted sr images through the discriminator
    e_S, d_S, _, _ = model_D(batch_S)
    e_H, d_H, _, _ = model_D(batch_H)

    # D Loss, for encoder end and decoder end
    loss_D_Enc_S = torch.nn.ReLU()(1.0 + e_S).mean()
    loss_D_Enc_H = torch.nn.ReLU()(1.0 - e_H).mean()

    loss_D_Dec_S = torch.nn.ReLU()(1.0 + d_S).mean()
    loss_D_Dec_H = torch.nn.ReLU()(1.0 - d_H).mean()

    loss_D = loss_D_Enc_H + loss_D_Dec_H

    # probability of doing cutmix
    # p_mix = batch / 100000  
    # if p_mix > 0.5:
    p_mix = 0.5

    # Perform CutMix
    if torch.rand(1) <= p_mix:
        loss_D += cutmix(batch_S.clone(), batch_H.clone(), d_S.clone(), d_H.clone()) # Discriminator loss for fake images after CutMix
    else:
        loss_D += loss_D_Enc_S + loss_D_Dec_S # Discriminator loss for fake images
    
    return loss_D

def discriminator_iteration(model_G, model_D, opt_G, opt_D, lr, hr) -> None:
    """
        Train the discriminator on the batch of lr and hr images.

        Args:
            model_G: generator model
            model_D: discriminator model
            opt_G: generator optimizer
            opt_D: discriminator optimizer
            lr: batch of lr images
            hr: batch of hr images
    """
    
    # Get data as cuda variables
    batch_L = Variable(lr).cuda()
    batch_H = Variable(hr).cuda()

    opt_G.zero_grad()
    opt_D.zero_grad()

    # Train discriminator
    loss_D = discriminator_loss(model_G, model_D, batch_L, batch_H)

    # Update
    loss_D.backward()
    torch.nn.utils.clip_grad_norm_(params_D, 0.1)
    opt_D.step()

def generator_loss(model_G, model_D, batch_L, batch_H):
    """
        Train the generator on the batch of lr and hr images.

        Args:
            model_G: generator model
            model_D: discriminator model
            batch_L: batch of lr images
            batch_H: batch of hr images
        
        Returns:
            loss_G: generator loss
    """
    
    # Get super-resolved batch (G(lr) ~ hr)
    #batch_S = model_G(batch_L)
       # 获取生成器的输出
    rain_prob, gamma_shape, gamma_scale = model_G(batch_L)
    # 生成样本估计值
    batch_S = generate_sample(rain_prob, gamma_shape, gamma_scale)

    # Run the hr, and predicted sr images through the discriminator
    e_S, d_S, _, _ = model_D(batch_S)

    # Pixel loss
    loss_Pixel = Huber(batch_S, batch_H)
    loss_G = loss_Pixel

    # GAN losses
    loss_Advs = []
    loss_Advs += [torch.nn.ReLU()(1.0 - e_S).mean() * L_ADV]
    loss_Advs += [torch.nn.ReLU()(1.0 - d_S).mean() * L_ADV]
    loss_Adv = torch.mean(torch.stack(loss_Advs))

    loss_G += loss_Adv

    return loss_G

def generator_iteration(model_G, model_D, opt_G, opt_D, lr, hr) -> None:
    """
        Train the generator on the batch of lr and hr images.

        Args:
            model_G: generator model
            model_D: discriminator model
            opt_G: generator optimizer
            opt_D: discriminator optimizer
            lr: batch of lr images
            hr: batch of hr images
    """
    
    # Get data as cuda variables
    batch_L = Variable(lr).cuda()
    batch_H = Variable(hr).cuda()

    opt_G.zero_grad()
    opt_D.zero_grad()

    # Train generator
    loss_G = generator_loss(model_G, model_D, batch_L, batch_H)

    # Update
    loss_G.backward()
    torch.nn.utils.clip_grad_norm_(params_G, 0.1)
    opt_G.step()

def get_performance(model_G, dataloader, epoch, batch=-1):
    """
    Test the model on the test dataset.

    Args:
        model_G: generator model
        dataloader: test dataloader
        epoch: current epoch (used for logging)
        batch: current batch (used for logging)

    Returns:
        rmse: average rmse
        mae: average mae
    """
    itr_test_time = time.time()

    # Disable gradient calculation
    with torch.no_grad():
        
        # Set models to eval mode
        model_G.eval()
        
        # Test for validation
        rmses = []
        maes = []

        # Load test image
        for lr_val, hr_val, _, _ in dataloader:
            print(f"LR shape: {lr_val.shape}, HR shape: {hr_val.shape}") 
            batch_H = np.asarray(hr_val).astype(np.float32)
            batch_L = np.asarray(lr_val).astype(np.float32)

            # Convert to torch.Tensor
            batch_L = Variable(torch.from_numpy(batch_L)).cuda()
            batch_L = torch.clamp(batch_L, 0., 1.)

            # Forward pass through generator
            rain_prob, gamma_shape, gamma_scale = model_G(batch_L)

            # Generate sample estimate from the distribution
            batch_Out = generate_sample(rain_prob, gamma_shape, gamma_scale).cpu().data.numpy()
            batch_Out = np.clip(batch_Out, 0., 1.)
            # Process output and ground truth for metric calculation
            batch_Out = np.squeeze(batch_Out, axis = 1)
            #batch_Out = np.transpose(batch_Out, [1, 2, 0])
            batch_Out = batch_Out.transpose(1, 2, 0) # 688*880*16

            batch_Out = cv2.resize(batch_Out, (366, 381), interpolation=cv2.INTER_CUBIC)
            if len(batch_Out.shape) == 2:
                batch_Out = batch_Out.reshape(batch_Out.shape[0], batch_Out.shape[1], 1)
            #batch_Out = np.transpose(batch_Out, [2, 0, 1])
            batch_Out = batch_Out.transpose(2, 0, 1)
            print('The final batch_out: ', batch_Out.shape)
            print('The final batch_H: ', batch_H.shape)
            img_gt = np.squeeze(batch_H)
            img_gt = np.expm1(img_gt * 4)  # Revert preprocessing on ground truth
            img_target = np.expm1(batch_Out * 4)  # Revert preprocessing on prediction

            rmses.append(RMSE(img_gt, img_target, 0))
            maes.append(MAE(img_gt, img_target, 0))
            
        rmse = np.mean(np.asarray(rmses))
        mae = np.mean(np.asarray(maes))
        
        write_log(f"RMSE: {rmse} MAE: {mae} for epoch {epoch}{f' batch {batch}' if batch != -1 else ''}. Completed in {time.time() - itr_test_time} seconds")

        return rmse, mae


def train_GAN(model_G, model_D, opt_G, opt_D, train_dataloders, val_dataloders, test_dataloders, nb_epoch):
    """
        Train the GAN model.

        Args:
            model_G: generator model
            model_D: discriminator model
            opt_G: generator optimizer
            opt_D: discriminator optimizer
            train_dataloders: train dataloader
            val_dataloders: validation dataloader
            test_dataloders: test dataloader
            nb_epoch: number of epochs to train for
    """

    best_avg_mae = 1000000

    training_start_time = time.time()

    ### TRAINING
    for epoch in range(nb_epoch):
        print("epoch right now:", epoch)

        epoch_start_time = time.time()

        for batch_number, (lr, hr, _, _) in enumerate(train_dataloders):
            batch_start_time = time.time()

            # Set models to training mode
            model_G.train()
            model_D.train()

            # Get patches for training
            lr, hr = get_patches(batch_number, lr, hr, PATCH_SIZE, UPSCALE)
            '''
            Post-patch LR shape: torch.Size([3, 1, 32, 32])
            Post-patch HR shape: torch.Size([3, 1, 256, 256])
            '''

            # Train discriminator then generator
            discriminator_iteration(model_G, model_D, opt_G, opt_D, lr, hr)
            generator_iteration(model_G, model_D, opt_G, opt_D, lr, hr)


            batch_end_time = time.time()
            write_log(f"Batch {batch_number} completed in {batch_end_time - batch_start_time} seconds")


            # Test the performance per 10 iterations
            if batch_number % 10 == 0:
                print("start get the performance")
                get_performance(model_G, test_dataloders, epoch, batch_number)
                break
                

        epoch_end_time = time.time()
        write_log(f"Epoch {epoch} completed in {epoch_end_time - epoch_start_time} seconds")

        ## Save models per iteration
        SaveCheckpoint(epoch, model_G, model_D, opt_G, opt_D)

        # Validate model
        print("start get validation the performance")
        val_rmse, val_mae = get_performance(model_G, val_dataloders, epoch)


        # Save best model
        if val_mae < best_avg_mae:
            best_avg_mae = val_mae
            SaveCheckpoint(epoch, model_G, model_D, opt_G, opt_D, best=True)
        
        write_log(f"Best MAE: {best_avg_mae}")

    write_log(f"Training completed in {time.time() - training_start_time} seconds")
    
if __name__ == "__main__":
    # Prepare directories
    prepare_directories()

    ### Generator ###
    model_G = G_arch.RRDBNetx4x2(1, 3, 64, 23, gc=32).cuda()
    if torch.cuda.device_count() > 1:
        write_log("Using " + str(torch.cuda.device_count()) + " GPUs!")
        model_G = nn.DataParallel(model_G, range(torch.cuda.device_count()))

    ### U-Net Discriminator ###
    model_D = D_arch.UnetD().cuda()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Optimizers
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(params_G, lr=LR_G)

    params_D = list(filter(lambda p: p.requires_grad, model_D.parameters()))
    opt_D = optim.Adam(params_D, lr=LR_D)


    ## Load saved params
    if START_ITER > 0:
        lm = torch.load('{}/checkpoint/v{}/model_G_i{:06d}.pth'.format(SAVE_PREFIX + EXP_NAME, str(VERSION), START_ITER))
        model_G.load_state_dict(lm.state_dict(), strict=True)


    data_set = ACCESS_AWAP_GAN(START_TIME, END_TIME)
    len_dataset = len(data_set)
    write_log("Total Dataset: " + str(len_dataset))

    # Train/Val/Test split 80/19.8/0.2
    train_data, val_data = random_split(data_set, [int(len(data_set) * 0.8), len(data_set) - int(len(data_set) * 0.8)])
    val_data, test_data = random_split(val_data, [int(len(val_data) * 0.99), len(val_data) - int(len(val_data) * 0.99)])

    train_dataloders = DataLoader(train_data, batch_size=NB_BATCH, shuffle=True, num_workers=NB_THREADS)
    val_dataloders = DataLoader(val_data, batch_size=NB_BATCH, shuffle=True, num_workers=NB_THREADS)
    test_dataloders = DataLoader(test_data, batch_size=NB_BATCH, shuffle=True, num_workers=NB_THREADS)

    # Train the GAN model
    train_GAN(model_G, model_D, opt_G, opt_D, train_dataloders, val_dataloders, test_dataloders, NB_Iteration)




    
    
