import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time
import math
from os import mkdir
from os.path import isdir
from torch.utils.data import random_split, DataLoader
from datetime import date
from datetime import datetime
from utils import Huber, ACCESS_AWAP_GAN, RMSE, MAE, log_loss2, generate_sample, crps_loss
import cv2
import pretrain_m_arch as G_arch
import RRDBNet_arch2 as pre_arch
import scipy.special


### USER PARAMS ###
START_TIME = date(1990, 1, 1)
END_TIME = date(2001, 12, 31)

SAVE_PREFIX = "/scratch/iu60/xs5813/EXTREME_MODEL_VERSION/"
EXP_NAME = "version_1"
VERSION = "TestRefactored"
UPSCALE = 8  # upscaling factor 40km->5km
NB_BATCH = 3  # mini-batch
NB_Iteration = 12 # Number of iterations (epochs)
PATCH_SIZE = 128
NB_THREADS = 36

START_ITER = 0  # Set 0 for from scratch, else will load saved params and trains further

L_ADV = 1e-3  # Scaling params for the Adv loss
L_FM = 1  # Scaling params for the feature matching loss
L_LPIPS = 1e-3  # Scaling params for the LPIPS loss
#add gamma loss and CRPS loss
L_LOG = 1e-4  # 权重参数 for the log loss
L_CRPS = 1e-3   # 权重参数 for the CRPS loss

LR_G = 1e-5  # Learning rate for the generator
OSAVE_PREFIX = "/scratch/iu60/xs5813/DESRGAN_ORIGINAL/"
OMODEL_PREFIX = OSAVE_PREFIX + "DESRGAN/checkpoint/" + "vTestRefactored" 
crps = []

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
    my_log_file = open("./save/" + VERSION + '/train_loss_huber_1e-3lr1e-5_2.txt', 'a')
    my_log_file.write(log + '\n')
    my_log_file.close()

def SaveCheckpoint(i, model_G=None, opt_G=None, best=False):
    """
    Save the model and optimizer params.

    Args:
        i: current iteration
        model_G: generator model
        opt_G: generator optimizer
        best: whether to save the best model
    """
    str_best = ''
    if best:
        str_best = '_best'

    # 获取当前的日期和时间
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    i = i + START_ITER + 1

    torch.save(model_G, '{}/checkpoint/v{}/model_G_i{:06d}{}_{}.pth'.format(SAVE_PREFIX + EXP_NAME, str(VERSION), i, str_best, current_time))

    torch.save(opt_G, '{}/checkpoint/v{}/opt_G_i{:06d}{}_{}.pth'.format(SAVE_PREFIX + EXP_NAME, str(VERSION), i, str_best, current_time))
    write_log("Checkpoint saved with timestamp (pretrain)" + current_time)

def get_patches(batch, lr, hr, patch_size, scaling_factor):

    np.random.seed(batch)
    #Low Resolution Image Size:  torch.Size([3, 1, 22, 18])
    #High Resolution Image Size:  torch.Size([3, 1, 161, 215])
    # 确保随机坐标加上补丁尺寸不会超出边界
    max_hh = min(lr.shape[2] * scaling_factor, hr.shape[2]) - patch_size
    max_hw = min(lr.shape[3] * scaling_factor, hr.shape[3]) - patch_size

    hh = np.random.randint(0, max_hh + 1)
    hw = np.random.randint(0, max_hw + 1)

    hr = hr[:, :, hh:(hh + patch_size), hw:(hw + patch_size)]
    lr = lr[:, :, int(hh / scaling_factor):int((hh + patch_size) / scaling_factor), int(hw / scaling_factor):int((hw + patch_size) / scaling_factor)]

    return lr, hr
''''''


def generator_loss(model_G, batch_L, batch_H):
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
    output = model_G(batch_L)
    # 生成样本估计值
    batch_S = generate_sample(output)
    # Pixel loss
    loss_Pixel = Huber(batch_S, batch_H)
    loss_G = loss_Pixel
    #log loss, ground truth and gamma distribution
    loss_Log = log_loss2(batch_H, output)
    #running test
    loss_CRPS = crps_loss(batch_H, output)
    print("loss_CPRS:",loss_CRPS)
    #crps = crps_batch(batch_H, rain_prob, gamma_shape, gamma_scale)
    # GAN losses
    loss_G += 1e-3*loss_Log #+ L_CRPS * crps
    print(f"Huber Loss: {loss_Pixel.item()}")
    print(f"Log Loss: {loss_Log .item()}")

    return loss_G

def generator_iteration(model_G, opt_G, lr, hr) -> None:
    """
        Train the generator on the batch of lr and hr images.

        Args:
            model_G: generator model
            model_D: discriminator model
            opt_G: generator optimizer
            lr: batch of lr images
            hr: batch of hr images
    """
    
    # Get data as cuda variables
    batch_L = Variable(lr).cuda()
    batch_H = Variable(hr).cuda()

    opt_G.zero_grad()

    # Train generator
    loss_G = generator_loss(model_G, batch_L, batch_H)

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
            #print(f"LR shape: {lr_val.shape}, HR shape: {hr_val.shape}") 
            batch_H = np.asarray(hr_val).astype(np.float32)
            batch_L = np.asarray(lr_val).astype(np.float32)

            # Convert to torch.Tensor
            batch_L = Variable(torch.from_numpy(batch_L)).cuda()
            batch_L = torch.clamp(batch_L, 0., 1.)

            # Forward pass through generator
            output = model_G(batch_L)

            # Generate sample estimate from the distribution
            batch_Out = generate_sample(output)
            #print("batch_Out shape", batch_Out.shape)
            batch_Out = np.clip(batch_Out.cpu().numpy(), 0., 1.)
            # Process output and ground truth for metric calculation
            batch_Out = np.squeeze(batch_Out, axis = 1)
            #batch_Out = np.transpose(batch_Out, [1, 2, 0])
            batch_Out = batch_Out.transpose(1, 2, 0) # 688*880*16
            batch_Out = cv2.resize(batch_Out, (207,172), interpolation=cv2.INTER_CUBIC)
            #(3,164,237) (3,172,237) 
            if len(batch_Out.shape) == 2:
                batch_Out = batch_Out.reshape(batch_Out.shape[0], batch_Out.shape[1], 1)
            #batch_Out = np.transpose(batch_Out, [2, 0, 1])
            batch_Out = batch_Out.transpose(2, 0, 1)
            img_gt = np.squeeze(batch_H)

            img_gt = np.expm1(img_gt * 7)  # Revert preprocessing on ground truth
            img_target = np.expm1(batch_Out * 7)  # Revert preprocessing on prediction
            print("img_gt max value", np.max(img_gt), "img_target",np.max(img_target))
            zero_count_gt = np.size(img_gt) - np.count_nonzero(img_gt)
            zero_count_target = np.size(img_target) - np.count_nonzero(img_target)

            print("Number of zero values in img_gt:", zero_count_gt,"Number of zero values in img_target:", zero_count_target)

            rmses.append(RMSE(img_gt, img_target, 0))
            maes.append(MAE(img_gt, img_target, 0))
            
        rmse = np.mean(np.asarray(rmses))
        mae = np.mean(np.asarray(maes))
        
        write_log(f"RMSE: {rmse} MAE: {mae} for epoch {epoch}{f' batch {batch}' if batch != -1 else ''}. Completed in {time.time() - itr_test_time} seconds")

        return rmse, mae


def train_GAN(model_G, opt_G, train_dataloders, val_dataloders, test_dataloders, nb_epoch):
    """
    Train the model using only the generator.
    """
    best_avg_mae = 1000000
    for epoch in range(nb_epoch):
        print("epoch right now:", epoch)
        epoch_start_time = time.time()
        for batch_number, (lr, hr, _, _) in enumerate(train_dataloders):
            batch_start_time = time.time()
            model_G.train()
            lr, hr = get_patches(batch_number, lr, hr, PATCH_SIZE, UPSCALE)
            batch_L = Variable(lr).cuda()
            batch_H = Variable(hr).cuda()
            generator_iteration(model_G, opt_G, lr, hr)

            # Forward pass
            output = model_G(batch_L)
            batch_S = generate_sample(output)
            batch_end_time = time.time()
            write_log(f"Batch {batch_number} completed in {batch_end_time - batch_start_time} seconds")

            if batch_number % 10 == 0:
                get_performance(model_G, test_dataloders, epoch, batch_number)
                break

        # Validation performance
        epoch_end_time = time.time()
        write_log(f"Epoch {epoch} completed in {epoch_end_time - epoch_start_time} seconds")

        ## Save models per iteration
        SaveCheckpoint(epoch, model_G, opt_G)

        # Validate mode
        val_rmse, val_mae = get_performance(model_G, val_dataloders, epoch)


        # Save best model
        if val_mae < best_avg_mae:
            best_avg_mae = val_mae
            SaveCheckpoint(epoch, model_G,  opt_G, best=True)
        
        write_log(f"Best MAE: {best_avg_mae}")

if __name__ == "__main__":
    # Prepare directories
    prepare_directories()
    model_G_name = "model_G_i000004_best_20240219-114819"

    # Load model
    model_path = OMODEL_PREFIX +  "/" + model_G_name + ".pth"
    # Check if the pretrained model is a DataParallel model
    # 如果您的预训练模型结构与RRDBNetx4x2相同，则可以直接创建一个实例并加载权重
    #pretrained_model = pre_arch.RRDBNetx4x2(1, 1, 64, 23, gc=32).cuda()

    #pretrained_model.load_state_dict(torch.load(model_path).module.state_dict(), strict=True)  
    ### Generator ###
    model_G = G_arch.ModifiedRRDBNet(model_path, 1, 3, 64, 23, gc=32).cuda()
    if torch.cuda.device_count() > 1:
        write_log("Using " + str(torch.cuda.device_count()) + f" GPUs!{model_G_name}")
        model_G = nn.DataParallel(model_G, range(torch.cuda.device_count()))

    print("pretrain")

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Optimizers
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(params_G, lr=LR_G)


    ## Load saved params
    if START_ITER > 0:
        lm = torch.load('{}/checkpoint/v{}/model_pretrain_G_i{:06d}.pth'.format(SAVE_PREFIX + EXP_NAME, str(VERSION), START_ITER))
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
    train_GAN(model_G, opt_G, train_dataloders, val_dataloders, test_dataloders, NB_Iteration)