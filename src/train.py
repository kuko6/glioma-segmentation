import argparse
# from azureml.core import Run

import os
import glob
from tkinter.messagebox import NO

import numpy as np
import segmentation_models_3D as sm
import tensorflow as tf
import pandas as pd
import wandb
from matplotlib import pyplot as plt
from scipy import ndimage
from wandb.keras import WandbCallback

import utils.utils as utils
from utils.callbacks import *
from utils.data_processing import *
import losses
from metrics import *
from segmentation_models_3D import losses as ls
from model import unet
from generator import BratsGen

# -------------------------------------------------------------------------------- #
# Script used for model training.
# Its important to set the config dict, because it affects the training process.
# Supports:
#   - standard and ensemble training
#   - multiple sequences
#   - categorical and binary crossentropy and dice loss functions
#   - wandb integration
#   - test prediction after each epoch (prediction callback)
#   - saves the training history as .csv and final model as .h5 
# -------------------------------------------------------------------------------- #

config = {
    "num_classes": 4, # 1, 2, 4
    "img_channels": 3, # 2, 3, 4
    "learning_rate": 1e-4, #1e-3, 1e-4, 1e-5, 1e-6
    "epochs": 50,
    "batch_size": 2, # 2, 4
    "loss": "categorical_crossentropy", # categorical_crossentropy, dice_loss, binary_crossentropy, binary_dice_loss
    "optimizer": "adam",
    "dataset": "BraTS2021"
}

def list_files(path):
    brains = os.listdir(path)
    files = []
    for brain in brains:
        brain_path = os.path.join(path, brain)
        files += [os.path.join(brain_path, file) for file in os.listdir(brain_path)]
    # print(len(files))

    return files

# tests the training generator
def test_generator(data_gen, channels):
    img, mask = data_gen.__getitem__(0)
    #mask = mask[0]
    print('img shape: ', img.shape)
    print('mask shape: ', mask.shape)
    mask = np.argmax(mask, axis=-1)
    custom_cmap = utils.get_custom_cmap()
    if channels == 4:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
        ax1.imshow(ndimage.rotate(img[0][:, :, 80, 0], 270), cmap='gray')
        ax1.set_title('Image flair')
        ax2.imshow(ndimage.rotate(img[0][:, :, 80, 1], 270), cmap='gray')
        ax2.set_title('Image t1ce')
        ax3.imshow(ndimage.rotate(img[0][:, :, 80, 2], 270), cmap='gray')
        ax3.set_title('Image t2')
        ax4.imshow(ndimage.rotate(img[0][:, :, 80, 3], 270), cmap='gray')
        ax4.set_title('Image t1')
        ax5.imshow(ndimage.rotate(mask[0][:, :, 80], 270), cmap=custom_cmap)
        ax5.set_title('Mask')
        fig.savefig(f'outputs/test.png')
    elif channels == 3:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
        ax1.imshow(ndimage.rotate(img[0][:, :, 80, 0], 270), cmap='gray')
        ax1.set_title('Image flair')
        ax2.imshow(ndimage.rotate(img[0][:, :, 80, 1], 270), cmap='gray')
        ax2.set_title('Image t1ce')
        ax3.imshow(ndimage.rotate(img[0][:, :, 80, 2], 270), cmap='gray')
        ax3.set_title('Image t2')
        ax4.imshow(ndimage.rotate(mask[0][:, :, 80], 270), cmap=custom_cmap)
        ax4.set_title('Mask')
        fig.savefig(f'outputs/test.png')
    elif channels == 2: # 2 channels
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
        ax1.imshow(ndimage.rotate(img[0][:, :, 80, 0], 270), cmap='gray')
        ax1.set_title('Image flair')
        ax2.imshow(ndimage.rotate(img[0][:, :, 80, 1], 270), cmap='gray')
        ax2.set_title('Image t1ce')
        ax3.imshow(ndimage.rotate(mask[0][:, :, 80], 270), cmap=custom_cmap)
        ax3.set_title('Mask')
        fig.savefig(f'outputs/test.png')
    plt.close(fig)


def main():
    # print(help('modules'))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--wandb', type=str, help='wandb id')
    args = parser.parse_args()

    wandb_key = args.wandb
    use_wandb = False
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.config = config
        use_wandb = True
    
    if args.data_path:
        data = args.data_path
    else:
        data = os.path.join(os.getcwd(), '../BraTS2021')
    print(os.listdir(data))

    training_path = os.path.join(data, 'train/')
    validation_path = os.path.join(data, 'val/')
    #testing_path = os.path.join(data, 'test/')

    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    train_t2_list = sorted(glob.glob(training_path + '/*/*t2.nii.gz'))
    train_t1_list = sorted(glob.glob(training_path + '/*/*t1.nii.gz'))
    train_t1ce_list = sorted(glob.glob(training_path + '/*/*t1ce.nii.gz'))
    train_flair_list = sorted(glob.glob(training_path + '/*/*flair.nii.gz'))
    train_mask_list = sorted(glob.glob(training_path + '/*/*seg.nii.gz'))

    val_t2_list = sorted(glob.glob(validation_path + '/*/*t2.nii.gz'))
    val_t1_list = sorted(glob.glob(validation_path + '/*/*t1.nii.gz'))
    val_t1ce_list = sorted(glob.glob(validation_path + '/*/*t1ce.nii.gz'))
    val_flair_list = sorted(glob.glob(validation_path + '/*/*flair.nii.gz'))
    val_mask_list = sorted(glob.glob(validation_path + '/*/*seg.nii.gz'))

    batch_size = config["batch_size"]
    if config['num_classes'] == 4:
        subregions = [0]
    else:
        subregions = [1, 2, 3]

    # loop relevant only for separate class training, otherwise will only do one iteration
    for subregion in subregions:
        if use_wandb:
            run = wandb.init(
            project="BraTS2021",
            name=f"{config['loss']}_{config['epochs']}_{config['img_channels']}ch_sub{subregion}",
            entity="kuko",
            reinit=True,
            config=config
        )

        # setting the training and valitation generators
        train_img_datagen = BratsGen(
            flair_list=train_flair_list, 
            t1ce_list=train_t1ce_list, 
            t2_list=train_t2_list, 
            t1_list=train_t1_list, 
            mask_list=train_mask_list,
            img_dim=(128, 128, 128), 
            img_channels=config['img_channels'],
            classes=config['num_classes'],
            batch_size=config['batch_size'],
            segmenting_subregion=subregion, aug=True
        )

        val_img_datagen = BratsGen(
            flair_list=val_flair_list, 
            t1ce_list=val_t1ce_list, 
            t2_list=val_t2_list, 
            t1_list=val_t1_list, 
            mask_list=val_mask_list,
            img_dim=(128, 128, 128), 
            img_channels=config['img_channels'],
            classes=config['num_classes'],
            batch_size=config['batch_size'],
            segmenting_subregion=subregion, aug=True
        )

        test_generator(train_img_datagen, channels=config['img_channels'])

        # defining metrics used during training
        if config['num_classes'] == 4:
            metrics = [
                sm.metrics.IOUScore(threshold=0.5),
                tf.keras.metrics.MeanIoU(num_classes=4),
                dice_coef_multilabel(classes=config['num_classes']), dice_coef2, dice_coef_necrotic,
                dice_coef_edema, dice_coef_enhancing
            ]
        elif config['num_classes'] == 2:
            metrics = [
                sm.metrics.IOUScore(threshold=0.5),
                tf.keras.metrics.MeanIoU(num_classes=2),
                dice_coef_multilabel(classes=config['num_classes']), dice_coef2
            ]
        else:
            metrics = [
                sm.metrics.IOUScore(threshold=0.5),
                tf.keras.metrics.MeanIoU(num_classes=2),
                dice_coef_binary, dice_coef2
            ]

        LR = config["learning_rate"]
        optim = tf.keras.optimizers.Adam(LR)
        steps_per_epoch = len(train_flair_list) // batch_size
        val_steps_per_epoch = len(val_flair_list) // batch_size
        #steps_per_epoch = 10
        #val_steps_per_epoch = 2

        # defining the U-Net model
        model = unet(
            img_height=128, img_width=128, img_depth=128, 
            img_channels=config["img_channels"], 
            num_classes=config["num_classes"]
        )

        # setting the loss function
        if config['loss'] == "dice_loss":
            model.compile(optimizer=optim, loss=losses.dice_loss, metrics=metrics)
            print('using dice_loss')
        elif config['num_classes'] == 4:
            if config['loss'] == "categorical_crossentropy":
                model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=metrics)
                print('using categorical_crossentropy')
        elif config['num_classes'] == 2:
            model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=metrics)
            print('using categorical_crossentropy')
        else:
            model.compile(optimizer=optim, loss="binary_crossentropy", metrics=metrics)
            print('using binary_crossentropy')

        # defining the callbacks
        # the sequences are needed because of the prediction callback
        callbacks = get_callbacks(
            flair=[training_path + 'BraTS2021_00002/BraTS2021_00002_flair.nii.gz'],
            t1ce=[training_path + 'BraTS2021_00002/BraTS2021_00002_t1ce.nii.gz'],
            t2=[training_path + 'BraTS2021_00002/BraTS2021_00002_t2.nii.gz'],
            t1=[training_path + 'BraTS2021_00002/BraTS2021_00002_t1.nii.gz'],
            mask=[training_path + 'BraTS2021_00002/BraTS2021_00002_seg.nii.gz'],
            channels=config['img_channels'],
            subregion=subregion,
            n_slice=80,
            classes=config['num_classes'], 
            use_wandb=use_wandb
        )

        if use_wandb:
            callbacks.append(WandbCallback())

        # training the model
        history = model.fit(
            train_img_datagen,
            steps_per_epoch=steps_per_epoch,
            epochs=config["epochs"],
            verbose=1,
            callbacks=callbacks,
            validation_data=val_img_datagen,
            validation_steps=val_steps_per_epoch
        )

        model.save(f'outputs/model_{subregion}.h5')
        
        # Saves history as .csv
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = f'outputs/history_{subregion}.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        if use_wandb:
            run.finish()


if __name__ == '__main__':
    main()
