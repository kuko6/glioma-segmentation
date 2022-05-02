import argparse
# from azureml.core import Run

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage
import glob
from scipy import ndimage
import segmentation_models_3D as sm
import cv2
import tensorflow as tf
import re

import wandb

import utils
import losses

batch_size = 2
subregion = 0
classes = 4
channels = 2
n_slice = 80
model_name = 'models/model_0.h5'

# util function for generating interactive image mask from components
def wandb_mask(img, true_mask, pred_mask):
    labels = {0: 'background', 1: 'necrotic', 2: 'edema', 3: 'enhancing'}

    return wandb.Image(img, masks={
        "ground truth": {
            "mask_data": true_mask,
            "class_labels": labels
        },
        "prediction": {
            "mask_data": pred_mask,
            "class_labels": labels
        }
    })


def predict_image(my_model, flair, t1ce, t2, mask, subdir='', counter=10000):
    if not os.path.isdir(f'outputs/{subdir}'):
        os.mkdir(f'outputs/{subdir}')

    img_name = re.search(r"\bBraTS2021_\d+", flair)
    img_name = img_name.group()
    os.mkdir(f'outputs/{subdir}{img_name}/')
    subdir = subdir + f'{img_name}/'

    test_img = utils.load_img([flair], [t1ce], [t2], img_channels=channels)

    test_prediction = my_model.predict(test_img)
    #print(np.unique(test_prediction))
    test_prediction_argmax = np.argmax(test_prediction, axis=-1)
    #print(np.unique(test_prediction_argmax))
    #print('original shape: ', test_prediction.shape)
    #print('new shape: ', test_prediction_argmax.shape)

    test_mask = utils.load_mask([mask], segmenting_subregion=0, classes=classes)
    test_mask_argmax = np.argmax(test_mask, axis=-1)
    #print('mask shape: ', test_mask.shape)
    # print(test_mask.dtype)
    # print(test_prediction.dtype)
    # test_mask = tf.cast(test_mask, tf.float32)
    # print(test_mask.dtype)

    print('dice:', losses.dice_coef(test_mask, test_prediction).numpy())
    print('dice edema:', losses.dice_coef_edema(test_mask, test_prediction).numpy())
    print('dice necrotic:', losses.dice_coef_necrotic(test_mask, test_prediction).numpy())
    print('dice enhancing:', losses.dice_coef_enhancing(test_mask, test_prediction).numpy())

    volume_start = 20
    volume_end = 125
    step = 5
    for i in range(volume_start, volume_end+step, step):
        if channels == 3:
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
            ax1.imshow(ndimage.rotate(test_img[0][:, :, i, 0], 270), cmap='gray')
            ax1.set_title('Image flair')
            ax2.imshow(ndimage.rotate(test_img[0][:, :, i, 1], 270), cmap='gray')
            ax2.set_title('Image t1ce')
            ax3.imshow(ndimage.rotate(test_img[0][:, :, i, 2], 270), cmap='gray')
            ax3.set_title('Image t2')
            ax4.imshow(ndimage.rotate(test_mask_argmax[0][:, :, i], 270))
            ax4.set_title('Mask')
            ax5.imshow(ndimage.rotate(test_prediction_argmax[0][:, :, i], 270))
            ax5.set_title('Prediction')
            #fig.savefig(f'outputs/test.png')
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
            ax1.imshow(ndimage.rotate(test_img[0][:, :, i, 0], 270), cmap='gray')
            ax1.set_title('Image flair')
            ax2.imshow(ndimage.rotate(test_img[0][:, :, i, 1], 270), cmap='gray')
            ax2.set_title('Image t1ce')
            ax3.imshow(ndimage.rotate(test_mask_argmax[0][:, :, i], 270))
            ax3.set_title('Mask')
            ax4.imshow(ndimage.rotate(test_prediction_argmax[0][:, :, i], 270))
            ax4.set_title('Prediction')
            #fig.savefig(f'outputs/test.png')

        fig.savefig(f'outputs/{subdir + img_name}_{i}.png')
        plt.close()

        flair = ndimage.rotate(test_img[0][:, :, i, 0], 270)
        t1ce = ndimage.rotate(test_img[0][:, :, i, 1], 270)
        true_mask = ndimage.rotate(test_mask_argmax[0][:, :, i], 270)
        pred_mask = ndimage.rotate(test_prediction_argmax[0][:, :, i], 270)

        mask = wandb_mask(flair, true_mask, pred_mask)
        wandb.log({f"{subdir}": mask}, step=counter+i)

    '''
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
    ax1.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 0], 270), cmap='gray')
    ax1.set_title('Testing Flair')
    ax2.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 1], 270), cmap='gray')
    ax2.set_title('Testing T1ce')
    # ax3.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 2], 270), cmap='gray')
    # ax3.set_title('Testing T2')
    ax3.imshow(ndimage.rotate(test_mask_argmax[0][:, :, n_slice], 270))
    ax3.set_title('Mask')
    # ax5.imshow(ndimage.rotate(test_prediction[0][:,:, n_slice, 1], 270))
    ax4.imshow(ndimage.rotate(test_prediction_argmax[0][:, :, n_slice], 270))
    ax4.set_title('Prediction')
    '''


def model_eval(my_model, flair_list, t1ce_list, t2_list, mask_list):
    dice_list = list()
    necrotic_list = list()
    edema_list = list()
    enhancing_list = list()

    i = 0
    for flair_name, t1ce_name, t2_name, mask_name in zip(flair_list, t1ce_list, t2_list, mask_list):
        # test_img = np.load(img_list[i])
        test_img = utils.load_img([flair_name], [t1ce_name], [t2_name], img_channels=channels)
        test_mask = utils.load_mask([mask_name], segmenting_subregion=0, classes=classes)
        # test_mask = np.argmax(test_mask, axis=-1)

        # test_img_input = np.expand_dims(test_img, axis=0)
        test_prediction = my_model.predict(test_img)
        # test_prediction = np.argmax(test_prediction, axis=-1)

        dice_list.append(losses.dice_coef(test_mask, test_prediction).numpy())
        necrotic_list.append(losses.dice_coef_necrotic(test_mask, test_prediction).numpy())
        edema_list.append(losses.dice_coef_edema(test_mask, test_prediction).numpy())
        enhancing_list.append(losses.dice_coef_enhancing(test_mask, test_prediction).numpy())

        img_name = re.search(r"\bBraTS2021_\d+", flair_list[i])
        print(f"image: {img_name.group()}")
        print(f"dice_coef: {dice_list[i]} | necrotic: {necrotic_list[i]} | edema: {edema_list[i]} | enhancing: {enhancing_list[i]}")
        print()
        i += 1

    print(f"\ndice_mean: {np.mean(dice_list)} | necrotic_mean: {np.mean(necrotic_list)} | edema_mean: {np.mean(edema_list)} | enhancing_mean: {np.mean(enhancing_list)}\n")

    counter = 10000
    worst = np.argsort(dice_list)[:5]
    print("\nThe worst 5:")
    for i in worst:
        img_name = re.search(r"\bBraTS2021_\d+", flair_list[i])
        print(f"image: {img_name.group()}, dice = {dice_list[i]}")
        predict_image(my_model, flair=flair_list[i], t1ce=t1ce_list[i], t2=t2_list[i], mask=mask_list[i],
                      subdir='worst/',counter=counter)
        print()
        counter = counter + 10000

    best = np.argsort(dice_list)[-5:]
    print("\nThe best 5:")
    for i in best:
        img_name = re.search(r"\bBraTS2021_\d+", flair_list[i])
        print(f"image: {img_name.group()}, dice = {dice_list[i]}")
        predict_image(my_model, flair=flair_list[i], t1ce=t1ce_list[i], t2=t2_list[i], mask=mask_list[i],
                      subdir='best/', counter=counter)
        print()
        counter = counter + 10000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--wandb', type=str, help='wandb id')
    args = parser.parse_args()

    wandb_key = args.wandb
    wandb.login(key=wandb_key)

    run = wandb.init(project="BraTS2021",
                     name=f"evaluation_{model_name}",
                     entity="kuko",
                     reinit=True)

    data = args.data_path
    print(os.listdir(data))
    training_path = os.path.join(data, 'train/')
    testing_path = os.path.join(data, 'test/')
    validation_path = os.path.join(data, 'val/')

    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    train_t2_list = glob.glob(training_path + '/*/*t2.nii.gz')
    train_t1ce_list = glob.glob(training_path + '/*/*t1ce.nii.gz')
    train_flair_list = glob.glob(training_path + '/*/*flair.nii.gz')
    train_mask_list = glob.glob(training_path + '/*/*seg.nii.gz')

    val_t2_list = glob.glob(validation_path + '/*/*t2.nii.gz')
    val_t1ce_list = glob.glob(validation_path + '/*/*t1ce.nii.gz')
    val_flair_list = glob.glob(validation_path + '/*/*flair.nii.gz')
    val_mask_list = glob.glob(validation_path + '/*/*seg.nii.gz')

    test_t2_list = glob.glob(testing_path + '/*/*t2.nii.gz')
    test_t1ce_list = glob.glob(testing_path + '/*/*t1ce.nii.gz')
    test_flair_list = glob.glob(testing_path + '/*/*flair.nii.gz')
    test_mask_list = glob.glob(testing_path + '/*/*seg.nii.gz')

    '''
    test_mask=nib.load(training_path + 'BraTS2021_00002/BraTS2021_00002_seg.nii.gz').get_fdata()
    fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
    ax1.imshow(ndimage.rotate(montage(test_mask[:,:,20:100]), 270), cmap ='gray')
    fig.savefig('outputs/test.png')
    return
    '''

    test_img_datagen = utils.image_loader(test_flair_list, test_t1ce_list, test_t2_list, test_mask_list,
                                          batch_size=batch_size, channels=channels,
                                          segmenting_subregion=subregion)

    if classes == 4:
        custom_objects = {
            'iou_score': sm.metrics.IOUScore(threshold=0.5),
            'dice_coef': losses.dice_coef,
            'dice_coef2': losses.dice_coef2,
            'dice_coef_edema': losses.dice_coef_edema,
            'dice_coef_necrotic': losses.dice_coef_necrotic,
            'dice_coef_enhancing': losses.dice_coef_enhancing
        }
    elif classes == 1:
        custom_objects = {
            'iou_score': sm.metrics.IOUScore(threshold=0.5),
            'dice_coef': losses.dice_coef,
            'dice_coef2': losses.dice_coef2
        }

    my_model = tf.keras.models.load_model(model_name,
                                          custom_objects=custom_objects,
                                          compile=False)

    model_eval(my_model, test_flair_list, test_t1ce_list, test_t2_list, test_mask_list)

    run.finish()

    # test_img, test_mask = test_img_datagen.__next__()
    test_img = utils.load_img([training_path + 'BraTS2021_00002/BraTS2021_00002_flair.nii.gz'],
                              [training_path + 'BraTS2021_00002/BraTS2021_00002_t1ce.nii.gz'],
                              [training_path + 'BraTS2021_00002/BraTS2021_00002_t2.nii.gz'],
                              img_channels=channels)

    test_prediction = my_model.predict(test_img)

    if classes == 4:
        print(np.unique(test_prediction))
        test_prediction_argmax = np.argmax(test_prediction, axis=-1)
        print(np.unique(test_prediction_argmax))
        print('original shape: ', test_prediction.shape)
        print('new shape: ', test_prediction_argmax.shape)

        test_mask = utils.load_mask([training_path + 'BraTS2021_00002/BraTS2021_00002_seg.nii.gz'],
                                    segmenting_subregion=0, classes=classes)
        test_mask_argmax = np.argmax(test_mask, axis=-1)
        print('mask shape: ', test_mask.shape)
        print(test_mask.dtype)
        print(test_prediction.dtype)
        # test_mask = tf.cast(test_mask, tf.float32)
        print(test_mask.dtype)

        print('dice:', losses.dice_coef(test_mask, test_prediction).numpy())
        print('dice edema:', losses.dice_coef_edema(test_mask, test_prediction).numpy())
        print('dice necrotic:', losses.dice_coef_necrotic(test_mask, test_prediction).numpy())
        print('dice enhancing:', losses.dice_coef_enhancing(test_mask, test_prediction).numpy())

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
        ax1.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 0], 270), cmap='gray')
        ax1.set_title('Testing Flair')
        ax2.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 1], 270), cmap='gray')
        ax2.set_title('Testing T1ce')
        # ax3.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 2], 270), cmap='gray')
        # ax3.set_title('Testing T2')
        ax3.imshow(ndimage.rotate(test_mask_argmax[0][:, :, n_slice], 270))
        ax3.set_title('Mask')
        # ax5.imshow(ndimage.rotate(test_prediction[0][:,:, n_slice, 1], 270))
        ax4.imshow(ndimage.rotate(test_prediction_argmax[0][:, :, n_slice], 270))
        ax4.set_title('Prediction')
        fig.savefig('outputs/test.png')

    elif classes == 1:
        print(np.unique(test_prediction))
        test_prediction_argmax = np.argmax(test_prediction[0], axis=-1)
        print(np.unique(test_prediction_argmax))
        print('original shape: ', test_prediction.shape)
        # print('new shape: ', test_prediction_argmax.shape)

        test_mask = utils.load_mask([training_path + 'BraTS2021_00002/BraTS2021_00002_seg.nii.gz'],
                                    segmenting_subregion=2, classes=classes)
        test_mask_argmax = np.argmax(test_mask, axis=-1)
        print('mask shape: ', test_mask.shape)

        print(test_mask.dtype)
        print(test_prediction.dtype)
        test_mask = tf.cast(test_mask, tf.float32)
        print(test_mask.dtype)

        print('dice:', losses.dice_coef(test_mask, test_prediction).numpy())

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(20, 10))
        ax1.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 0], 270), cmap='gray')
        ax1.set_title('Testing Flair')
        ax2.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 1], 270), cmap='gray')
        ax2.set_title('Testing T1ce')
        ax3.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 2], 270), cmap='gray')
        ax3.set_title('Testing T2')
        ax4.imshow(ndimage.rotate(test_mask[0][:, :, n_slice], 270))
        ax4.set_title('Mask')
        # ax5.imshow(ndimage.rotate(test_prediction[0][:,:, n_slice, 1], 270))
        ax5.imshow(ndimage.rotate(test_prediction[0][:, :, n_slice], 270))
        ax5.set_title('Prediction')
        ax6.imshow(ndimage.rotate(test_prediction_argmax[:, :, n_slice], 270))
        ax6.set_title('Prediction2')
        fig.savefig('outputs/test.png')

    '''
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1, 7, figsize=(20, 10))
    ax1.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 0], 270), cmap='gray')
    ax1.set_title('Testing Image')
    ax2.imshow(ndimage.rotate(test_mask_argmax[:, :, n_slice], 270))
    ax2.set_title('Mask')
    ax3.imshow(ndimage.rotate(test_prediction[0, :, :, n_slice, :], 270))
    ax3.set_title('Prediction')
    ax4.imshow(ndimage.rotate(test_prediction[0, :, :, n_slice, 0], 270))
    ax4.set_title('Prediction0')
    ax5.imshow(ndimage.rotate(test_prediction[0, :, :, n_slice, 1], 270))
    ax5.set_title('Prediction1')
    ax6.imshow(ndimage.rotate(test_prediction[0, :, :, n_slice, 2], 270))
    ax6.set_title('Prediction2')
    ax7.imshow(ndimage.rotate(test_prediction[0, :, :, n_slice, 3], 270))
    ax7.set_title('Prediction3')
    fig.savefig('outputs/test.png')
    '''


if __name__ == '__main__':
    main()
