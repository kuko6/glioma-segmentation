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
from scipy.spatial.distance import directed_hausdorff
import cv2
import tensorflow as tf
import re

import wandb

import utils.utils as utils
from utils.data_processing import *
import losses
from metrics import *

batch_size = 1
subregion = 0
classes = 4
channels = 3
n_slice = 80
model_name = 'models/model_3ch_aug_e5.h5'

def hausdorff_distance(y_true, y_pred, classes=[1, 2, 3]):
    haussdorf_dist = 0
    for i in classes:
        true_coords = np.argwhere(y_true[i])
        preds_coords = np.argwhere(y_pred[i])
        haussdorf_dist = directed_hausdorff(preds_coords, true_coords)[0]

    return haussdorf_dist

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


def predict_image(my_model, flair, t1ce, t2, t1, mask, subdir='', counter=10000):
    if not os.path.isdir(f'outputs/{subdir}'):
        os.mkdir(f'outputs/{subdir}')

    img_name = re.search(r"\bBraTS2021_\d+", flair)
    img_name = img_name.group()
    os.mkdir(f'outputs/{subdir}{img_name}/')
    subdir = subdir + f'{img_name}/'

    test_img = load_img([flair], [t1ce], [t2], [t1], img_channels=channels)

    test_prediction = my_model.predict(test_img)
    #print(np.unique(test_prediction))
    test_prediction_argmax = np.argmax(test_prediction, axis=-1)
    #print(np.unique(test_prediction_argmax))
    #print('original shape: ', test_prediction.shape)
    #print('new shape: ', test_prediction_argmax.shape)

    test_mask = load_mask([mask], segmenting_subregion=subregion, classes=classes)
    test_mask_argmax = np.argmax(test_mask, axis=-1)
    #print('mask shape: ', test_mask.shape)
    # print(test_mask.dtype)
    # print(test_prediction.dtype)
    # test_mask = tf.cast(test_mask, tf.float32)
    # print(test_mask.dtype)

    print('dice:', dice_coef_multilabel(classes=classes)(test_mask, test_prediction).numpy())
    if classes == 4:
        print('dice edema:', dice_coef_edema(test_mask, test_prediction).numpy())
        print('dice necrotic:', dice_coef_necrotic(test_mask, test_prediction).numpy())
        print('dice enhancing:', dice_coef_enhancing(test_mask, test_prediction).numpy())

    custom_cmap = utils.get_custom_cmap()

    volume_start = 20
    volume_end = 126 # 125
    step = 2 # 5
    for i in range(volume_start, volume_end+step, step):
        if channels == 3:
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
            ax1.imshow(ndimage.rotate(test_img[0][:, :, i, 0], 270), cmap='gray')
            ax1.set_title('Image flair')
            ax1.axis('off')
            ax2.imshow(ndimage.rotate(test_img[0][:, :, i, 1], 270), cmap='gray')
            ax2.set_title('Image t1ce')
            ax2.axis('off')
            ax3.imshow(ndimage.rotate(test_img[0][:, :, i, 2], 270), cmap='gray')
            ax3.set_title('Image t2')
            ax3.axis('off')
            ax4.imshow(ndimage.rotate(test_mask_argmax[0][:, :, i], 270), cmap=custom_cmap)
            ax4.set_title('Mask')
            ax4.axis('off')
            ax5.imshow(ndimage.rotate(test_prediction_argmax[0][:, :, i], 270), cmap=custom_cmap)
            ax5.set_title('Prediction')
            ax5.axis('off')
            #fig.savefig(f'outputs/test.png')
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
            ax1.imshow(ndimage.rotate(test_img[0][:, :, i, 0], 270), cmap='gray')
            ax1.set_title('Image flair')
            ax1.axis('off')
            ax2.imshow(ndimage.rotate(test_img[0][:, :, i, 1], 270), cmap='gray')
            ax2.set_title('Image t1ce')
            ax2.axis('off')
            ax3.imshow(ndimage.rotate(test_mask_argmax[0][:, :, i], 270), cmap=custom_cmap)
            ax3.set_title('Mask')
            ax3.axis('off')
            ax4.imshow(ndimage.rotate(test_prediction_argmax[0][:, :, i], 270), cmap=custom_cmap)
            ax4.set_title('Prediction')
            ax4.axis('off')
            #fig.savefig(f'outputs/test.png')

        fig.savefig(f'outputs/{subdir + img_name}_{i}.png')
        plt.close()

        # flair = ndimage.rotate(test_img[0][:, :, i, 0], 270)
        # t1ce = ndimage.rotate(test_img[0][:, :, i, 1], 270)
        # true_mask = ndimage.rotate(test_mask_argmax[0][:, :, i], 270)
        # pred_mask = ndimage.rotate(test_prediction_argmax[0][:, :, i], 270)

        # mask = wandb_mask(flair, true_mask, pred_mask)
        # wandb.log({f"{subdir}": mask}, step=counter+i)


def model_eval(my_model, flair_list, t1ce_list, t2_list, t1_list, mask_list):
    dice_list = list()
    necrotic_list = list()
    edema_list = list()
    enhancing_list = list()

    i = 0
    for flair_name, t1ce_name, t2_name, t1_name, mask_name in zip(flair_list, t1ce_list, t2_list, t1_list, mask_list):
        # test_img = np.load(img_list[i])
        test_img = load_img([flair_name], [t1ce_name], [t2_name], [t1_name], img_channels=channels)
        test_mask = load_mask([mask_name], segmenting_subregion=subregion, classes=classes)
        
        test_prediction = my_model.predict(test_img)
        # test_prediction = np.argmax(test_prediction, axis=-1)

        dice_list.append(dice_coef_multilabel(classes=classes)(test_mask, test_prediction).numpy())
        if classes == 4:
            necrotic_list.append(dice_coef_necrotic(test_mask, test_prediction).numpy())
            edema_list.append(dice_coef_edema(test_mask, test_prediction).numpy())
            enhancing_list.append(dice_coef_enhancing(test_mask, test_prediction).numpy())

        img_name = re.search(r"\bBraTS2021_\d+", flair_list[i])
        print(f"image: {img_name.group()} | {i+1}/{len(flair_list)}")
        if classes == 4:
            print(f"dice_coef: {dice_list[i]} | necrotic: {necrotic_list[i]} | edema: {edema_list[i]} | enhancing: {enhancing_list[i]}")
        else:
            print(f"dice_coef: {dice_list[i]}")
        print()
        i += 1

    if classes == 4:
        print(f"\ndice_mean: {np.mean(dice_list)} | necrotic_mean: {np.mean(necrotic_list)} | edema_mean: {np.mean(edema_list)} | enhancing_mean: {np.mean(enhancing_list)}\n")
    else:
        print(f"\ndice_mean: {np.mean(dice_list)}\n")

    counter = 10000
    worst = np.argsort(dice_list)[:5]
    print("\nThe worst 5:")
    for i in worst:
        img_name = re.search(r"\bBraTS2021_\d+", flair_list[i])
        print(f"image: {img_name.group()}, dice = {dice_list[i]}")
        predict_image(my_model, flair=flair_list[i], t1ce=t1ce_list[i], t2=t2_list[i], t1=t1_list[i], mask=mask_list[i],
                      subdir='worst/',counter=counter)
        print()
        counter = counter + 10000

    best = np.argsort(dice_list)[-5:]
    print("\nThe best 5:")
    for i in best:
        img_name = re.search(r"\bBraTS2021_\d+", flair_list[i])
        print(f"image: {img_name.group()}, dice = {dice_list[i]}")
        predict_image(my_model, flair=flair_list[i], t1ce=t1ce_list[i], t2=t2_list[i], t1=t1_list[i], mask=mask_list[i],
                      subdir='best/', counter=counter)
        print()
        counter = counter + 10000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--wandb', type=str, help='wandb id')
    args = parser.parse_args()

    wandb_key = args.wandb
    # wandb.login(key=wandb_key)

    # run = wandb.init(project="BraTS2021-evaluation",
    #                  name=f"evaluation_{model_name}",
    #                  entity="kuko",
    #                  reinit=True)

    data = args.data_path
    print(os.listdir(data))
    training_path = os.path.join(data, 'train/')
    validation_path = os.path.join(data, 'val/')
    testing_path = os.path.join(data, 'test/')

    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    test_flair_list = glob.glob(testing_path + '/*/*flair.nii.gz')
    test_t1ce_list = glob.glob(testing_path + '/*/*t1ce.nii.gz')
    test_t2_list = glob.glob(testing_path + '/*/*t2.nii.gz')
    test_t1_list = glob.glob(testing_path + '/*/*t1.nii.gz')
    test_mask_list = glob.glob(testing_path + '/*/*seg.nii.gz')

    if classes == 4:
        custom_objects = {
            'iou_score': sm.metrics.IOUScore(threshold=0.5),
            'dice_coef': dice_coef_multilabel,
            'dice_coef2': dice_coef2,
            'dice_coef_edema': dice_coef_edema,
            'dice_coef_necrotic': dice_coef_necrotic,
            'dice_coef_enhancing': dice_coef_enhancing
        }
    else:
        custom_objects = {
            'iou_score': sm.metrics.IOUScore(threshold=0.5),
            'dice_coef': dice_coef_multilabel,
            'dice_coef2': dice_coef2
        }

    my_model = tf.keras.models.load_model(model_name, custom_objects=custom_objects, compile=False)

    print(f'\n|Testing of model: {model_name}|\n')
    model_eval(my_model, test_flair_list, test_t1ce_list, test_t2_list, test_t1_list, test_mask_list)

    # run.finish()

    test_img = load_img(
        [training_path + 'BraTS2021_00002/BraTS2021_00002_flair.nii.gz'],
        [training_path + 'BraTS2021_00002/BraTS2021_00002_t1ce.nii.gz'],
        [training_path + 'BraTS2021_00002/BraTS2021_00002_t2.nii.gz'],
        [training_path + 'BraTS2021_00002/BraTS2021_00002_t1.nii.gz'],
        img_channels=channels
    )

    test_mask = load_mask(
        [training_path + 'BraTS2021_00002/BraTS2021_00002_seg.nii.gz'],
        segmenting_subregion=0, 
        classes=4
    )
    test_mask = np.argmax(test_mask, axis=-1)


if __name__ == '__main__':
    main()
