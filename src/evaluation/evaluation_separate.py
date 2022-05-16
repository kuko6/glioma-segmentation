import argparse
# from azureml.core import Run

import os
from pyexpat import model
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage
import glob
from scipy import ndimage
import segmentation_models_3D as sm
from scipy.spatial.distance import directed_hausdorff
from tensorflow.keras.utils import to_categorical
import cv2
import tensorflow as tf
import re

import wandb

import utils.utils as utils
from utils.data_processing import *
import losses
from metrics import *

# -------------------------------------------------------------------------------------------- #
# Script used for evaluating models segmenting each class separately.
# Combines the individual predictions and evaluates the specified model on the testing dataset.
# Saves the individual slice for the 5 worst and best predictions.
#
# Its important to set the configuration variables.
# -------------------------------------------------------------------------------------------- #

#batch_size = 1
#subregion = 0

classes = 4
channels = 3
n_slice = 80
model1_name = '../../models/separate/3ch/model_NCR.h5'
model2_name = '../../models/separate/3ch/model_ED.h5'
model3_name = '../../models/separate/3ch/model_ET.h5'
custom_cmap = utils.get_custom_cmap()


# combines the individual predictions based on based on their values in the selected voxel, where 
# in case of a collision, or when multiple models segmented the same voxel, we chose the one 
# with a higher value in the probability map returned by the respective model
def combine_predictions(model1, model2, model3, img):
    # generate individual segmentations and set the corresponding values
    prediction1 = model1.predict(img)
    prediction1_argmax = np.argmax(prediction1, axis=-1)

    prediction2 = model2.predict(img)
    prediction2_argmax = np.argmax(prediction2, axis=-1)
    prediction2_argmax[prediction2_argmax == 1] = 2

    prediction3 = model3.predict(img)
    prediction3_argmax = np.argmax(prediction3, axis=-1)
    prediction3_argmax[prediction3_argmax == 1] = 3

    combined_prediction = np.zeros((1, 128, 128, 128))

    predictions = [prediction1, prediction2, prediction3]
    predictions_argmax = [prediction1_argmax, prediction2_argmax, prediction3_argmax]

    # merging of the predictions
    for pred, pred_argmax in zip(predictions, predictions_argmax):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    if pred_argmax[0, i, j, k] != 0 and combined_prediction[0, i, j, k] == 0:
                        combined_prediction[0, i, j, k] = pred_argmax[0, i, j, k]
                    # colisions
                    elif combined_prediction[0, i, j, k] != 0: 
                        if combined_prediction[0, i, j, k] == 1:
                            if prediction1[0, i, j, k, 1] < pred[0, i, j, k, 1]:
                                combined_prediction[0, i, j, k] = pred_argmax[0, i, j, k]

                        if combined_prediction[0, i, j, k] == 2:
                            if prediction2[0, i, j, k, 1] < pred[0, i, j, k, 1]:
                                combined_prediction[0, i, j, k] = pred_argmax[0, i, j, k]

                        if combined_prediction[0, i, j, k] == 3:
                            if prediction3[0, i, j, k, 1] < pred[0, i, j, k, 1]:
                                combined_prediction[0, i, j, k] = pred_argmax[0, i, j, k]

    return combined_prediction


# segments the specified image and saves the individual slices into outputs/{subdir}
def predict_image(model1, model2, model3, flair, t1ce, t2, t1, mask, subdir=''):
    if not os.path.isdir(f'outputs/{subdir}'):
        os.mkdir(f'outputs/{subdir}')

    img_name = re.search(r"\bBraTS2021_\d+", flair)
    img_name = img_name.group()
    os.mkdir(f'outputs/{subdir}{img_name}/')
    subdir = subdir + f'{img_name}/'

    test_img = load_img([flair], [t1ce], [t2], [t1], img_channels=channels)

    prediction = combine_predictions(model1, model2, model3, test_img)
    prediction_encoded = to_categorical(prediction, num_classes=4)

    test_mask = load_mask([mask], segmenting_subregion=0, classes=classes)
    test_mask_argmax = np.argmax(test_mask, axis=-1)

    # prints dice coefficient for the given image
    print('dice:', dice_coef_multilabel(classes=classes)(test_mask, prediction_encoded).numpy())
    print('dice edema:', dice_coef_edema(test_mask, prediction_encoded).numpy())
    print('dice necrotic:', dice_coef_necrotic(test_mask, prediction_encoded).numpy())
    print('dice enhancing:', dice_coef_enhancing(test_mask, prediction_encoded).numpy())

    # prints Hausdorff distance for the given image
    hd, hd_classes = hausdorff_distance(test_mask[0], prediction_encoded[0])
    print('Hausdorff mean:', hd)
    print('Hausdorff edema:', hd_classes[1])
    print('Hausdorff necrotic:', hd_classes[0])
    print('Hausdorff enhancing:', hd_classes[2])

    volume_start = 20
    volume_end = 126 # 125
    step = 2 # 5
    for i in range(volume_start, volume_end+step, step):
        if channels == 3:
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
            ax1.imshow(ndimage.rotate(test_img[0][:, :, i, 0], 270), cmap='gray')
            ax1.set_title('FLAIR')
            ax1.axis('off')
            ax2.imshow(ndimage.rotate(test_img[0][:, :, i, 1], 270), cmap='gray')
            ax2.set_title('T1ce')
            ax2.axis('off')
            ax3.imshow(ndimage.rotate(test_img[0][:, :, i, 2], 270), cmap='gray')
            ax3.set_title('T2')
            ax3.axis('off')
            ax4.imshow(ndimage.rotate(test_mask_argmax[0][:, :, i], 270), cmap=custom_cmap)
            ax4.set_title('Mask')
            ax4.axis('off')
            ax5.imshow(ndimage.rotate(prediction[0][:, :, i], 270), cmap=custom_cmap)
            ax5.set_title('Prediction')
            ax5.axis('off')
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
            ax1.imshow(ndimage.rotate(test_img[0][:, :, i, 0], 270), cmap='gray')
            ax1.set_title('FLAIR')
            ax1.axis('off')
            ax2.imshow(ndimage.rotate(test_img[0][:, :, i, 1], 270), cmap='gray')
            ax2.set_title('T1ce')
            ax2.axis('off')
            ax3.imshow(ndimage.rotate(test_mask_argmax[0][:, :, i], 270), cmap=custom_cmap)
            ax3.set_title('Mask')
            ax3.axis('off')
            ax4.imshow(ndimage.rotate(prediction[0][:, :, i], 270), cmap=custom_cmap)
            ax4.set_title('Prediction')
            ax4.axis('off')

        fig.savefig(f'outputs/{subdir + img_name}_{i}.png')
        plt.close()


# runs the evaluation of the provided models on the testing dataset
def model_eval(model1, model2, model3, flair_list, t1ce_list, t2_list, t1_list, mask_list):
    dice_list = list()
    necrotic_list = list()
    edema_list = list()
    enhancing_list = list()

    hd_list = list()
    hd_necrotic_list = list() 
    hd_edema_list = list()
    hd_enhancing_list = list()

    i = 0
    for flair_name, t1ce_name, t2_name, t1_name, mask_name in zip(flair_list, t1ce_list, t2_list, t1_list, mask_list):
        test_img = load_img([flair_name], [t1ce_name], [t2_name], [t1_name], img_channels=channels)
        test_mask = load_mask([mask_name], segmenting_subregion=0, classes=classes)

        prediction = combine_predictions(model1, model2, model3, test_img)
        prediction = to_categorical(prediction, num_classes=4)
        
        # computes dice coefficient for each subregion
        dice_list.append(dice_coef_multilabel(classes=classes)(test_mask, prediction).numpy())
        necrotic_list.append(dice_coef_necrotic(test_mask, prediction).numpy())
        edema_list.append(dice_coef_edema(test_mask, prediction).numpy())
        enhancing_list.append(dice_coef_enhancing(test_mask, prediction).numpy())

        # computes Hausdorff distance for each subregion
        hd, hd_classes = hausdorff_distance(test_mask[0], prediction[0])
        hd_list.append(hd)
        hd_necrotic_list.append(hd_classes[0])
        hd_edema_list.append(hd_classes[1])
        hd_enhancing_list.append(hd_classes[2])

        # prints dice coefficient for the given image
        img_name = re.search(r"\bBraTS2021_\d+", flair_list[i])
        print(f"image: {img_name.group()} | {i+1}/{len(flair_list)}")
        print(f"dice_coef: {dice_list[i]} | necrotic: {necrotic_list[i]} | edema: {edema_list[i]} | enhancing: {enhancing_list[i]}")
        print(f"Hausdorff: {hd_list[i]} | Hausdorff necrotic: {hd_necrotic_list[i]} | Hausdorff edema: {hd_edema_list[i]} | Hausdorff enhancing: {hd_enhancing_list[i]}")
        print()
        i += 1

    # prints the final mean dice coefficient
    print(f"\ndice_mean: {np.mean(dice_list)} | necrotic_mean: {np.mean(necrotic_list)} | edema_mean: {np.mean(edema_list)} | enhancing_mean: {np.mean(enhancing_list)}")
    print(f"hausdorff_mean: {np.nanmean(hd_list)} | hausdorff_necrotic_mean: {np.nanmean(hd_necrotic_list)} | hausdorff_edema_mean: {np.nanmean(hd_edema_list)} | hausdorff_enhancing_mean: {np.nanmean(hd_enhancing_list)}\n")

    # prints the 5 worst predictions
    worst = np.argsort(dice_list)[:5]
    print("\nThe worst 5:")
    for i in worst:
        img_name = re.search(r"\bBraTS2021_\d+", flair_list[i])
        print(f"image: {img_name.group()}, dice = {dice_list[i]}")
        predict_image(model1, model2, model3, flair=flair_list[i], t1ce=t1ce_list[i], t2=t2_list[i], t1=t1_list[i], mask=mask_list[i],
                      subdir='worst/')
        print()

    # prints the 5 best predictions
    best = np.argsort(dice_list)[-5:]
    print("\nThe best 5:")
    for i in best:
        img_name = re.search(r"\bBraTS2021_\d+", flair_list[i])
        print(f"image: {img_name.group()}, dice = {dice_list[i]}")
        predict_image(model1, model2, model3, flair=flair_list[i], t1ce=t1ce_list[i], t2=t2_list[i], t1=t1_list[i], mask=mask_list[i],
                      subdir='best/')
        print()

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--wandb', type=str, help='wandb id')
    args = parser.parse_args()

    if args.data_path:
        data = args.data_path
    else:
        data = os.path.join(os.getcwd(), '../../BraTS2021')
    print(os.listdir(data))
    #training_path = os.path.join(data, 'train/')
    #validation_path = os.path.join(data, 'val/')
    testing_path = os.path.join(data, 'test/')

    test_flair_list = glob.glob(testing_path + '/*/*flair.nii.gz')
    test_t1ce_list = glob.glob(testing_path + '/*/*t1ce.nii.gz')
    test_t2_list = glob.glob(testing_path + '/*/*t2.nii.gz')
    test_t1_list = glob.glob(testing_path + '/*/*t1.nii.gz')
    test_mask_list = glob.glob(testing_path + '/*/*seg.nii.gz')

    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    custom_objects = {
        'iou_score': sm.metrics.IOUScore(threshold=0.5),
        'dice_coef': dice_coef_multilabel,
        'dice_coef2': dice_coef2
    }

    model1 = tf.keras.models.load_model(model1_name, custom_objects=custom_objects, compile=False)
    model2 = tf.keras.models.load_model(model2_name, custom_objects=custom_objects, compile=False)
    model3 = tf.keras.models.load_model(model3_name, custom_objects=custom_objects, compile=False)
    
    model_eval(model1, model2, model3, test_flair_list, test_t1ce_list, test_t2_list, test_t1_list, test_mask_list)


if __name__ == '__main__':
    main()
