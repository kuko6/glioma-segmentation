import argparse
import cmath
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

batch_size = 1
subregion = 0
classes = 4
channels = 3
n_slice = 80
model1_path = 'models/separate/3ch/model_1.h5'
model2_path = 'models/separate/3ch/model_2.h5'
model3_path = 'models/separate/3ch/model_3.h5'
custom_cmap = utils.get_custom_cmap()


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


# TODO: - make use of numpy methods in the loop
def combine_predictions(model1, model2, model3, img):
    prediction1 = model1.predict(img)
    prediction1_argmax = np.argmax(prediction1, axis=-1)
    #print(np.unique(prediction1_argmax))

    prediction2 = model2.predict(img)
    prediction2_argmax = np.argmax(prediction2, axis=-1)
    prediction2_argmax[prediction2_argmax == 1] = 2
    #print(np.unique(prediction2_argmax))

    prediction3 = model3.predict(img)
    prediction3_argmax = np.argmax(prediction3, axis=-1)
    prediction3_argmax[prediction3_argmax == 1] = 3
    #print(np.unique(prediction3_argmax))

    #print(prediction1.shape)
    #print(prediction1_argmax.shape)

    combined_prediction = np.zeros((1, 128, 128, 128))
    #print(stacked_prediction.shape)
    #print(np.unique(stacked_prediction))

    predictions = [prediction1, prediction2, prediction3]
    predictions_argmax = [prediction1_argmax, prediction2_argmax, prediction3_argmax]

    for pred, pred_argmax in zip(predictions, predictions_argmax):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    if pred_argmax[0, i, j, k] != 0 and combined_prediction[0, i, j, k] == 0:
                        combined_prediction[0, i, j, k] = pred_argmax[0, i, j, k]
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

    #print(np.unique(stacked_prediction))
    #print(prediction1[0, 1, 1, 1, 0], prediction1[0, 1, 1, 1, 1])
    return combined_prediction


# TODO: - do something with the dice so it doesnt require to_categorical
def predict_image(model1, model2, model3, flair, t1ce, t2, t1, mask, subdir='', counter=10000):
    if not os.path.isdir(f'outputs/{subdir}'):
        os.mkdir(f'outputs/{subdir}')

    img_name = re.search(r"\bBraTS2021_\d+", flair)
    img_name = img_name.group()
    os.mkdir(f'outputs/{subdir}{img_name}/')
    subdir = subdir + f'{img_name}/'

    test_img = load_img([flair], [t1ce], [t2], [t1], img_channels=channels)

    prediction = combine_predictions(model1, model2, model3, test_img)
    
    test_mask = load_mask([mask], segmenting_subregion=subregion, classes=classes)
    test_mask_argmax = np.argmax(test_mask, axis=-1)
    prediction_encoded = to_categorical(prediction, num_classes=4)

    print('dice:', dice_coef_multilabel(classes=classes)(test_mask, prediction_encoded).numpy())
    print('dice edema:', dice_coef_edema(test_mask, prediction_encoded).numpy())
    print('dice necrotic:', dice_coef_necrotic(test_mask, prediction_encoded).numpy())
    print('dice enhancing:', dice_coef_enhancing(test_mask, prediction_encoded).numpy())

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
            ax5.imshow(ndimage.rotate(prediction[0][:, :, i], 270), cmap=custom_cmap)
            ax5.set_title('Prediction')
            ax5.axis('off')
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
            ax4.imshow(ndimage.rotate(prediction[0][:, :, i], 270), cmap=custom_cmap)
            ax4.set_title('Prediction')
            ax4.axis('off')

        fig.savefig(f'outputs/{subdir + img_name}_{i}.png')
        plt.close()

        # flair = ndimage.rotate(test_img[0][:, :, i, 0], 270)
        # t1ce = ndimage.rotate(test_img[0][:, :, i, 1], 270)
        # true_mask = ndimage.rotate(test_mask_argmax[0][:, :, i], 270)
        # pred_mask = ndimage.rotate(prediction[0][:, :, i], 270)

        #mask = wandb_mask(flair, true_mask, pred_mask)
        #wandb.log({f"{subdir}": mask}, step=counter+i)


# TODO: - do something with the dice so it doesnt require to_categorical
#       - save best and worst predictions so it doesnt have to create them again 
def model_eval(model1, model2, model3, flair_list, t1ce_list, t2_list, t1_list, mask_list):
    dice_list = list()
    necrotic_list = list()
    edema_list = list()
    enhancing_list = list()

    i = 0
    for flair_name, t1ce_name, t2_name, t1_name, mask_name in zip(flair_list, t1ce_list, t2_list, t1_list, mask_list):
        # test_img = np.load(img_list[i])
        test_img = load_img([flair_name], [t1ce_name], [t2_name], [t1_name], img_channels=channels)
        test_mask = load_mask([mask_name], segmenting_subregion=subregion, classes=classes)
        # test_mask = np.argmax(test_mask, axis=-1)

        # test_img_input = np.expand_dims(test_img, axis=0)

        prediction = combine_predictions(model1, model2, model3, test_img)
        prediction = to_categorical(prediction, num_classes=4)
        # test_prediction = np.argmax(test_prediction, axis=-1)

        dice_list.append(dice_coef_multilabel(classes=classes)(test_mask, prediction).numpy())
        necrotic_list.append(dice_coef_necrotic(test_mask, prediction).numpy())
        edema_list.append(dice_coef_edema(test_mask, prediction).numpy())
        enhancing_list.append(dice_coef_enhancing(test_mask, prediction).numpy())

        img_name = re.search(r"\bBraTS2021_\d+", flair_list[i])
        print(f"image: {img_name.group()} | {i+1}/{len(flair_list)}")
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
        predict_image(model1, model2, model3, flair=flair_list[i], t1ce=t1ce_list[i], t2=t2_list[i], mask=mask_list[i],
                      subdir='worst/',counter=counter)
        print()
        counter = counter + 10000

    best = np.argsort(dice_list)[-5:]
    print("\nThe best 5:")
    for i in best:
        img_name = re.search(r"\bBraTS2021_\d+", flair_list[i])
        print(f"image: {img_name.group()}, dice = {dice_list[i]}")
        predict_image(model1, model2, model3, flair=flair_list[i], t1ce=t1ce_list[i], t2=t2_list[i], mask=mask_list[i],
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
    #                  name=f"evaluation_separate",
    #                  entity="kuko",
    #                  reinit=True)

    data = args.data_path
    print(os.listdir(data))
    training_path = os.path.join(data, 'train/')
    validation_path = os.path.join(data, 'val/')
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

    model1 = tf.keras.models.load_model(model1_path, custom_objects=custom_objects, compile=False)
    model2 = tf.keras.models.load_model(model2_path, custom_objects=custom_objects, compile=False)
    model3 = tf.keras.models.load_model(model3_path, custom_objects=custom_objects, compile=False)
    
    #model_eval(model1, model2, model3, test_flair_list, test_t1ce_list, test_t2_list, test_t1_list, test_mask_list)

    #run.finish()

    '''
    stacked_prediction = prediction1[:,:,:,:]
    print(stacked_prediction.shape)
    print(np.unique(stacked_prediction))

    stacked_prediction = np.stack([prediction1[:,:,:,:,0], prediction1[:,:,:,:,1], prediction2[:,:,:,:,1], prediction3[:,:,:,:,1]], axis=-1)
    print(stacked_prediction.shape)
    print(np.unique(stacked_prediction))

    stacked_prediction = np.argmax(stacked_prediction, axis=-1)
    print(stacked_prediction.shape)
    print(np.unique(stacked_prediction))
    '''

    test_img = load_img(
        [testing_path + 'BraTS2021_01627/BraTS2021_01627_flair.nii.gz'],
        [testing_path + 'BraTS2021_01627/BraTS2021_01627_t1ce.nii.gz'],
        [testing_path + 'BraTS2021_01627/BraTS2021_01627_t2.nii.gz'],
        [testing_path + 'BraTS2021_01627/BraTS2021_01627_t1.nii.gz'],
        img_channels=channels
    )

    test_mask = load_mask(
        [testing_path + 'BraTS2021_01627/BraTS2021_01627_seg.nii.gz'],
        segmenting_subregion=0, 
        classes=4
    )

    prediction1 = model1.predict(test_img)
    prediction1_argmax = np.argmax(prediction1, axis=-1)
    prediction2 = model2.predict(test_img)
    prediction2_argmax = np.argmax(prediction2, axis=-1)
    prediction2_argmax[prediction2_argmax == 1] = 2
    prediction3 = model3.predict(test_img)
    prediction3_argmax = np.argmax(prediction3, axis=-1)
    prediction3_argmax[prediction3_argmax == 1] = 3

    prediction = combine_predictions(model1, model2, model3, test_img)
    prediction_encoded = to_categorical(prediction, num_classes=4)

    print('dice:', dice_coef_multilabel(classes=classes)(test_mask, prediction_encoded).numpy())
    print('dice edema:', dice_coef_edema(test_mask, prediction_encoded).numpy())
    print('dice necrotic:', dice_coef_necrotic(test_mask, prediction_encoded).numpy())
    print('dice enhancing:', dice_coef_enhancing(test_mask, prediction_encoded).numpy())

    test_mask = np.argmax(test_mask, axis=-1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 0], 270), cmap='gray')
    ax1.set_title('Image flair', fontsize=30)
    ax1.axis('off')
    ax2.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 1], 270), cmap='gray')
    ax2.set_title('Image t1ce', fontsize=30)
    ax2.axis('off')
    fig.savefig(f'outputs/seq.png')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(ndimage.rotate(test_mask[0][:, :, n_slice], 270), cmap=custom_cmap)
    ax1.set_title('Mask')
    ax1.axis('off')
    ax2.imshow(ndimage.rotate(prediction[0][:, :, n_slice], 270), cmap=custom_cmap)
    ax2.set_title('Prediction')
    ax2.axis('off')
    fig.savefig(f'outputs/masks.png')
    plt.close()

    fig, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
    ax1.imshow(ndimage.rotate(prediction1_argmax[0][:, :, n_slice], 270), cmap=custom_cmap)
    ax1.set_title('Prediction NCR')
    ax1.axis('off')
    fig.savefig(f'outputs/ncr.png')
    plt.close()

    fig, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
    ax1.imshow(ndimage.rotate(prediction2_argmax[0][:, :, n_slice], 270), cmap=custom_cmap)
    ax1.set_title('Prediction ED')
    ax1.axis('off')
    fig.savefig(f'outputs/ed.png')
    plt.close()

    fig, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
    ax1.imshow(ndimage.rotate(prediction3_argmax[0][:, :, n_slice], 270), cmap=custom_cmap)
    ax1.set_title('Prediction ET')
    ax1.axis('off')
    fig.savefig(f'outputs/et.png')
    plt.close()

    '''
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1, 7, figsize=(20, 10))
    ax1.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 0], 270), cmap='gray')
    ax1.set_title('Image flair')
    ax2.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 1], 270), cmap='gray')
    ax2.set_title('Image t1ce')
    ax3.imshow(ndimage.rotate(test_mask[0][:, :, n_slice], 270))
    ax3.set_title('Mask')
    ax4.imshow(ndimage.rotate(prediction[0][:, :, n_slice], 270))
    ax4.set_title('Prediction')
    ax5.imshow(ndimage.rotate(prediction1_argmax[0][:, :, n_slice], 270))
    ax5.set_title('Prediction 1')
    ax6.imshow(ndimage.rotate(prediction2_argmax[0][:, :, n_slice], 270))
    ax6.set_title('Prediction 2')
    ax7.imshow(ndimage.rotate(prediction3_argmax[0][:, :, n_slice], 270))
    ax7.set_title('Prediction 3')
    fig.savefig(f'outputs/test.png')
    plt.close()
    '''

if __name__ == '__main__':
    main()
