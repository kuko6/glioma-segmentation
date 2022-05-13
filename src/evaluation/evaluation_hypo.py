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
from tensorflow.keras.utils import to_categorical
import re
import keras.backend as K

import utils.utils as utils
import losses
from metrics import *

def dice_coef_binary(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# multilabel 2
def dice_coef(y_true, y_pred, numLabels=2):
    # numLabels = y_true.shape[4]
    dice = 0
    for index in range(numLabels):
        dice += dice_coef_binary(y_true[:, :, :, :, index], y_pred[:, :, :, :, index])
    dice = dice / numLabels
    return dice


def prepare_img8_sagittal(img8_path, mask8_path):
  img8 = nib.load(img8_path).get_fdata()
  mask8 = nib.load(mask8_path).get_fdata()

  volume_start = 40
  volume_end = 141
  total_slices = 128
  tmp_img8 = np.zeros((128, 128, 128))
  tmp_mask8 = np.zeros((128, 128, 128))
  # resize the images and mask
  inter = cv2.INTER_NEAREST
  for i in range(total_slices):
      tmp_img8[:, :, i] = cv2.resize(img8[:, :, i + volume_start], (128, 128), interpolation=inter)
      tmp_mask8[:, :, i] = cv2.resize(mask8[:, :, i + volume_start], (128, 128), interpolation=inter)
  #tmp_img8 = np.swapaxes(tmp_img8, 1, 2)
  tmp_img8 = utils.normalise(tmp_img8)
  #tmp_mask8 = np.swapaxes(tmp_mask8, 1, 2)

  image = np.stack([tmp_img8, tmp_img8], axis=3)
  mask = to_categorical(tmp_mask8, num_classes=4)
  return np.array([image]), np.array([mask])


def prepare_img8_axial(img8_path, mask8_path):
  img8 = nib.load(img8_path).get_fdata()
  mask8 = nib.load(mask8_path).get_fdata()

  volume_start = 40
  volume_end = 141
  total_slices = 128
  tmp_img8 = np.zeros((128, 128, 128))
  tmp_mask8 = np.zeros((128, 128, 128))
  # resize the images and mask
  inter = cv2.INTER_NEAREST
  for i in range(total_slices):
      tmp_img8[:, i, :] = cv2.resize(img8[:, i + volume_start, :], (128, 128), interpolation=inter)
      tmp_mask8[:, i, :] = cv2.resize(mask8[:, i + volume_start, :], (128, 128), interpolation=inter)
  tmp_img8 = np.swapaxes(tmp_img8, 1, 2)
  tmp_img8 = utils.normalise(tmp_img8)
  tmp_mask8 = np.swapaxes(tmp_mask8, 1, 2)

  image = np.stack([tmp_img8, tmp_img8], axis=3)
  mask = to_categorical(tmp_mask8, num_classes=4)
  return np.array([image]), np.array([mask])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, help='path to data')
  parser.add_argument('--wandb', type=str, help='wandb id')
  args = parser.parse_args()

  data = args.data_path
  print(os.listdir(data))

  brains_path = os.path.join(data, 'Hotove_segmentacie/')
  print(os.listdir(brains_path))

  if not os.path.isdir('outputs'):
      os.mkdir('outputs')

  img4_path = brains_path + '4/t1_mprage_tra_iso_ce.nii'
  mask4_path = brains_path + '4/Segmentation-label.nii'

  img26_path = brains_path + '26/t1_mprage_tra_iso_ce.nii'
  mask26_path = brains_path + '26/Segmentation-label.nii'

  # tieto maju iny tvar
  img8_path = brains_path + '8/t1_mpr_sag_p2_iso.nii'
  mask8_path = brains_path + '8/Segmentation-label.nii'

  custom_objects = {
    'iou_score': sm.metrics.IOUScore(threshold=0.5),
    'dice_coef': dice_coef_multilabel,
    'dice_coef2': dice_coef2,
    'dice_coef_edema': dice_coef_edema,
    'dice_coef_necrotic': dice_coef_necrotic,
    'dice_coef_enhancing': dice_coef_enhancing
  }

  model_name = 'models/categorical_crossentropy_50_2ch_sub0_aug.h5'
  my_model = tf.keras.models.load_model(model_name, custom_objects=custom_objects, compile=False)

  # Img 4
  test_img4 = utils.load_img([img4_path], [img4_path], [img4_path], img_channels=2)
  test_mask4 = utils.load_mask([mask4_path], segmenting_subregion=0)

  test_prediction4 = my_model.predict(test_img4)
  test_prediction4[:,:,:,:,1] += test_prediction4[:,:,:,:,2]
  test_prediction4[:,:,:,:,1] += test_prediction4[:,:,:,:,3]
  print('img4 dice:', dice_coef(test_mask4, test_prediction4).numpy())

  test_prediction4 = np.argmax(test_prediction4, axis=-1)
  test_mask4 = np.argmax(test_mask4, axis=-1)

  # Img 26
  test_img26 = utils.load_img([img26_path], [img26_path], [img26_path], img_channels=2)
  test_mask26 = utils.load_mask([mask26_path], segmenting_subregion=0)

  test_prediction26 = my_model.predict(test_img26)
  test_prediction26[:,:,:,:,1] += test_prediction26[:,:,:,:,2]
  test_prediction26[:,:,:,:,1] += test_prediction26[:,:,:,:,3]
  print('img26 dice:', dice_coef(test_mask26, test_prediction26).numpy())

  test_mask26 = np.argmax(test_mask26, axis=-1)
  test_prediction26 = np.argmax(test_prediction26, axis=-1)

  # Img 8
  test_img8, test_mask8 = prepare_img8_axial(img8_path, mask8_path)
  #test_img8, test_mask8 = prepare_img8_sagittal(img8_path, mask8_path)

  test_prediction8 = my_model.predict(test_img8)
  test_prediction8[:,:,:,:,1] += test_prediction8[:,:,:,:,2]
  test_prediction8[:,:,:,:,1] += test_prediction8[:,:,:,:,3]
  print('img8 dice:', dice_coef(test_mask8, test_prediction8).numpy())

  test_mask8 = np.argmax(test_mask8, axis=-1)
  test_prediction8 = np.argmax(test_prediction8, axis=-1)

  os.mkdir('outputs/4')
  os.mkdir('outputs/26')
  os.mkdir('outputs/8')

  volume_start = 20
  volume_end = 126
  step = 2
  for i in range(volume_start, volume_end+step, step):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(ndimage.rotate(test_img4[0][:, :, i, 0], 270), cmap='gray')
    ax1.set_title('Image 4')
    ax2.imshow(ndimage.rotate(test_mask4[0][:, :, i], 270))
    ax2.set_title('Mask 4')
    ax3.imshow(ndimage.rotate(test_prediction4[0][:, :, i], 270))
    ax3.set_title('Prediction 4')
    fig.savefig(f'outputs/4/slice_{i}.png')
    plt.close()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(ndimage.rotate(test_img26[0][:, :, i, 0], 270), cmap='gray')
    ax1.set_title('Image 26')
    ax2.imshow(ndimage.rotate(test_mask26[0][:, :, i], 270))
    ax2.set_title('Mask 26')
    ax3.imshow(ndimage.rotate(test_prediction26[0][:, :, i], 270))
    ax3.set_title('Prediction 26')
    fig.savefig(f'outputs/26/slice_{i}.png')
    plt.close()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(test_img8[0][:, :, volume_end-i, 0], cmap='gray')
    ax1.set_title('Image 8')
    ax2.imshow(test_mask8[0][:, :, volume_end-i])
    ax2.set_title('Mask 8')
    ax3.imshow(test_prediction8[0][:, :, volume_end-i])
    ax3.set_title('Prediction 8')
    fig.savefig(f'outputs/8/slice_{i}.png')
    plt.close()


if __name__ == '__main__':
  main()