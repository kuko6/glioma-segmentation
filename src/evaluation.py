import argparse
# from azureml.core import Run

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import ndimage
import segmentation_models_3D as sm
import cv2
import tensorflow as tf

import utils
import losses


def main():
    # print(help('modules'))

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to data')
    parser.add_argument('--wandb', type=str, help='wandb id')
    args = parser.parse_args()

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

    batch_size = 2
    subregion = 1
    test_img_datagen = utils.image_loader(train_flair_list, train_t1ce_list, train_t2_list, train_mask_list, batch_size,
                                          subregion)

    '''
    my_model = tf.keras.models.load_model('models/model_2.h5',
                                          custom_objects={'iou_score': sm.metrics.IOUScore(threshold=0.5),
                                                          'dice_loss_plus_1focal_loss': losses.loss,
                                                          'dice_coef': losses.dice_coef,
                                                          'dice_coef2': losses.dice_coef2
                                                          })

    '''
    my_model = tf.keras.models.load_model('model_2.h5',
                                          custom_objects={'iou_score': sm.metrics.IOUScore(threshold=0.5),
                                                          'dice_coef': losses.dice_coef,
                                                          'dice_coef2': losses.dice_coef2,
                                                          })

    # test_img, test_mask = test_img_datagen.__next__()
    test_img = utils.load_img([training_path + 'BraTS2021_00002/BraTS2021_00002_flair.nii.gz'],
                              [training_path + 'BraTS2021_00002/BraTS2021_00002_t1ce.nii.gz'],
                              [training_path + 'BraTS2021_00002/BraTS2021_00002_t1.nii.gz'])

    #test_prediction = my_model.predict(test_img[:,:,:,:,0:2])
    test_prediction = my_model.predict(test_img)
    print(np.unique(test_prediction))
    test_prediction_argmax=np.argmax(test_prediction, axis=4)
    print(np.unique(test_prediction_argmax))
    print('original shape: ', test_prediction.shape)
    print('new shape: ', test_prediction_argmax.shape)

    test_mask = utils.load_mask([training_path + 'BraTS2021_00002/BraTS2021_00002_seg.nii.gz'], segmenting_subregion=2)
    test_mask_argmax=np.argmax(test_mask, axis=4)
    print('mask shape: ', test_mask.shape)

    print('dice:', losses.dice_coef(test_mask, test_prediction).numpy())
    #print('dice edema:', losses.dice_coef_edema(test_mask, test_prediction).numpy())
    #print('dice necrotic:', losses.dice_coef_necrotic(test_mask, test_prediction).numpy())
    #print('dice enhancing:', losses.dice_coef_enhancing(test_mask, test_prediction).numpy())

    n_slice = 80
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
    ax1.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 0], 270), cmap='gray')
    ax1.set_title('Testing Flair')
    ax2.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 1], 270), cmap='gray')
    ax2.set_title('Testing T1ce')
    ax3.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 2], 270), cmap='gray')
    ax3.set_title('Testing T2')
    ax4.imshow(ndimage.rotate(test_mask_argmax[0][:, :, n_slice], 270))
    ax4.set_title('Mask')
    #ax5.imshow(ndimage.rotate(test_prediction[0][:,:, n_slice, 1], 270))
    ax5.imshow(ndimage.rotate(test_prediction_argmax[0][:,:, n_slice], 270))
    ax5.set_title('Prediction')
    fig.savefig('outputs/test.png')

    '''
    test_mask = utils.load_mask(test_mask_list[:1], 2)
    test_mask = test_mask[0]
    test_mask_argmax = np.argmax(test_mask, axis=3)
    print(test_prediction.shape)
    test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]
    print(test_prediction_argmax.shape)

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
