import argparse
import os


# import cv2 # dalo by sa pouzit na resize
from scipy import ndimage
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def list_files(path):
    brains = os.listdir(path)
    files = []
    for brain in brains:
        brain_path = os.path.join(path, brain)
        files += [os.path.join(brain_path, file) for file in os.listdir(brain_path)]
    #print(len(files))

    return files

if __name__ == '__main__':
    data = '/Users/kuko/Desktop/projekt/BraTS2021'
    print(os.listdir(data))
    training_path = os.path.join(data, 'train/')
    testing_path = os.path.join(data, 'test/')
    validation_path = os.path.join(data, 'val/')

    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    test_image_t1ce=nib.load(training_path + 'BraTS2021_00002/BraTS2021_00002_t1ce.nii.gz').get_fdata()
    test_image_t1=nib.load(training_path + 'BraTS2021_00002/BraTS2021_00002_t1.nii.gz').get_fdata()
    test_image_flair=nib.load(training_path + 'BraTS2021_00002/BraTS2021_00002_flair.nii.gz').get_fdata()
    test_image_t2=nib.load(training_path + 'BraTS2021_00002/BraTS2021_00002_t2.nii.gz').get_fdata()
    test_mask=nib.load(training_path + 'BraTS2021_00002/BraTS2021_00002_seg.nii.gz').get_fdata()

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
    ax1.imshow(ndimage.rotate(test_image_flair[:,:,80], 270), cmap = 'gray')
    ax1.set_title('Image flair')
    ax2.imshow(ndimage.rotate(test_image_t1[:,:,80], 270), cmap = 'gray')
    ax2.set_title('Image t1')
    ax3.imshow(ndimage.rotate(test_image_t1ce[:,:,80], 270), cmap = 'gray')
    ax3.set_title('Image t1ce')
    ax4.imshow(ndimage.rotate(test_image_t2[:,:,80], 270), cmap = 'gray')
    ax4.set_title('Image t2')
    ax5.imshow(ndimage.rotate(test_mask[:,:,80], 270))
    ax5.set_title('Mask')
    plt.show()

    fig.savefig('outputs/test.png')

    '''
    plt.imsave('outputs/flair.png', test_image_flair[:,:,80], cmap = 'gray')
    plt.imsave('outputs/t1.png', test_image_t1[:,:,80], cmap = 'gray')
    plt.imsave('outputs/t1ce.png', test_image_t1ce[:,:,80], cmap = 'gray')
    plt.imsave('outputs/t2.png', test_image_t2[:,:,80], cmap = 'gray')
    plt.imsave('outputs/mask.png', test_mask[:,:,80], cmap = 'gray')
    '''

    '''
    training_files = list_files(training_path)
    validation_files = list_files(validation)
    testing_files = list_files(testing)

    print(f'training_path: {len(training_files)}, validation: {len(validation_files)}, testing: {len(testing_files)}')
    '''