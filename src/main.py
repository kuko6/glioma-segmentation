import argparse
import os
#from azureml.core import Run

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
    #print(help('modules'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to data')
    args = parser.parse_args()

    #config = configuration()

    data = args.data_path

    '''
    if data == None:
        data = config.data_path
    '''

    print(os.listdir(data))
    training_path = os.path.join(data, 'train/')
    testing = os.path.join(data, 'test/')
    validation = os.path.join(data, 'val/')

    test_image_t1ce=nib.load(training_path + 'BraTS2021_00002/BraTS2021_00002_t1ce.nii.gz').get_fdata()
    test_image_t1=nib.load(training_path + 'BraTS2021_00002/BraTS2021_00002_t1.nii.gz').get_fdata()
    test_image_flair=nib.load(training_path + 'BraTS2021_00002/BraTS2021_00002_flair.nii.gz').get_fdata()
    test_image_t2=nib.load(training_path + 'BraTS2021_00002/BraTS2021_00002_t2.nii.gz').get_fdata()
    test_mask=nib.load(training_path + 'BraTS2021_00002/BraTS2021_00002_seg.nii.gz').get_fdata()

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))

    ax1.imshow(test_image_flair[:,:,80], cmap = 'gray')
    ax1.set_title('Image flair')
    ax2.imshow(test_image_t1[:,:,80], cmap = 'gray')
    ax2.set_title('Image t1')
    ax3.imshow(test_image_t1ce[:,:,80], cmap = 'gray')
    ax3.set_title('Image t1ce')
    ax4.imshow(test_image_t2[:,:,80], cmap = 'gray')
    ax4.set_title('Image t2')
    ax5.imshow(test_mask[:,:,80])
    ax5.set_title('Mask')



    fig.savefig('outputs/images/test.png')
    #plt.imsave('outputs/images/test.png', fig, cmap='gray')

    '''
    training_files = list_files(training_path)
    validation_files = list_files(validation)
    testing_files = list_files(testing)

    print(f'training_path: {len(training_files)}, validation: {len(validation_files)}, testing: {len(testing_files)}')
    '''
