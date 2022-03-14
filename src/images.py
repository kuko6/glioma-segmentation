'''
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

    fig.savefig('outputs/test.png')
    #plt.imsave('outputs/images/test.png', fig, cmap='gray')

    t2_list = glob.glob(training_path + '/*/*t2.nii.gz')
    t1ce_list = glob.glob(training_path + '/*/*t1ce.nii.gz')
    flair_list = glob.glob(training_path + '/*/*flair.nii.gz')
    mask_list = glob.glob(training_path + '/*/*seg.nii.gz')
    
    print(len(t2_list))

    batch_size = 4
    train_img_datagen = utils.imageLoader(flair_list, t1ce_list, t2_list, mask_list, batch_size)
    img, msk = train_img_datagen.__next__()

    #img_num = random.randint(0,img.shape[0]-1)
    img_num = 2
    test_img = img[img_num]
    test_mask = msk[img_num]
    test_mask = np.argmax(test_mask, axis=3)

    #n_slice=random.randint(0, test_mask.shape[2])
    n_slice = 80
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20, 10))
    ax1.imshow(ndimage.rotate(test_img[:, :, n_slice, 0], 270), cmap='gray')
    ax1.set_title('Image flair')
    ax2.imshow(ndimage.rotate(test_img[:, :, n_slice, 1], 270), cmap='gray')
    ax2.set_title('Image t1ce')
    ax3.imshow(ndimage.rotate(test_img[:, :, n_slice, 2], 270), cmap='gray')
    ax3.set_title('Image t2')
    ax4.imshow(ndimage.rotate(test_mask[:, :, n_slice], 270))
    ax4.set_title('Mask')
    #plt.show()
    fig.savefig('outputs/test.png')
    #test_mask.shape
'''