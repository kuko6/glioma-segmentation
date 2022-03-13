import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

def test():
    print("test")

def load_img(flair_list, t1ce_list, t2_list):
    scaler = MinMaxScaler()
    images=[]
    for flair_name, t1ce_name, t2_name in zip(flair_list, t1ce_list, t2_list):
        temp_image_flair = nib.load(flair_name).get_fdata()
        print(np.max(temp_image_flair))
        temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
        print("========================================")
        print(np.max(temp_image_flair))
        temp_image_t1ce = nib.load(t1ce_name).get_fdata()
        temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)

        temp_image_t2 = nib.load(t2_name).get_fdata()
        temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

        image = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
        image = image[56:184, 56:184, 13:141]

        images.append(image)

    images = np.array(images)

    return(images)

def load_mask(mask_list):
    images=[]
    for mask_name in mask_list:
        temp_mask = nib.load(mask_name).get_fdata()
        temp_mask = temp_mask.astype(np.uint8)
        temp_mask[temp_mask==4] = 3  # Reassign mask values 4 to 3
        image = temp_mask[56:184, 56:184, 13:141]
        image=to_categorical(image, num_classes=4)

        images.append(image)

    images = np.array(images)

    return(images)

def imageLoader(flair_list, t1ce_list, t2_list, mask_list, batch_size):
    img_len = len(flair_list)

    # keras needs the generator infinite
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < img_len:
            limit = min(batch_end, img_len)
            x = load_img(flair_list[batch_start:limit], t1ce_list[batch_start:limit], t2_list[batch_start:limit])
            y = load_mask(mask_list[batch_start:limit])

            yield (x,y) # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size
