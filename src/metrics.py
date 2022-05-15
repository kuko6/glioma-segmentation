from cmath import inf
import keras.backend as K
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff
from tensorflow.keras.utils import to_categorical

# -------------------------------------------------------------------------------- #
# Metrics used in training and evaluation.
# Includes:
#   - dice coeficient
# -------------------------------------------------------------------------------- #

# a implementation of for multiclass segmentation
# computes a mean of dice coefficient for each class
# https://github.com/keras-team/keras/issues/9395#issuecomment-379276452
def dice_coef_binary(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_multilabel(classes=4):
    def dice_coef(y_true, y_pred):
        dice = 0
        for index in range(classes):
            dice += dice_coef_binary(y_true[:, :, :, :, index], y_pred[:, :, :, :, index])
        dice = dice / classes
        return dice
    return dice_coef


# a different implementation of dice coefficient
def dice_coef2(y_true, y_pred, epsilon=1e-6):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    #axis = (0, 1)
    axis = (0, 1, 2, 3)
    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true * y_true, axis=axis) + K.sum(y_pred * y_pred, axis=axis) + epsilon
    return K.mean((dice_numerator) / (dice_denominator))


# Dice coeficient defined for each subregion
def dice_coef_necrotic(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true[:, :, :, :, 1])
    y_pred_f = K.flatten(y_pred[:, :, :, :, 1])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_edema(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true[:, :, :, :, 2])
    y_pred_f = K.flatten(y_pred[:, :, :, :, 2])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_enhancing(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true[:, :, :, :, 3])
    y_pred_f = K.flatten(y_pred[:, :, :, :, 3])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Hausdorff distance
# based on Henry et al. (https://arxiv.org/abs/2011.01045)
def hausdorff_distance(y_true, y_pred, classes=[1, 2, 3]):
    y_pred_argmax = np.argmax(y_pred, axis=-1)
    y_pred = to_categorical(y_pred_argmax, num_classes=4)

    hd_classes = []
    for i in classes:
        if np.sum(y_pred[:,:,:,i]) == 0:
            if np.sum(y_true[:,:,:,i]) == 0: hd = np.nan
            else: hd = 99
        else:
            true_coords = np.argwhere(y_true[:,:,:,i])
            pred_coords = np.argwhere(y_pred[:,:,:,i])
            hd = directed_hausdorff(pred_coords, true_coords)[0]
            if hd == inf: hd = 99
        hd_classes.append(hd)

    return np.nanmean(hd_classes), hd_classes


'''
# sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    return true_positives / (possible_positives + K.epsilon())


# specificity
def specificity(y_true, y_pred):
    y_pred_argmax = np.argmax(y_pred, axis=-1)
    y_pred = to_categorical(y_pred_argmax, num_classes=4)
    
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    
    return true_negatives / (possible_negatives + K.epsilon())
'''