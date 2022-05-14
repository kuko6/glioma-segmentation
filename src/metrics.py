import re
import keras.backend as K
from segmentation_models_3D import losses
import numpy as np
import tensorflow as tf
import math

# -------------------------------------------------------------------------------- #
# Metrics used in training and evaluation.
# Includes:
#   - dice coeficient
# -------------------------------------------------------------------------------- #

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

'''
# multilabel
def dice_coef(y_true, y_pred, numLabels=2):
    # numLabels = y_true.shape[4]
    dice = 0
    for index in range(numLabels):
        dice += dice_coef_binary(y_true[:, :, :, :, index], y_pred[:, :, :, :, index])
    dice = dice / numLabels
    return dice
'''

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

# Dice coeficient defined for each subrehion
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