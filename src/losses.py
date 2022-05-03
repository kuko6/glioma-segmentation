import re
import keras.backend as K
from segmentation_models_3D import losses
import numpy as np
import tensorflow as tf
import math


def weighted_categorical_crossentropy(weights):
    # weights = [0.9,0.05,0.04,0.01]
    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        # if not K.is_tensor(y_pred): y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)

    return wcce


# https://github.com/keras-team/keras/issues/9395#issuecomment-379228094
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = K.ones_like(y_true)
    p0 = y_pred  # p that voxels are class i
    p1 = ones - y_pred  # p that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


'''
# multilabel 
def dice_coef(y_true, y_pred, smooth=1.0, classes=4):
    total_loss = 0

    for i in range(classes):
        y_true_f = K.flatten(y_true[:,:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / classes

    return total_loss
'''


# https://github.com/keras-team/keras/issues/9395#issuecomment-379276452
# binary
def dice_coef_binary(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef(classes=4):
    def dsc(y_true, y_pred):
        dice = 0
        for index in range(classes):
            dice += dice_coef_binary(y_true[:, :, :, :, index], y_pred[:, :, :, :, index])
        dice = dice / classes
        return dice
    return dsc

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

def dice_loss(y_true, y_pred, numLabels=4):
    dice = 0
    for index in range(numLabels):
        dice -= dice_coef_binary(y_true[:, :, :, :, index], y_pred[:, :, :, :, index])
        #weight = 1/ sum(y_true[:,:,:,:,index])
        #dice -= weight * dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice


def dice_coef_binary_loss(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


'''
def dice_loss(y_true, y_pred, numLabels=4):
    dice = dice_coef(y_true, y_pred)
    return 1.0 - dice
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


'''
# define per class evaluation of dice coef
# inspired by https://github.com/keras-team/keras/issues/9395
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, :, 1] * y_pred[:, :, :, :, 1]))
    return (2. * intersection) / (
                K.sum(K.square(y_true[:, :, :, :, 1])) + K.sum(K.square(y_pred[:, :, :, :, 1])) + epsilon)


def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, :, 2] * y_pred[:, :, :, :, 2]))
    return (2. * intersection) / (
                K.sum(K.square(y_true[:, :, :, :, 2])) + K.sum(K.square(y_pred[:, :, :, :, 2])) + epsilon)


def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, :, 3] * y_pred[:, :, :, :, 3]))
    return (2. * intersection) / (
                K.sum(K.square(y_true[:, :, :, :, 3])) + K.sum(K.square(y_pred[:, :, :, :, 3])) + epsilon)
'''


def loss():
    wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25
    dice_loss = losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
    # dice_loss = losses.DiceLoss()
    # focal_loss = losses.CategoricalFocalLoss()
    # total_loss = dice_loss + (1 * focal_loss)

    # wt0, wt1 = 0.50, 0.50
    # dice_loss = losses.DiceLoss(class_weights=np.array([wt0, wt1]))
    total_loss = dice_loss
    return total_loss
