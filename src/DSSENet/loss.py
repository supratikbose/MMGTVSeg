import tensorflow as tf
from tensorflow.keras import backend  as K #import tensorflow.keras.backend as K

# mean Dice loss (mean of multiple labels with option to ignore zero (background) label)
def dice_coef(y_true, y_pred, smooth = 1.0, squared_denominator = False, ignore_zero_label = True):
    num_dim = len(K.int_shape(y_pred)) 
    num_labels = K.int_shape(y_pred)[-1]
    reduce_axis = list(range(1, num_dim - 1))
    y_true = y_true[..., 0]
    dice = 0.0

    if (ignore_zero_label == True):
        label_range = range(1, num_labels)
    else:
        label_range = range(0, num_labels)

    for i in label_range:
        y_pred_b = y_pred[..., i]
        y_true_b = K.cast(K.equal(y_true, i), K.dtype(y_pred))
        intersection = K.sum(y_true_b * y_pred_b, axis = reduce_axis)        
        if squared_denominator: 
            y_pred_b = K.square(y_pred_b)
        y_true_o = K.sum(y_true_b, axis = reduce_axis)
        y_pred_o =  K.sum(y_pred_b, axis = reduce_axis)     
        d = (2. * intersection + smooth) / (y_true_o + y_pred_o + smooth) 
        dice = dice + K.mean(d)
    dice = dice / len(label_range)
    return dice

def dice_loss(y_true, y_pred):
    f = 1 - dice_coef(y_true, y_pred, squared_denominator = False, ignore_zero_label = False)
    return f

def dice_loss_fg(y_true, y_pred):
    f = 1 - dice_coef(y_true, y_pred, squared_denominator = False, ignore_zero_label = True)
    return f

def modified_dice_loss(y_true, y_pred):
    f = 1 - dice_coef(y_true, y_pred, squared_denominator = True, ignore_zero_label = False)
    return f

def modified_dice_loss_fg(y_true, y_pred):
    f = 1 - dice_coef(y_true, y_pred, squared_denominator = True, ignore_zero_label = True)
    return f
