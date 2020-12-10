# Implementation of DSSE-VNet-AC
# Ref 1: P. Liu et al.: Encoder-Decoder Neural Network With 3D SE and Deep Supervision
# Ref 2: Xu Chen et.al: Learning Active Contour Models for Medical Image Segmentation
# Ref 3: Squeeze and Excitation Networks https://arxiv.org/abs/1709.01507

#Note 1: https://stackoverflow.com/questions/47538391/keras-batchnormalization-axis-clarification
# A note about difference in meaning of axis in np.mean versus in BatchNormalization. When we take the mean along an axis, we collapse
# that dimension and preserve all other dimensions. In your example data.mean(axis=0) collapses the 0-axis, which is the vertical 
# dimension of data. When we compute a BatchNormalization along an axis, we preserve the dimensions of the array, and we normalize 
# with respect to the mean and standard deviation over every other axis. So in your 2D example BatchNormalization with axis=1 is subtracting 
# the mean for axis=0, just as you expect. This is why bn.moving_mean has shape (4,).
# In our case we we should do the batch_normalization along (i.e., preserving) the channel axis


import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, Activation, Add, Concatenate, BatchNormalization, ELU, SpatialDropout3D, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling3D, Reshape, Dense, Multiply,  Permute
import tensorflow_addons as tfa
from tensorflow.keras import regularizers



def getBNAxis(data_format='channels_last'):
    if 'channels_last' == data_format:
        bnAxis = -1 #last axis is channel : (batch, slice, row, col, channel)
    else:
        bnAxis = -4 #4th-last axis is channel : (batch, channel, slice, row, col)
    return bnAxis

#D x H x W x ? ==> D x H x W x f and  strides = (1,1,1)
def ConvBnElu(x, filters, kernel_size = (3,3,3), strides = (1,1,1), kernel_initializer = 'he_normal', padding = 'same', use_bias = False, data_format='channels_last'):
    x = Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, kernel_initializer = kernel_initializer, padding = padding, data_format=data_format, use_bias = use_bias)(x)
    #See note 1 
    x = BatchNormalization(axis=getBNAxis(data_format))(x)
    x=  ELU()(x)
    return x

#D x H x W x f_in ==> D/2 x H/2 x W/2 x 2f_in 
def DownConvBnElu(x, in_filters,  kernel_initializer = 'he_normal', padding = 'same', use_bias = False, data_format='channels_last'):
    x = Conv3D(filters = 2 * in_filters, kernel_size = (2,2,2), strides = (2,2,2), kernel_initializer = kernel_initializer, padding = padding, data_format=data_format, use_bias = use_bias)(x)
    #See note 1 
    x = BatchNormalization(axis=getBNAxis(data_format))(x)
    x=  ELU()(x)
    return x

#D x H x W x f_in ==> 2D x 2H x 2W x f_in/2
def UpConvBnElu(x, in_filters,  kernel_initializer = 'he_normal', padding = 'same', use_bias = False, data_format='channels_last'):
    x = Conv3DTranspose(filters = in_filters // 2, kernel_size = (2,2,2), strides = (2,2,2), kernel_initializer = kernel_initializer, padding = padding, data_format=data_format, use_bias = use_bias)(x)
    #See note 1 
    x = BatchNormalization(axis=getBNAxis(data_format))(x)
    x=  ELU()(x)
    return x

#This is essentially the orange block in upper layers of  Figure 1 of Ref 1:
# Note  2nd ConvBnElu has kernel size = 3x3x3
#D x H x W x f ==> D x H x W x f  
def UpperLayerSingleResidualBlock(x,data_format='channels_last'):
    filters = x._keras_shape[getBNAxis(data_format)]
    shortcut = x     
    x = ConvBnElu(x, filters=filters, kernel_size = (5,5,5), strides = (1,1,1), data_format=data_format)
    x = ConvBnElu(x, filters=filters, kernel_size = (3,3,3), strides = (1,1,1), data_format=data_format)
    x = ELU()(Add()([x,shortcut]))
    return x

#This is essentially the concatenation of orange (single Residual) block and pink (bottom double residual) block in Figure 1 of Ref 1: 
#D x H x W x f ==> D x H x W x f 
def SingleAndDoubleResidualBlock(x,data_format='channels_last'):
    filters = x._keras_shape[getBNAxis(data_format)]
    #First single Res block
    shortcut_1 = x
    #Residualblock1:Conv1
    x = ConvBnElu(x, filters=filters, kernel_size = (5,5,5), strides = (1,1,1), data_format=data_format)
    #Residualblock1:Conv2
    x = ConvBnElu(x, filters=filters, kernel_size = (5,5,5), strides = (1,1,1), data_format=data_format)
    x = ELU()(Add()([x,shortcut_1]))
    #Then bottom double Res block
    shortuct_2 = x
    #BottResidualBlock:Conv11
    x = ConvBnElu(x, filters=filters//4, kernel_size = (1,1,1), strides = (1,1,1), data_format=data_format)
    #BottResidualBlock:Conv12
    x = ConvBnElu(x, filters=filters//4, kernel_size = (3,3,3), strides = (1,1,1), data_format=data_format)
    #BottResidualBlock:Conv13
    x = ConvBnElu(x, filters=filters, kernel_size = (3,3,3), strides = (1,1,1), data_format=data_format)
    #BottResidualBlock:Conv21
    x = ConvBnElu(x, filters=filters//4, kernel_size = (1,1,1), strides = (1,1,1), data_format=data_format)
    #BottResidualBlock:Conv22
    x = ConvBnElu(x, filters=filters//4, kernel_size = (3,3,3), strides = (1,1,1), data_format=data_format)
    #BottResidualBlock:Conv23
    x = ConvBnElu(x, filters=filters, kernel_size = (3,3,3), strides = (1,1,1), data_format=data_format)
    x = ELU()(Add()([x, shortuct_2]))
    return x

def DeepSupervision(x, filters=2, upSamplingRatio=1,data_format='channels_last'):
    x = Conv3D(filters = filters, kernel_size = (1,1,1), strides = (1,1,1), kernel_initializer = 'he_normal', padding = 'same', data_format=data_format, use_bias = False)(x)
    x = UpSampling3D(size=(upSamplingRatio, upSamplingRatio, upSamplingRatio), data_format=data_format)(x)
    return x

# https://github.com/titu1994/keras-squeeze-excite-network/blob/master/se.py : Used
def Squeeze_Excite_block(x, ratio=16, data_format='channels_last'):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        ratio: number of output filters
    Returns: a keras tensor
    '''
    #init = input
    #channel_axis = getBNAxis(data_format) #1 if K.image_data_format() == "channels_first" else -1
    filters =  x._keras_shape[getBNAxis(data_format)] #init._keras_shape[channel_axis] ##x.get_shape().as_list()[getBNAxis(data_format)] 
    # Note se_shape is the target shape with out considering the batch_size rank
    # In the 2D case it was : se_shape = (1, 1, filters)  
    se_shape = (1, 1, 1, filters) 

    se = GlobalAveragePooling3D(data_format=data_format)(x) #In the 2D case it was : GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if 'channels_first' == data_format: #K.image_data_format() == 'channels_first':
        # Note : dims: Tuple of integers. Permutation pattern, does not include the  samples (i.e., batch) dimension. Indexing starts at 1.
        # In the 2D case it was : se = Permute((3, 1, 2))(se)
        se = Permute((4, 1, 2, 3))(se)

    x = Multiply()([x, se])
    return x


def DSSE_VNet(input_shape, dropout_prob = 0.25, data_format='channels_last'):

    ########## Encode path ##########       (using the terminology from Ref1)
    #InTr  128 x 128 x 128 x 2 ==> 128 x 128 x 128 x 16
    #2 channel 128 x 128 x 128 
    img_input = Input(shape = input_shape) # (Nc, D, H, W) if channels_first else  (Batch, D, H, W, Nc)
    # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input
    # channel
    input_channels =  input_shape[-1] if 'channels_last' == data_format else input_shape[-4] #config["inputChannels"]
    tile_tensor    =  [1,1,1,1,16]    if 'channels_last' == data_format else [1,16,1,1,1]    #config["inputChannels"]
    if 1 == input_channels:
        sixteen_channelInput = tf.tile(img_input,tile_tensor)
    else:
        sixteen_channelInput = ConvBnElu(img_input, filters=16, kernel_size = (5,5,5), strides = (1,1,1),  data_format=data_format)
    #In Table 1 of Ref 1, stride of 1x1x5 was mentioned for conv1, but we are sicking to stride 1x1x1; And here conv1 step includes add + elu
    _InTr =  ELU()(Add()([ConvBnElu(sixteen_channelInput, filters=16, kernel_size = (5,5,5), strides = (1,1,1),  data_format=data_format), sixteen_channelInput]))
    _InTrDropout = SpatialDropout3D(rate=dropout_prob, data_format='channels_last')(_InTr)

    #DownTr32  128 x 128 x 128 x 16 ==> 64 x 64 x 64 x 32  
    _DownTr32  =  DownConvBnElu(x=_InTr, in_filters=16,  data_format=data_format)  
    _DownTr32  =  UpperLayerSingleResidualBlock(x=_DownTr32, data_format=data_format)
    _DownTr32  =  UpperLayerSingleResidualBlock(x=_DownTr32, data_format=data_format)
    _DownTr32  =  Squeeze_Excite_block(x=_DownTr32, ratio=8, data_format=data_format)
    _DownTr32Dropout = SpatialDropout3D(rate=dropout_prob, data_format='channels_last')(_DownTr32)

    #DownTr64   64 x 64 x 64 x 32 ==> 32 x 32 x 32 x 64  
    _DownTr64  =  DownConvBnElu(x=_DownTr32, in_filters=32,  data_format=data_format)  
    _DownTr64  =  SingleAndDoubleResidualBlock(x=_DownTr64,  data_format=data_format)
    _DownTr64  =  Squeeze_Excite_block(x=_DownTr64, ratio=8, data_format=data_format)
    _DownTr64Dropout = SpatialDropout3D(rate=dropout_prob, data_format='channels_last')(_DownTr64)

     #DownTr128   32 x 32 x 32 x 64 ==> 16 x 16 x 16 x 128  
    _DownTr128  =  DownConvBnElu(x=_DownTr64, in_filters=64,  data_format=data_format)  
    _DownTr128  =  SingleAndDoubleResidualBlock(x=_DownTr128,  data_format=data_format)
    _DownTr128  =  Squeeze_Excite_block(x=_DownTr128, ratio=8, data_format=data_format)
    _DownTr128Dropout = SpatialDropout3D(rate=dropout_prob, data_format='channels_last')(_DownTr128)

    #DownTr256   16 x 16 x 16 x 128 ==> 8 x 8 x 8 x 256
    _DownTr256  =  DownConvBnElu(x=_DownTr128, in_filters=128,  data_format=data_format)  
    _DownTr256  =  SingleAndDoubleResidualBlock(x=_DownTr256, data_format=data_format)
    _DownTr256  =  Squeeze_Excite_block(x=_DownTr256, ratio=8, data_format=data_format)       


    ########## Dncode path ##########
    #UpTr256    8 x 8 x 8 x 256 ==> 16 x 16 x 16 x 128 => 16 x 16 x 16 x 256 (due to concatenation)
    _UpTr256  = UpConvBnElu(_DownTr256, in_filters=256, data_format=data_format)
    _UpTr256  = Concatenate()([_UpTr256,_DownTr128Dropout], axis = getBNAxis(data_format))
    _UpTr256  =  SingleAndDoubleResidualBlock(x=_UpTr256, data_format=data_format)
    _UpTr256  =  Squeeze_Excite_block(x=_UpTr256, ratio=8, data_format=data_format)
    #Also Dsv4 16 x 16 x 16 x 256 ==> 128 x 128 x 128 x 4
    _Dsv4 = DeepSupervision(_UpTr256, filters=4, upSamplingRatio=128//16, data_format=data_format)


    #UpTr128    16 x 16 x 16 x 256 ==> 32 x 32 x 32 x 64 => 32 x 32 x 32 x 128 (due to concatenation)
    _UpTr128  = UpConvBnElu(_UpTr256, in_filters=128, data_format=data_format)
    _UpTr128  = Concatenate()([_UpTr128,_DownTr64Dropout], axis = getBNAxis(data_format))
    _UpTr128  =  SingleAndDoubleResidualBlock(x=_UpTr128, data_format=data_format)
    _UpTr128  =  Squeeze_Excite_block(x=_UpTr128, ratio=8, data_format=data_format)
    #Also Dsv3 32 x 32 x 32 x 128 ==> 128 x 128 x 128 x 4
    _Dsv3 = DeepSupervision(_UpTr128, filters=4, upSamplingRatio=128//32, data_format=data_format)

    #UpTr64    32 x 32 x 32 x 128 ==> 64 x 64 x 64 x 32 => 64 x 64 x 64 x 64 (due to concatenation)
    _UpTr64  = UpConvBnElu(_UpTr128, in_filters=64, data_format=data_format)
    _UpTr64  = Concatenate()([_UpTr64,_DownTr32Dropout], axis = getBNAxis(data_format))
    _UpTr64  =  SingleAndDoubleResidualBlock(x=_UpTr64, data_format=data_format)
    _UpTr64  =  Squeeze_Excite_block(x=_UpTr64, ratio=8, data_format=data_format)
    #Also Dsv2 64 x 64 x 64 x 64 ==> 128 x 128 x 128 x 4
    _Dsv2 = DeepSupervision(_UpTr64, filters=4, upSamplingRatio=128//64, data_format=data_format)

    #UpTr32    64 x 64 x 64 x 64 ==> 128 x 128 x 128 x 16 => 128 x 128 x 128 x 32 (due to concatenation)
    _UpTr32  = UpConvBnElu(_UpTr128, in_filters=32, data_format=data_format)
    _UpTr32  = Concatenate()([_UpTr32,_InTrDropout], axis = getBNAxis(data_format))
    _UpTr32  =   UpperLayerSingleResidualBlock(x=_UpTr32, data_format=data_format)
    _UpTr32  =   UpperLayerSingleResidualBlock(x=_UpTr32, data_format=data_format)
    _UpTr32  =  Squeeze_Excite_block(x=_UpTr32, ratio=8, data_format=data_format)
    #Also Dsv1 128 x 128 x 128 x 32 ==> 128 x 128 x 128 x 4
    _Dsv1 = DeepSupervision(_UpTr32, filters=4, upSamplingRatio=128//128, data_format=data_format)

    #Final concatenation and convolution
    #128 x 128 x 128 x 4 ==> 128 x 128 x 128 x 16
    _DsvConcat = Concatenate()([_Dsv1, _Dsv2, _Dsv3, _Dsv4], axis = getBNAxis(data_format))
    #128 x 128 x 128 x 1 ==> 128 x 128 x 128 x 1
    _Final = Conv3D(filters = 1, kernel_size = (1,1,1), strides = (1,1,1), kernel_initializer = 'he_normal', padding = 'same', data_format=data_format, use_bias = False)(_DsvConcat)

    # model instantiation
    model = Model(img_input, _Final)
    return model


def vmsnet(input_shape, nb_classes, init_filters = 2, filters = 4, nb_layers_per_block = [1,2,2], dropout_prob = 0, 
        kernel_size = 3, asymmetric = True, group_normalization = True, activation_type = 'relu', final_activation_type = 'softmax'):

    # validate parameters
    nb_layers = list(nb_layers_per_block)
    if (asymmetric == True):
        revNb_layers = list([1]*len(nb_layers))
    else:
        revNb_layers = list(reversed(nb_layers))

    if (group_normalization == True):
        group_norm_groups = init_filters
    else:
        group_norm_groups = 0

    # encoder section
    img_input = Input(shape = input_shape)
    x = convolution_block(img_input, filters = init_filters, kernel_size = kernel_size, dropout_prob = dropout_prob) 
    x, skips = down_path(x, nb_layers, filters = filters, dropout_prob = dropout_prob, kernel_size = kernel_size, group_norm_groups = group_norm_groups, activation_type=activation_type, concatenate = True)
    revSkips = list(reversed(skips))

    # decoder section
    x = up_path(x, skips = revSkips, nb_layers = revNb_layers, dropout_prob = dropout_prob, kernel_size = kernel_size, group_norm_groups = group_norm_groups, activation_type=activation_type, concatenate = True)
    if (final_activation_type.lower() == 'softmax'):
        x = convolution_block(x, filters = nb_classes, kernel_size = 1, dropout_prob = 0) 
        x = Activation('softmax')(x)
    elif (final_activation_type.lower() == 'sigmoid'):
        x = convolution_block(x, filters = 1, kernel_size = 1, dropout_prob = 0) 
        x = Activation('sigmoid')(x)
    else:
        print('Incorrect final_activation_type!')
        exit()

    # model instantiation
    model = Model(img_input, x)

    return model