import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import src


from src.DSSENet import loss #from vmsseg import loss
from src.DSSENet import augment #from vmsseg import augment
from src.DSSENet import callbacks #from vmsseg import callbacks
from src.DSSENet import DSSE_VNet
from src.preprocess import data_util
from datetime import datetime

def train(trainConfigFilePath):
    # load trainInputParams  from JSON config files
    with open(trainConfigFilePath) as f:
        trainInputParams = json.load(f)
        f.close()


    # validate and initialize config parameters
    if 'loss_func' not in trainInputParams:
        trainInputParams['loss_func'] = loss.dice_loss
    elif trainInputParams['loss_func'].lower() == 'dice_loss':
        trainInputParams['loss_func'] = loss.dice_loss
    elif trainInputParams['loss_func'].lower() == 'modified_dice_loss':
        trainInputParams['loss_func'] = loss.modified_dice_loss  
    elif trainInputParams['loss_func'].lower() == 'dice_loss_fg':
        trainInputParams['loss_func'] = loss.dice_loss_fg
    elif trainInputParams['loss_func'].lower() == 'modified_dice_loss_fg':
        trainInputParams['loss_func'] = loss.modified_dice_loss_fg  

    if 'acc_func' not in trainInputParams:
        trainInputParams['acc_func'] = metrics.categorical_accuracy
    elif trainInputParams['acc_func'].lower() == 'categorical_accuracy':
        trainInputParams['acc_func'] = metrics.categorical_accuracy

    if 'labels_to_train' not in trainInputParams:
        trainInputParams['labels_to_train'] = [1]
    if 'asymmetric' not in trainInputParams:
        trainInputParams['asymmetric'] = True
    if 'group_normalization' not in trainInputParams:
        trainInputParams['group_normalization'] = False
    if 'activation_type' not in trainInputParams:
        trainInputParams['activation_type'] = 'relu'
    if 'final_activation_type' not in trainInputParams:
        trainInputParams['final_activation_type'] = 'softmax'
    if 'AMP' not in trainInputParams:
        trainInputParams['AMP'] = False
    if 'XLA' not in trainInputParams:
        trainInputParams['XLA'] = False

    #Original
    # determine number of available GPUs and CPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    num_gpus = len(gpus)    
    print('Number of GPUs used for training: ', num_gpus)
    # prevent tensorflow from allocating all available GPU memory
    if (num_gpus > 0):
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # #Limit GPU use to a single GPU as I am not sure whether that messed up tensorboard
    # # I earlier saw an error message about multiGPU and tensorboard
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # num_gpus = len(gpus)    
    # print('Number of GPUs AVAILABLE for training: ', num_gpus)
    # if gpus:
    #     print("Restricting TensorFlow to only use the first GPU.")
    #     try:
    #         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #     except RuntimeError as e:
    #         # Visible devices must be set before GPUs have been initialized
    #         print(e)

    num_cpus = min(os.cpu_count(), 24)   # don't use more than 16 CPU threads
    print('Number of CPUs used for training: ', num_cpus)

    if (trainInputParams['AMP']):
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
        os.environ['CUDNN_TENSOROP_MATH'] = '1'
        print('Using Automatic Mixed Precision (AMP) Arithmentic...')

    if (trainInputParams['XLA']):
        tf.config.optimizer.set_jit(True)
    

    train_sequence = augment.DSSENet_Generator(trainConfigFilePath = trainConfigFilePath, 
                                                data_format=trainInputParams['data_format'],
                                                useDataAugmentationDuringTraining = True,
                                                batch_size = 2,
                                                cvFoldIndex = 0, #Can be between 0 to 4
                                                isValidationFlag = False
                                                )

    test_sequence = augment.DSSENet_Generator(trainConfigFilePath = trainConfigFilePath, 
                                                data_format=trainInputParams['data_format'],
                                                useDataAugmentationDuringTraining = False, #No augmentation during validation
                                                batch_size = 2,
                                                cvFoldIndex = 0, #Can be between 0 to 4
                                                isValidationFlag = True
                                                )
    
    # count number of training and test cases
    num_train_cases = train_sequence.__len__()
    num_test_cases = test_sequence.__len__()

    print('Number of train cases: ', num_train_cases)
    print('Number of test cases: ', num_test_cases)
    print("labels to train: ", trainInputParams['labels_to_train'])
    
    sampleCube_dim = [trainInputParams["sampleInput_Depth"], trainInputParams["patientVol_Height"], trainInputParams["patientVol_width"]]
    if 'channels_last' == trainInputParams['data_format']:
        input_shape = tuple(sampleCube_dim+[2]) # 2 channel CT and PET
        output_shape = tuple(sampleCube_dim+[1]) # 1 channel output
    else: # 'channels_first'
        input_shape = tuple([2] + sampleCube_dim) # 2 channel CT and PET
        output_shape = tuple([1] + sampleCube_dim) # 1 channel output

    # # distribution strategy (multi-GPU or TPU training), disabled because model.fit 
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    # with strategy.scope():
    
    # load existing or create new model
    if os.path.exists(trainInputParams["lastSavedModel"]):
        #from tensorflow.keras.models import load_model        
        #model = load_model(trainInputParams['fname'], custom_objects={'dice_loss_fg': loss.dice_loss_fg, 'modified_dice_loss': loss.modified_dice_loss})
        model = tf.keras.models.load_model(trainInputParams['fname'], custom_objects={'dice_loss_fg': loss.dice_loss_fg, 'modified_dice_loss': loss.modified_dice_loss})
        print('Loaded model: ' + trainInputParams["lastSavedModel"])
    else:
        model = DSSE_VNet.DSSE_VNet(input_shape=input_shape, dropout_prob = 0.25, data_format=trainInputParams['data_format'])                              

        optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)        
        if trainInputParams['AMP']:
            optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

        model.compile(optimizer = optimizer, loss = trainInputParams['loss_func'], metrics = [trainInputParams['acc_func']])
        model.summary(line_length=140)
        
    # TODO: clean up the evaluation callback
    #tb_logdir = './logs/' + os.path.basename(trainInputParams['fname'])
    tb_logdir = './logs/' + os.path.basename(trainInputParams["lastSavedModel"]) + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    train_callbacks = [tf.keras.callbacks.TensorBoard(log_dir = tb_logdir),
                tf.keras.callbacks.ModelCheckpoint(trainInputParams["lastSavedModel"], monitor = "loss", save_best_only = True, mode='min')]
#                callbacks.evaluate_validation_data_callback(test_images[0,:], test_labels[0,:], image_size_cropped=trainInputParams['image_cropped_size'], 
#                    resolution = trainInputParams['spacing'], save_to_nii=True) ]

    model.fit_generator(train_sequence,
                        steps_per_epoch = num_train_cases // trainInputParams['batch_size'],
                        max_queue_size = 40,
                        epochs = trainInputParams['num_training_epochs'],
                        validation_data = test_sequence,
                        validation_steps = 1,
                        callbacks = train_callbacks,
                        use_multiprocessing = True,
                        workers = num_cpus, 
                        shuffle = True)

    model.save(trainInputParams['fname'] + '_final')