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
    
    # count number of training and test cases
    train_fnames = sorted(glob.glob(trainInputParams['path'] + '/train/' + "img*.nii.gz"))
    test_fnames = sorted(glob.glob(trainInputParams['path'] + '/test/' + "img*.nii.gz"))
    num_train_cases = len(train_fnames)
    num_test_cases = len(test_fnames)

    print('Number of train cases: ', num_train_cases)
    print('Number of test cases: ', num_test_cases)
    print("labels to train: ", trainInputParams['labels_to_train'])
    
    train_sequence = augment.images_and_labels_sequence(data_path = trainInputParams['path'] + '/train/', 
                                                        batch_size = trainInputParams['batch_size'], 
                                                        out_size = trainInputParams['image_cropped_size'], 
                                                        translate_random = trainInputParams['translate_random'],  
                                                        rotate_random = trainInputParams['rotate_random'],       
                                                        scale_random = trainInputParams['scale_random'],         
                                                        change_intensity = trainInputParams['change_intensity'],
                                                        labels_to_train = trainInputParams['labels_to_train'],
                                                        lr_flip = trainInputParams['lr_flip'],                                                     
                                                        label_symmetry_map = trainInputParams['label_symmetry_map'] )

    test_sequence = augment.images_and_labels_sequence( data_path = trainInputParams['path'] + '/test/',
                                                        batch_size = min(trainInputParams['batch_size'], num_test_cases), 
                                                        out_size = trainInputParams['image_cropped_size'],
                                                        translate_random = 0.0,    
                                                        rotate_random = 0.0,       
                                                        scale_random = 0.0,         
                                                        change_intensity = 0.0 )

    # # distribution strategy (multi-GPU or TPU training), disabled because model.fit 
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    # with strategy.scope():
    
    # load existing or create new model
    if os.path.exists(trainInputParams['fname']):
        #from tensorflow.keras.models import load_model        
        #model = load_model(trainInputParams['fname'], custom_objects={'dice_loss_fg': loss.dice_loss_fg, 'modified_dice_loss': loss.modified_dice_loss})
        model = tf.keras.models.load_model(trainInputParams['fname'], custom_objects={'dice_loss_fg': loss.dice_loss_fg, 'modified_dice_loss': loss.modified_dice_loss})
        print('Loaded model: ' + trainInputParams['fname'])
    else:
        model = vmsnet.vmsnet(trainInputParams['image_cropped_size'] + [1,], 
            nb_classes = len(trainInputParams['labels_to_train']) + 1, 
            init_filters = trainInputParams['init_filters'], 
            filters = trainInputParams['num_filters'], 
            nb_layers_per_block = trainInputParams['num_layers_per_block'], 
            dropout_prob = trainInputParams['dropout_rate'],
            kernel_size = trainInputParams['kernel_size'],
            asymmetric = trainInputParams['asymmetric'], 
            group_normalization = trainInputParams['group_normalization'], 
            activation_type = trainInputParams['activation_type'], 
            final_activation_type = trainInputParams['final_activation_type'])                              

        optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)        
        if trainInputParams['AMP']:
            optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

        model.compile(optimizer = optimizer, loss = trainInputParams['loss_func'], metrics = [trainInputParams['acc_func']])
        model.summary(line_length=140)
        
    # TODO: clean up the evaluation callback
    #tb_logdir = './logs/' + os.path.basename(trainInputParams['fname'])
    tb_logdir = './logs/' + os.path.basename(trainInputParams['fname']) + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    train_callbacks = [tf.keras.callbacks.TensorBoard(log_dir = tb_logdir),
                tf.keras.callbacks.ModelCheckpoint(trainInputParams['fname'], monitor = "loss", save_best_only = True, mode='min')]
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