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
from preprocess import data_util

from datetime import datetime

def train(model_file, data_file):
    # load model and data parameters from JSON config files
    with open(model_file) as f:
        model_pars = json.load(f)
        f.close()
    with open(data_file) as f:
        data_pars = json.load(f)
        f.close()

    # validate and initialize config parameters
    if 'loss_func' not in model_pars:
        model_pars['loss_func'] = loss.dice_loss
    elif model_pars['loss_func'].lower() == 'dice_loss':
        model_pars['loss_func'] = loss.dice_loss
    elif model_pars['loss_func'].lower() == 'modified_dice_loss':
        model_pars['loss_func'] = loss.modified_dice_loss  
    elif model_pars['loss_func'].lower() == 'dice_loss_fg':
        model_pars['loss_func'] = loss.dice_loss_fg
    elif model_pars['loss_func'].lower() == 'modified_dice_loss_fg':
        model_pars['loss_func'] = loss.modified_dice_loss_fg  
    if 'acc_func' not in model_pars:
        model_pars['acc_func'] = metrics.categorical_accuracy
    elif model_pars['acc_func'].lower() == 'categorical_accuracy':
        model_pars['acc_func'] = metrics.categorical_accuracy
    if 'labels_to_train' not in model_pars:
        model_pars['labels_to_train'] = [1]
    if 'asymmetric' not in model_pars:
        model_pars['asymmetric'] = True
    if 'group_normalization' not in model_pars:
        model_pars['group_normalization'] = False
    if 'activation_type' not in model_pars:
        model_pars['activation_type'] = 'relu'
    if 'final_activation_type' not in model_pars:
        model_pars['final_activation_type'] = 'softmax'
    if 'AMP' not in model_pars:
        model_pars['AMP'] = False
    if 'XLA' not in model_pars:
        model_pars['XLA'] = False

    # #Original
    # # determine number of available GPUs and CPUs
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # num_gpus = len(gpus)    
    # print('Number of GPUs used for training: ', num_gpus)
    # # prevent tensorflow from allocating all available GPU memory
    # if (num_gpus > 0):
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)

    #Limit GPU use to a single GPU as I am not sure whether that messed up tensorboard
    # I earlier saw an error message about multiGPU and tensorboard
    gpus = tf.config.experimental.list_physical_devices('GPU')
    num_gpus = len(gpus)    
    print('Number of GPUs AVAILABLE for training: ', num_gpus)
    if gpus:
        print("Restricting TensorFlow to only use the first GPU.")
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    num_cpus = min(os.cpu_count(), 24)   # don't use more than 16 CPU threads
    print('Number of CPUs used for training: ', num_cpus)



    if (model_pars['AMP']):
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
        os.environ['CUDNN_TENSOROP_MATH'] = '1'
        print('Using Automatic Mixed Precision (AMP) Arithmentic...')

    if (model_pars['XLA']):
        tf.config.optimizer.set_jit(True)
    
    # count number of training and test cases
    train_fnames = sorted(glob.glob(data_pars['path'] + '/train/' + "img*.nii.gz"))
    test_fnames = sorted(glob.glob(data_pars['path'] + '/test/' + "img*.nii.gz"))
    num_train_cases = len(train_fnames)
    num_test_cases = len(test_fnames)

    print('Number of train cases: ', num_train_cases)
    print('Number of test cases: ', num_test_cases)
    print("labels to train: ", model_pars['labels_to_train'])
    
    train_sequence = augment.images_and_labels_sequence(data_path = data_pars['path'] + '/train/', 
                                                        batch_size = model_pars['batch_size'], 
                                                        out_size = model_pars['image_cropped_size'], 
                                                        translate_random = data_pars['translate_random'],  
                                                        rotate_random = data_pars['rotate_random'],       
                                                        scale_random = data_pars['scale_random'],         
                                                        change_intensity = data_pars['change_intensity'],
                                                        labels_to_train = model_pars['labels_to_train'],
                                                        lr_flip = data_pars['lr_flip'],                                                     
                                                        label_symmetry_map = data_pars['label_symmetry_map'] )

    test_sequence = augment.images_and_labels_sequence( data_path = data_pars['path'] + '/test/',
                                                        batch_size = min(model_pars['batch_size'], num_test_cases), 
                                                        out_size = model_pars['image_cropped_size'],
                                                        translate_random = 0.0,    
                                                        rotate_random = 0.0,       
                                                        scale_random = 0.0,         
                                                        change_intensity = 0.0 )

    # # distribution strategy (multi-GPU or TPU training), disabled because model.fit 
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    # with strategy.scope():
    
    # load existing or create new model
    if os.path.exists(model_pars['fname']):
        #from tensorflow.keras.models import load_model        
        #model = load_model(model_pars['fname'], custom_objects={'dice_loss_fg': loss.dice_loss_fg, 'modified_dice_loss': loss.modified_dice_loss})
        model = tf.keras.models.load_model(model_pars['fname'], custom_objects={'dice_loss_fg': loss.dice_loss_fg, 'modified_dice_loss': loss.modified_dice_loss})
        print('Loaded model: ' + model_pars['fname'])
    else:
        model = vmsnet.vmsnet(model_pars['image_cropped_size'] + [1,], 
            nb_classes = len(model_pars['labels_to_train']) + 1, 
            init_filters = model_pars['init_filters'], 
            filters = model_pars['num_filters'], 
            nb_layers_per_block = model_pars['num_layers_per_block'], 
            dropout_prob = model_pars['dropout_rate'],
            kernel_size = model_pars['kernel_size'],
            asymmetric = model_pars['asymmetric'], 
            group_normalization = model_pars['group_normalization'], 
            activation_type = model_pars['activation_type'], 
            final_activation_type = model_pars['final_activation_type'])                              

        optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)        
        if model_pars['AMP']:
            optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

        model.compile(optimizer = optimizer, loss = model_pars['loss_func'], metrics = [model_pars['acc_func']])
        model.summary(line_length=140)
        
    # TODO: clean up the evaluation callback
    #tb_logdir = './logs/' + os.path.basename(model_pars['fname'])
    tb_logdir = './logs/' + os.path.basename(model_pars['fname']) + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    train_callbacks = [tf.keras.callbacks.TensorBoard(log_dir = tb_logdir),
                tf.keras.callbacks.ModelCheckpoint(model_pars['fname'], monitor = "loss", save_best_only = True, mode='min')]
#                callbacks.evaluate_validation_data_callback(test_images[0,:], test_labels[0,:], image_size_cropped=model_pars['image_cropped_size'], 
#                    resolution = data_pars['spacing'], save_to_nii=True) ]

    model.fit_generator(train_sequence,
                        steps_per_epoch = num_train_cases // model_pars['batch_size'],
                        max_queue_size = 40,
                        epochs = model_pars['num_training_epochs'],
                        validation_data = test_sequence,
                        validation_steps = 1,
                        callbacks = train_callbacks,
                        use_multiprocessing = True,
                        workers = num_cpus, 
                        shuffle = True)

    model.save(model_pars['fname'] + '_final')