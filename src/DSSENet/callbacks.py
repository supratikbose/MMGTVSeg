from tensorflow.keras import callbacks
import numpy as np
import time
from pipeline import metrics #from vmsseg import metrics
from pipeline import volume #from vmsseg import volume

class evaluate_validation_data_callback(callbacks.Callback):

    def __init__(self, test_data, test_labels, image_size_cropped, resolution=[1,1,1], save_to_nii=True):
        self.test_data = test_data
        self.test_labels = test_labels
        self.image_size_cropped = image_size_cropped
        self.resolution = resolution
        self.save_to_nii = save_to_nii

    def on_epoch_end(self, epoch, logs={}):
        X = self.test_data
        X = np.expand_dims(X, axis=0)

        # crop input volume
        X = volume.centered_crop(X, self.image_size_cropped)

        # # generic normalization
        # X = X.astype(np.float32)
        # X = X - X.mean()
        # X_std = X.std()
        # if X_std > 0:              # avoid division by zero           
        #     X = X / X_std

        # # robust normalization (handles intensity outliers better)
        # X = X.astype(np.float32)            
        # p_low = np.percentile(X, 5)        
        # p_high = np.percentile(X, 95)
        # p_mid = (p_high - p_low) / 2
        # p_range = p_high - p_low
        # if p_range > 0:
        #     X = X - p_mid
        #     X = X / p_range

        # # improved generic normalization (only works when padding value is 0, not -1000)
        # X = X.astype(np.float32)                        
        # X_flat = X.ravel()
        # ti = (X_flat != 0)
        # X_nonzero = X_flat[ti]
        # X[X == 0] = X_nonzero.mean()
        # X = X - X_nonzero.mean()
        # X_std = X_nonzero.std()
        # if X_std > 0:      # avoid division by zero     
        #     X = X / X_std  

        # # simple normalization (works for AAPM T2w MRI)            
        # X = X.astype(np.float32)
        # X = (X - 180.0) / 360.0
     
        # simple CT normalization            
        X = X.astype(np.float32)
        X = X  / 1000.0
        # run model inference
        t = time.time()
        preds = self.model.predict(X, batch_size=1)
        print('\nInference time: ', time.time() - t)

        #ipreds=np.argmax(preds, axis=-1).astype('int16')
        
        fg_ti = ipred >= 0.5
        ipreds = np.zeros(preds.shape,dtype=np.int16)
        ipreds[fg_ti] = 1

        ipreds=np.squeeze(ipreds, axis=0)
        Xave = np.squeeze(X, axis=0)
        y = self.test_labels
        y = np.squeeze(y, axis=-1)   
        y = volume.centered_crop(y, self.image_size_cropped)

        # save results to NIFTI
        if (self.save_to_nii == True):
            print('\nWriting test segmentation results to disk')
            Xave = np.squeeze(Xave, axis=3)
            volume.save_to_nii(Xave.astype('float32'), 'X.nii.gz', self.resolution)
            volume.save_to_nii(ipreds.astype('uint8'), 'y.nii.gz', self.resolution)
            volume.save_to_nii(y.astype('uint8'), 'y_gt.nii.gz', self.resolution)    
        
        # compute surface distance metrics
        [lbls, msd, rms, hd] = metrics.surface_distance_multi_label(ipreds, y, self.resolution) 
        print('Labels: ', lbls)
        print('Mean surface distance (mm): ', msd)
        print('RMS surface distance (mm): ', rms)
        print('Hausdorff surface distance (mm): ', hd)        
        print('\n')
