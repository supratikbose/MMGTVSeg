# Author: Oskar Radermecker
# Date: 09/20/2019
# Company: Varian Medical Systems

import SimpleITK
import numpy as np
import nibabel as nib
from skimage.measure import label

def centered_crop(vol, crop_size):
    if vol.ndim == 5:
        x = (vol.shape[1] - crop_size[0]) // 2  
        y = (vol.shape[2] - crop_size[1]) // 2
        z = (vol.shape[3] - crop_size[2]) // 2
        ix = slice(x, x + crop_size[0])
        iy = slice(y, y + crop_size[1])
        iz = slice(z, z + crop_size[2])
        vol = vol[:, ix,  iy, iz, :]
    elif vol.ndim == 4:
        x = (vol.shape[0] - crop_size[0]) // 2  
        y = (vol.shape[1] - crop_size[1]) // 2
        z = (vol.shape[2] - crop_size[2]) // 2
        ix = slice(x, x + crop_size[0])
        iy = slice(y, y + crop_size[1])
        iz = slice(z, z + crop_size[2])
        vol = vol[ix, iy, iz, :]
    elif vol.ndim == 3:
        x = (vol.shape[0] - crop_size[0]) // 2  
        y = (vol.shape[1] - crop_size[1]) // 2
        z = (vol.shape[2] - crop_size[2]) // 2
        ix = slice(x, x + crop_size[0])
        iy = slice(y, y + crop_size[1])
        iz = slice(z, z + crop_size[2])
        vol = vol[ix, iy, iz]
    elif vol.ndim == 2:
        x = (vol.shape[0] - crop_size[0]) // 2  
        y = (vol.shape[1] - crop_size[1]) // 2
        ix = slice(x, x + crop_size[0])
        iy = slice(y, y + crop_size[1])
        vol = vol[ix, iy]
    else:
        vol = 0    
    return vol

def random_crop(vol, crop_size):
    if vol.ndim == 5:
        x = np.random.randint(0, vol.shape[1] - crop_size[0])
        y = np.random.randint(0, vol.shape[2] - crop_size[1])
        z = np.random.randint(0, vol.shape[3] - crop_size[2])       
        ix = slice(x, x + crop_size[0])
        iy = slice(y, y + crop_size[1])
        iz = slice(z, z + crop_size[2])
        vol = vol[:, ix,  iy, iz, :]
    elif vol.ndim == 4:
        x = np.random.randint(0, vol.shape[0] - crop_size[0])
        y = np.random.randint(0, vol.shape[1] - crop_size[1])
        z = np.random.randint(0, vol.shape[2] - crop_size[2])       
        ix = slice(x, x + crop_size[0])
        iy = slice(y, y + crop_size[1])
        iz = slice(z, z + crop_size[2])
        vol = vol[ix,  iy, iz, :]
    elif vol.ndim == 3:
        x = np.random.randint(0, vol.shape[0] - crop_size[0])
        y = np.random.randint(0, vol.shape[1] - crop_size[1])
        z = np.random.randint(0, vol.shape[2] - crop_size[2])       
        ix = slice(x, x + crop_size[0])
        iy = slice(y, y + crop_size[1])
        iz = slice(z, z + crop_size[2])
        vol = vol[ix,  iy, iz]
    else:
        vol = 0    
    return vol

def largest_connected_component(mask):
    labels = label(mask)
    #largest_cc = labels == np.argmax(np.bincount(labels.flat))
    largest_cc = (labels == 1)
    return largest_cc

def save_to_nii(vol, fname, res=[1,1,1], origin=[0,0,0]):
    if vol.ndim > 3:
        print('Cannot save volume with more than 3 dimensional information.')
    else:
        vol = np.transpose(vol, (2,1,0))
        vol = np.flip(vol, axis=1)
        output = SimpleITK.GetImageFromArray(vol, isVector=False)
        output.SetOrigin(origin)
        output.SetSpacing(res)
        SimpleITK.WriteImage(output, fname)

# load NIFTI volume using SimpleITK
def load_nii(fname):
    img = SimpleITK.ReadImage(fname)
    return np.transpose(SimpleITK.GetArrayFromImage(img))

# faster version of load_nii using nibabel package
def load_nii_nib(fname):
    ni_array = nib.load(fname).get_data()
    if ni_array.ndim == 2:
        return np.transpose(ni_array.astype(np.int16), axes=(1,0))
    else:    
        return np.transpose(ni_array, axes=(1,0,2))

def normalize(image, max_value = 1):
    """ Normalize an image such that the minimum value is equal to zero, and the maximum
    value to max_value.
    """
    return  max_value * ((image - np.amin(image)) / (np.amax(image) - np.amin(image)))

def correct_intensities(dicom, wmin=-1000, wmax=1000):
    dicom[dicom < wmin] = wmin
    dicom[dicom > wmax] = wmax
    return dicom

def save_to_nii2(vol, fname, res=[1,1,1], origin=[0,0,0]):
    if vol.ndim > 3:
        print('Cannot save volume with more than 3 dimensional information.')
    else:
        #Why are we transposing? One possibility : In scikit.draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        # vertex_row_coords is NOT ALONG ROW BUT ACTUALLY ROW INDEX (along column); But when we created the mask this was
        # filled with X co-ordinate from contours which is ALONG ROWS. So  scikit.draw.polygon created a tansposed mask
        #t that need to be transposed again.
        vol = np.transpose(vol)
        #vol = np.flip(vol, axis=1)
        output = SimpleITK.GetImageFromArray(vol, isVector=False)
        output.SetOrigin(origin)
        output.SetSpacing(res)
        SimpleITK.WriteImage(output, fname)

def normalize_intensities_ct(X, simple_normalization=True, exclude_air=False):
    X = X.astype(np.float32)
    
    if simple_normalization:
        X = X / 1000.0
    else:    
        if exclude_air:
            XX = X.ravel()
            ti = (XX > -900) & (XX < 4095)
            XXX = XX[ti]
            X = X - XXX.mean()
            X_std = XXX.std()
        else:
            X = X - X.mean()
            X_std = X.std()    
        if X_std > 0:      # avoid division by zero     
            X = X / X_std       
    
    return X

    # # simple normalization (works for CT scans in HU)            
    # X = X.astype(np.float32)
    # X = X / 1000.0

    # # simple normalization (works for AAPM T2w MRI)            
    # X = X.astype(np.float32)
    # X = (X - 180.0) / 360.0
    
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