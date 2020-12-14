import numpy as np
from scipy.ndimage import morphology
import SimpleITK

def surface_distance_array(test_labels, gt_labels, sampling=1, connectivity=1):
        input_1 = np.atleast_1d(test_labels.astype(np.bool))
        input_2 = np.atleast_1d(gt_labels.astype(np.bool))      
        conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
        s = input_1 ^ morphology.binary_erosion(input_1, conn)         # ^ is the logical XOR operator
        s_prime = input_2 ^ morphology.binary_erosion(input_2, conn)   # ^ is the logical XOR operator     
        dta = morphology.distance_transform_edt(~s, sampling)
        dtb = morphology.distance_transform_edt(~s_prime, sampling)
        sds = np.concatenate([np.ravel(dta[s_prime!=0]), np.ravel(dtb[s!=0])])        
        msd = sds.mean()
        sd_stdev=sds.std()
        rms = np.sqrt((sds**2).mean())
        hd  = sds.max()
        return msd,sd_stdev,rms, hd, sds

def surface_distance(test_labels, gt_labels, sampling=1, connectivity=1):
        input_1 = np.atleast_1d(test_labels.astype(np.bool))
        input_2 = np.atleast_1d(gt_labels.astype(np.bool))      
        conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
        s = input_1 ^ morphology.binary_erosion(input_1, conn)         # ^ is the logical XOR operator
        s_prime = input_2 ^ morphology.binary_erosion(input_2, conn)   # ^ is the logical XOR operator     
        dta = morphology.distance_transform_edt(~s, sampling)
        dtb = morphology.distance_transform_edt(~s_prime, sampling)
        sds = np.concatenate([np.ravel(dta[s_prime!=0]), np.ravel(dtb[s!=0])])        
        msd = sds.mean()
        sd_stdev=sds.std()
        rms = np.sqrt((sds**2).mean())
        hd  = sds.max()
        return msd, sd_stdev, rms, hd

def surface_distance_multi_label(test, gt, sampling=1):
        labels = np.unique(gt)
        ti = labels > 0
        unique_lbls = labels[ti]
        msd = np.zeros(len(unique_lbls))
        sd_stdev = np.zeros(len(unique_lbls))
        rms = np.zeros(len(unique_lbls))
        hd = np.zeros(len(unique_lbls))
        i = 0
        for lbl_num in unique_lbls:
                ti = (test == lbl_num)
                ti2 = (gt == lbl_num)
                test_mask = np.zeros(test.shape, dtype=np.uint8)
                test_mask[ti] = 1
                gt_mask = np.zeros(gt.shape, dtype=np.uint8)
                gt_mask[ti2] = 1
                msd[i], sd_stdev[i], rms[i], hd[i] = surface_distance(test_mask, gt_mask, sampling)
                i = i + 1
        return unique_lbls, msd, rms, hd

def dice_multi_label(test, gt):
        labels = np.unique(gt)
        ti = labels > 0
        unique_lbls = labels[ti]
        dice = np.zeros(len(unique_lbls))
        i = 0
        for lbl_num in unique_lbls:
                ti = (test == lbl_num)
                ti2 = (gt == lbl_num)
                test_mask = np.zeros(test.shape, dtype=np.uint8)
                test_mask[ti] = 1
                gt_mask = np.zeros(gt.shape, dtype=np.uint8)
                gt_mask[ti2] = 1
                dice[i] = dice_coef(test_mask, gt_mask)
                i = i + 1
        return dice

def surface_distance_from_nii(test_file, gt_file):    
        test = SimpleITK.ReadImage(test_file)
        test_lbl = np.transpose(SimpleITK.GetArrayFromImage(test), (2,1,0))
        gt = SimpleITK.ReadImage(gt_file)
        gt_lbl = np.transpose(SimpleITK.GetArrayFromImage(gt), (2,1,0))
        lbls, msd, rms, hd = surface_distance_multi_label(test_lbl, gt_lbl, gt.GetSpacing())
        return lbls, msd, rms, hd

def dice_from_nii(test_file, gt_file):    
        test = SimpleITK.ReadImage(test_file)
        test_lbl = np.transpose(SimpleITK.GetArrayFromImage(test), (2,1,0))
        gt = SimpleITK.ReadImage(gt_file)
        gt_lbl = np.transpose(SimpleITK.GetArrayFromImage(gt), (2,1,0))
        dice = dice_multi_label(test_lbl, gt_lbl)
        return dice

def dice_coef(a,b):
        a = a.astype(np.uint8).flatten()
        b = b.astype(np.uint8).flatten()
        dice = (2 * np.sum(np.multiply(a,b))) / (np.sum(a) + np.sum(b))
        return dice

def volume(label):
        volume=np.sum(label)
        return volume

def COM(label):
        #Get the coordinates of the nonzero indices
        indices=np.nonzero(label)
        #Take the average of the index values in each direction to get the center of mass
        COMx=np.mean(indices[0])
        COMy=np.mean(indices[1])
        COMz=np.mean(indices[2])
        return COMx, COMy, COMz