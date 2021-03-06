# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 21:08:40 2021
Helper Methods for scribble and graph cut
@author: Supratik Bose
"""

import os
import sys
import shutil
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import nibabel as nib
from scipy.ndimage import morphology
import SimpleITK
import pandas as pd


#Method to create 2D-disk and 3D ball to be used for fat scribble
def disk(n):
    struct = np.zeros((2 * n + 1, 2 * n + 1))
    x, y = np.indices((2 * n + 1, 2 * n + 1))
    mask = (x - n)**2 + (y - n)**2 <= n**2
    struct[mask] = 1
    return struct.astype(np.bool)

def ball(n):
    struct = np.zeros((2*n+1, 2*n+1, 2*n+1))
    x, y, z = np.indices((2*n+1, 2*n+1, 2*n+1))
    mask = (x - n)**2 + (y - n)**2 + (z - n)**2 <= n**2
    struct[mask] = 1
    return struct.astype(np.bool)

def dice_coef_func(a,b):
    a = a.astype(np.uint8).flatten()
    b = b.astype(np.uint8).flatten()
    dice = (2 * np.sum(np.multiply(a,b))) / (np.sum(a) + np.sum(b))
    return dice

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
            dice[i] = dice_coef_func(test_mask, gt_mask)
            i = i + 1
    return dice

def checkFolderExistenceAndCreate(folderPath):
    #if folderPath does not exist create it
    if os.path.exists(folderPath):
        #Check if it is a directory or not
        if os.path.isfile(folderPath): 
            sys.exit(folderPath, ' is a file and not directory. Exiting.') 
    else:
        #create 
        os.makedirs(folderPath)

def readAndScaleImageData(fileName, folderName, clipFlag, clipLow, clipHigh, scaleFlag, scaleFactor,\
                          meanSDNormalizeFlag, finalDataType, \
                          isLabelData, labels_to_train_list, verbose=False): 
    returnNow = False
    #check file existence
    filePath = os.path.join(folderName, fileName)            
    if os.path.exists(filePath):
        pass
    else:
        print(filePath, ' does not exist')  
        returnNow = True  
    if returnNow:
        sys.exit() 
    #We are here => returnNow = False
    #Also note #axes: depth, height, width
    fileData = np.transpose(nib.load(filePath).get_fdata(), axes=(2,1,0))  
    #Debug code
    if verbose:
        dataMin = fileData.min()
        dataMax = fileData.max()
        print('fileName - shape - type -min -max: ', fileName, ' ', fileData.shape, ' ', fileData.dtype, ' ', dataMin, ' ', dataMax)
    #Clamp                          
    if True == clipFlag:
        np.clip(fileData, clipLow, clipHigh, out= fileData)
    #Scale   
    if True == scaleFlag:
        fileData = fileData / scaleFactor
    #mean SD Normalization
    if True == meanSDNormalizeFlag:
        fileData = (fileData - np.mean(fileData))/np.std(fileData)
    #Type conversion
    fileData = fileData.astype(finalDataType)
    if True == isLabelData:
        # pick specific labels to train (if training labels other than 1s and 0s)
        if labels_to_train_list != [1]:
            temp = np.zeros(shape=fileData.shape, dtype=fileData.dtype)
            new_label_value = 1
            for lbl in labels_to_train_list: 
                ti = (fileData == lbl)
                temp[ti] = new_label_value
                new_label_value += 1
            fileData = temp
    return fileData


#Understand Bounding box function
#Rewriting: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def bbox2_3D(img, expandBBoxFlag=False, pad=0):
    """
    In 3D np array assuming:
    - first axis is slice, -> axial plane index
    - 2nd axis row in each slice, -> coronal plane index
    - last axis is column in each slice -> sagittal plane index
    """
    #np.any will search for non zero elemens in the axis mentioned 
    #So if axis=(1,2): Axial plane, it will search over axis (1=row and 2=col) and return  result
    #for each slice - axial plane
    nonZeroAxialPlanes = np.any(img, axis=(1, 2))
    #So if axis=(0,2) - Corronal plane, it will search over axis (0=slice and 2=col) and return  result
    #for corronal plane
    nonZeroCoronalPlanes = np.any(img, axis=(0, 2))
    #So if axis=(0,1)- sagittal plane, it will search over axis (0=slice and 1=row) and return  result
    #for each sagittal plane
    nonZeroSagittalPlanes = np.any(img, axis=(0, 1))
    
    #result from np.any(): [False  True  True  True False]
    #result of applying np.where() : (array([ 1,2,3]),)
    #So its a tuple of 1-D array on which one applies [0][[0, -1]]
    #The first [0] takes the first array element out of the tuple
    #The next [[0,-1]] is using list based indexing and getting the first and last element out

    axial_min, axial_max = np.where(nonZeroAxialPlanes)[0][[0, -1]]
    coronal_min, coronal_max = np.where(nonZeroCoronalPlanes)[0][[0, -1]]
    sagittal_min, sagittal_max = np.where(nonZeroSagittalPlanes)[0][[0, -1]]
    
    if True == expandBBoxFlag:
        axial_min = max(axial_min-pad,0)
        axial_max = min(axial_max+pad,img.shape[0]-1)
        coronal_min = max(coronal_min-pad,0)
        coronal_max = min(coronal_max+pad,img.shape[1]-1)        
        sagittal_min = max(sagittal_min-pad,0)
        sagittal_max = min(sagittal_max+pad,img.shape[2]-1)
    return axial_min, axial_max, coronal_min, coronal_max, sagittal_min, sagittal_max

def getUnionBoundingBoxWithPadding(gt, pred, bbPad):
    """
    gt: volume 1; 1st dim: slice; 2nd dim: row; 3rd dim col
    pred: volume 2 of same shape as gt
    bbPad: Padding amount to be added over union
    
    return: bounding box limits (inclusive on both end)
    """
    #In BB calculation: a : axial, c: corronal, s : sagittal
    #BB around  GT
    a_min_g, a_max_g, c_min_g, c_max_g, s_min_g, s_max_g = bbox2_3D(gt, expandBBoxFlag=False, pad=0)
    #BB around  pred
    a_min_p, a_max_p, c_min_p, c_max_p, s_min_p, s_max_p = bbox2_3D(pred, expandBBoxFlag=False, pad=0)
    #common BB encompassing both GT and pred  and padding added
    a_min, a_max, c_min, c_max, s_min, s_max = \
            min(a_min_g, a_min_p), max(a_max_g, a_max_p),\
            min(c_min_g, c_min_p), max(c_max_g, c_max_p),\
            min(s_min_g, s_min_p), max(s_max_g, s_max_p)   
    #After added padding: Note both GT and Pred has the same shape
    a_min = max(a_min-bbPad,0)
    a_max = min(a_max+bbPad,     gt.shape[0]-1)
    c_min = max(c_min-bbPad,0)
    c_max = min(c_max+bbPad,   gt.shape[1]-1)        
    s_min = max(s_min-bbPad,0)
    s_max = min(s_max+bbPad,  gt.shape[2]-1) 
    return a_min, a_max, c_min, c_max, s_min, s_max

#Choose scribble from misclassified region (2D) : input includes fraction of misclassified pixels 
# to be added as initial scribble as well as scribble-brush diameter
def chooseScribbleFromMissedFGOrWrongCBG2D(misclassifiedRegion, listOfSlicesWithGT, fractionDivider, dilation_diam):
    """
        misclassifiedRegion : int8 binary volume of fgMissed or bgWrongC with slice as first dimension
        listOfSlicesWithGT : list of slices where grounf truth is present
        fractionDivider : positive integer by which number of one pixels in a slice will be divide
                to decide what fraction of them will be chosen. 
                If fractionDivider=1, all of them get chosen
        dilation_diam: dimeter of disk : 1,2,3: disk diameter of  scribble
    """
    resultBinary = np.zeros_like(misclassifiedRegion)
    #numSlices = misclassifiedRegion.shape[0]
    #for sliceId in range(numSlices):
    for sliceId in listOfSlicesWithGT:
        onePixelsThisSlice = np.where(misclassifiedRegion[sliceId,...]==1)    
        onePixelCoordsThisSlice = list(zip(onePixelsThisSlice[0], onePixelsThisSlice[1]))
        #print(missedFGResultCoordsThisSlice)
        numOnePixelCoordsThisSlice = len(onePixelCoordsThisSlice)
        #Debug:
        #print('Slice: ', sliceId, 'numOnePixelhisSlice ', numOnePixelCoordsThisSlice)
        numScribblesFromThisSlice = numOnePixelCoordsThisSlice // fractionDivider
        chosenScribbleCoordsThisSlice = random.sample(onePixelCoordsThisSlice, numScribblesFromThisSlice)
        #Convert chose coords  in 3D
        chosenScribbleCoordsThisSlice = \
          [ tuple([sliceId] + list(tupleElem)) for  tupleElem in chosenScribbleCoordsThisSlice]
        #print(chosenFGScribbleFromMissedFGCoordsThisSlice)
        for coord in chosenScribbleCoordsThisSlice : resultBinary[coord] = 1 
        #dialate in slice        
        resultSlice = resultBinary[sliceId,...]
        #print('result before dilation ')
        #print(resultSlice)
        resultSlice = \
           scipy.ndimage.binary_dilation(resultSlice,structure=disk(dilation_diam)).astype(resultSlice.dtype)
        #print('result after dilation ')
        #print(resultSlice)
        #But make sure it does not go beyond original binary
        resultSlice = resultSlice * misclassifiedRegion[sliceId,...]
        #print('result after clipping ')
        #print(resultSlice)
        resultBinary[sliceId,...] = resultSlice
    #Debug
    #print('Debug: numScrVoxelsFromMissed-2D: ', np.sum(resultBinary))
    return resultBinary

#Method to choose ring shaped scribble in definitely correctly idenified region (2D)
def chooseScribbleFromDefiniteRegion2D(definiteRegion,  listOfSlicesWithGT,  dilation_diam, \
        stayCloseToGT, gt, dia_close):
    """
        definiteRegion : int8 binary volume of definiteRegion with slice as first dimension 
        listOfSlicesWithGT : list of slices where grounf truth is present        
        dilation_diam: dimeter of disk : 2, 3, 4 : a diam x diam window is placed to choose scribble
        stayCloseToGT: flag to indicate to generate mask, esp bgScribbleFromDefiniteBG close to GT
        gt: ground truth
        dia_close : how close to GT; should be larger than dilation_diam
    """
    resultBinary = np.zeros_like(definiteRegion)
    #numSlices = misclassifiedRegion.shape[0]
    #for sliceId in range(numSlices):
    for sliceId in listOfSlicesWithGT:
        definiteSlice = definiteRegion[sliceId,...]
        erodedSlice = \
          scipy.ndimage.binary_erosion(definiteSlice,structure=disk(dilation_diam)).astype(definiteSlice.dtype)
        scribbleSlice = definiteSlice - erodedSlice
        if True ==stayCloseToGT:
            gtSlice = gt[sliceId,...]
            expandedGTSlice = \
              scipy.ndimage.binary_dilation(gtSlice,structure=disk(dia_close)).astype(gtSlice.dtype)
            #Modify scribbleSlice to stay close to GT
            scribbleSlice = scribbleSlice * expandedGTSlice
        #print('result after subtraction of erosion ')
        #print(scribbleSlice)
        resultBinary[sliceId,...] = scribbleSlice
    #Debug
    #print('Debug: numScrVoxelsFromDefinite-2D: ', np.sum(resultBinary))
    return resultBinary

   #function to generate  different scribble regions automatically
def autoGenerateScribbleRegions2D(gt, pred, fractionDivider, dilation_diam):
    """
        gt : int8 binary volume of ground truth  with slice as first dimension 
        pred : int8 binary volume of prediction  with slice as first dimension
        fractionDivider : positive integer by which number of one pixels in a slice will be divide
                to decide what fraction of them will be chosen. 
                If fractionDivider=1, all of them get chosen
        dilation_diam: dimeter of disk : 2, 3, 4 : a diam x diam window is placed to choose scribble
    """    
    #https://stackoverflow.com/questions/4260280/if-else-in-a-list-comprehension
    listOfSlicesWithGT = [sliceId for sliceId in range(gt.shape[0]) if np.sum(gt[sliceId,...]) >0]
    #print(listOfSlicesWithGT)

    gt_comp = 1-gt 
    pred_comp = 1-pred
    gtMinusPred = gt * pred_comp 
    predMinusGt = pred * gt_comp

    definiteBG = gt_comp # This is also definite background
    definiteFG = gt * pred #This is definite FG
    fgMissed = gtMinusPred  #  FG missed by classified
    bgWrongC = predMinusGt #  BG wrongly classified as FG


    fgScribbleFromFGMissed = chooseScribbleFromMissedFGOrWrongCBG2D(misclassifiedRegion=fgMissed, \
      listOfSlicesWithGT=listOfSlicesWithGT, fractionDivider=fractionDivider,\
      dilation_diam=dilation_diam)

    bgScribbleFromBGWrongC = chooseScribbleFromMissedFGOrWrongCBG2D(misclassifiedRegion=bgWrongC, \
      listOfSlicesWithGT=listOfSlicesWithGT, fractionDivider=fractionDivider,\
      dilation_diam=dilation_diam)

    fgScribbleFromDefiniteFG = chooseScribbleFromDefiniteRegion2D(definiteRegion=definiteFG,\
       listOfSlicesWithGT=listOfSlicesWithGT, dilation_diam=dilation_diam, \
      stayCloseToGT=False, gt=gt, dia_close=dilation_diam+5)
    #Use stayCloseToGT: True  to choose bgScribbleFromDefiniteBG close to GT 
    #and not on outer boundary of image; 
    #dia_close : how close to GT; should be larger than dilation_diam
    bgScribbleFromDefiniteBG = chooseScribbleFromDefiniteRegion2D(definiteRegion=definiteBG,\
       listOfSlicesWithGT=listOfSlicesWithGT, dilation_diam=dilation_diam, \
      stayCloseToGT=True, gt=gt, dia_close=dilation_diam+5)
    
    return listOfSlicesWithGT, fgScribbleFromFGMissed, bgScribbleFromBGWrongC,\
        fgScribbleFromDefiniteFG, bgScribbleFromDefiniteBG 


#Choose scribble from misclassified region (3D) : input includes fraction of misclassified pixels 
# to be added as initial scribble as well as scribble-brush diameter
def chooseScribbleFromMissedFGOrWrongCBG3D(misclassifiedRegion, boundingBoxVol, fractionDivider, dilation_diam,\
                                           useAtmostNScribbles):
    """
        misclassifiedRegion : int8 binary volume of fgMissed or bgWrongC with slice as first dimension
        boundingBoxVol:  Binary (int0, 0-1) volume within which scribbles should be limited
        fractionDivider : positive integer by which number of one pixels in a slice will be divide
                to decide what fraction of them will be chosen. 
                If fractionDivider=1, all of them get chosen
        dilation_diam: dimeter of disk : 1,2,3: disk diameter of  scribble
        useAtmostNScribbles: Integer, If <=0, ignored 
    """
    resultBinary = np.zeros_like(misclassifiedRegion)
    #Constrain the misclassified region with boundary volume
    misclassifiedRegion *= boundingBoxVol
    onePixelsThisVol = np.where(misclassifiedRegion)
    onePixelCoordsThisVol = list(zip(onePixelsThisVol[0], onePixelsThisVol[1], onePixelsThisVol[2]))
    #print(onePixelCoordsThisVol)
    numOnePixelCoordsThisVol = len(onePixelCoordsThisVol)
    #Debug:
    #print('numOnePixelCoordsThisVol ', numOnePixelCoordsThisVol)
    numScribblesFromThisVol = numOnePixelCoordsThisVol // fractionDivider
    if int(useAtmostNScribbles) > 0 :
        numScribblesFromThisVol = min(numScribblesFromThisVol, int(useAtmostNScribbles))
    chosenScribbleCoordsThisVol = random.sample(onePixelCoordsThisVol, numScribblesFromThisVol)
    #print(chosenScribbleCoordsThisVol)
    for coord in chosenScribbleCoordsThisVol : resultBinary[coord] = 1
    #dilate in volume       
    #print('result before dilation ')
    #print(resultBinary)
    resultBinary = \
       scipy.ndimage.binary_dilation(resultBinary,structure=ball(dilation_diam)).astype(resultBinary.dtype)
    #print('result after dilation ')
    #print(resultBinary)
    #But make sure it does not go beyond original binary 
    resultBinary = resultBinary * misclassifiedRegion
    #print('result after clipping ')
    #print(resultBinary)
    #Debug
    #print('Debug: numScrVoxelsFromMissed-3D: ', np.sum(resultBinary))
    return resultBinary, numScribblesFromThisVol

#Method to choose scribble in definitely correctly idenified region (3D): 
#Its chosen from a 3D shell within definite region
def chooseScribbleFromDefiniteRegion3D(definiteRegion,  boundingBoxVol,   fractionDivider, dilation_diam,\
                                           useAtmostNScribbles):
    """
        definiteRegion : int8 binary volume of definiteRegion with slice as first dimension 
        boundingBoxVol:  Binary (int0, 0-1) volume within which scribbles should be limited  
        fractionDivider : positive integer by which number of one pixels in a slice will be divide
                to decide what fraction of them will be chosen. 
                If fractionDivider=1, all of them get chosen
        dilation_diam: dimeter of disk : 2, 3, 4 : a diam x diam window is placed to choose scribble
        useAtmostNScribbles: Integer, If <=0, ignored 
    """
    resultBinary = np.zeros_like(definiteRegion)
    #Erode the definite region 
    erodedRegion = \
          scipy.ndimage.binary_erosion(definiteRegion,structure=ball(dilation_diam)).astype(definiteRegion.dtype)
    scribbleShell = definiteRegion - erodedRegion
    #Constrain the scribbleShell region with boundary volume
    scribbleShell *= boundingBoxVol
    
    onePixelsThisVol = np.where(scribbleShell)
    onePixelCoordsThisVol = list(zip(onePixelsThisVol[0], onePixelsThisVol[1], onePixelsThisVol[2]))
    #print(onePixelCoordsThisVol)
    numOnePixelCoordsThisVol = len(onePixelCoordsThisVol)
    #Debug:
    #print('numOnePixelCoordsThisVol ', numOnePixelCoordsThisVol)
    numScribblesFromThisVol = numOnePixelCoordsThisVol // fractionDivider
    if int(useAtmostNScribbles) > 0 :
        numScribblesFromThisVol = min(numScribblesFromThisVol, int(useAtmostNScribbles))
    chosenScribbleCoordsThisVol = random.sample(onePixelCoordsThisVol, numScribblesFromThisVol)
    #print(chosenScribbleCoordsThisVol)
    for coord in chosenScribbleCoordsThisVol : resultBinary[coord] = 1
    #dilate in volume       
    #print('result before dilation ')
    #print(resultBinary)
    resultBinary = \
       scipy.ndimage.binary_dilation(resultBinary,structure=ball(dilation_diam)).astype(resultBinary.dtype)
    #print('result after dilation ')
    #print(resultBinary)
    #But make sure it does not go beyond original binary 
    resultBinary = resultBinary * scribbleShell
    #print('result after clipping ')
    #print(resultBinary)
    #Debug
    #print('Debug: numScrVoxelsFromDefinite-3D: ', np.sum(resultBinary))
    return resultBinary, numScribblesFromThisVol

#function to generate  different scribble regions automatically
def autoGenerateScribbleAndBBox3D(gt, pred, bbPad, fractionDivider, dilation_diam,\
                                           useAtmostNScribblesPerRegion):
    """
        gt : int8 binary volume of ground truth  with slice as first dimension 
        pred : int8 binary volume of prediction  with slice as first dimension
        bbPad: padding to be used while creating  BB 
        fractionDivider : positive integer by which number of one pixels in a slice will be divide
                to decide what fraction of them will be chosen. 
                If fractionDivider=1, all of them get chosen
        dilation_diam: dimeter of disk : 2, 3, 4 : a diam x diam window is placed to choose scribble
        useAtmostNScribblesPerRegion: Integer, If <=0, ignored 
        
        Return: 
            binLimit: axial, corronal and sagittal limits of the bounding box to be used in graphcut, 
            bbVolume: binary bounding box volume; 0 out side, 1 inside bounding box, 
            numFGS: number foreground Scribble (in missed and definite FG region)
            numBGS: number of backrground Scribble  (in wrongly classified and definite BG region) 
            fgScribbleFromFGMissed: foreground scribbles from missed FG region 
            bgScribbleFromBGWrongC: background scribbles from BG region wrongly classified as FG
            fgScribbleFromDefiniteFG: foreground scribbles from definite FG region 
            bgScribbleFromDefiniteBG: BG scribbles from definite BG region
            fgScribble: union of foreground scribbles 
            bgScribble: union of background scribbles 
    """   
    #In BB calculation: a : axial, c: corronal, s : sagittal
    a_min, a_max, c_min, c_max, s_min, s_max = getUnionBoundingBoxWithPadding(gt, pred, bbPad)
    binLimit = [a_min, a_max, c_min, c_max, s_min, s_max]
    bbVolume = np.zeros_like(gt)
    #Note the +1 without which we were notmaking the highest limit 1
    bbVolume[a_min:a_max+1, c_min:c_max+1, s_min:s_max+1]=1

    gt_comp = 1-gt 
    pred_comp = 1-pred
    gtMinusPred = gt * pred_comp 
    predMinusGt = pred * gt_comp

    definiteBG = gt_comp # This is also definite background
    definiteFG = gt * pred #This is definite FG
    fgMissed = gtMinusPred  #  FG missed by classified
    bgWrongC = predMinusGt #  BG wrongly classified as FG

    #misclassifiedRegion, boundingBoxVol, fractionDivider, dilation_diam
    fgScribbleFromFGMissed, numFGSM = chooseScribbleFromMissedFGOrWrongCBG3D(misclassifiedRegion=fgMissed, \
      boundingBoxVol=bbVolume, fractionDivider=fractionDivider,\
    dilation_diam=dilation_diam, useAtmostNScribbles=useAtmostNScribblesPerRegion)
    
    #misclassifiedRegion, boundingBoxVol, fractionDivider, dilation_diam
    bgScribbleFromBGWrongC, numBGSM = chooseScribbleFromMissedFGOrWrongCBG3D(misclassifiedRegion=bgWrongC, \
      boundingBoxVol=bbVolume, fractionDivider=fractionDivider,\
    dilation_diam=dilation_diam, useAtmostNScribbles=useAtmostNScribblesPerRegion)
    
    #definiteRegion,  boundingBox,  dilation_diam
    fgScribbleFromDefiniteFG, numFGSD = chooseScribbleFromDefiniteRegion3D(definiteRegion=definiteFG,\
       boundingBoxVol=bbVolume, fractionDivider=fractionDivider,\
    dilation_diam=dilation_diam, useAtmostNScribbles=useAtmostNScribblesPerRegion)
    
    #definiteRegion,  boundingBox,  dilation_diam
    bgScribbleFromDefiniteBG, numBGSD = chooseScribbleFromDefiniteRegion3D(definiteRegion=definiteBG,\
       boundingBoxVol=bbVolume, fractionDivider=fractionDivider,\
    dilation_diam=dilation_diam, useAtmostNScribbles=useAtmostNScribblesPerRegion)
    
    numFGS = numFGSM + numFGSD
    numBGS = numBGSM + numBGSD
    
    # FG and BG Scribble  are supposed to be disjoint; Still clip to 1    
    fgScribble = np.clip(fgScribbleFromFGMissed + fgScribbleFromDefiniteFG, 0, 1)
    bgScribble = np.clip(bgScribbleFromBGWrongC + bgScribbleFromDefiniteBG, 0, 1)

    return binLimit, bbVolume, numFGS, numBGS, fgScribbleFromFGMissed, bgScribbleFromBGWrongC,\
        fgScribbleFromDefiniteFG, bgScribbleFromDefiniteBG, fgScribble, bgScribble

defaultPatDataConfig = \
    {   
         "ctSuffix": "_ct.nii.gz",
         "ptSuffix": "_pt.nii.gz",
   		 "gtSuffix": "_ct_gtvt.nii.gz",
   		 "predSuffix": "_segment.nii.gz",
   		 "softmaxSuffix": "_softMax.nii.gz",
   		 "fgScribbleSuffix": "_fgScribble.nii.gz",
   		 "bgScribbleSuffix": "_bgScribble.nii.gz",
         "ct_gCutSuffix": "_ct_gCut.nii.gz",
   		 "pet_gCutSuffix": "_pet_gCut.nii.gz",         
         "softmax_gCutSuffix": "_softmax_gCut.nii.gz",
   		 "ct_low": -1000,
   		 "ct_high": 3095,
   		 "ct_clipImages": False,
   		 "ct_scaleImages": False,
   		 "ct_scaleFactor": 1000,
   		 "ct_normalizeImages": False,
   		 "pt_low": -1000,
   		 "pt_high": 3095,
   		 "pt_clipImages": False,
   		 "pt_scaleImages": False,
   		 "pt_scaleFactor": 1,
   		 "pt_normalizeImages": False,
   		 "labels_to_train": [1]
    }
defaultAutoScribbleAndGCConfig1 = \
    {
      	 "bbPad": 2,
      	 "fractionDivider": 10,
      	 "dilation_diam": 2,
      	 "useAtmostNScribblesPerRegion": 3,
    	 "segparams_ssgc": 
        {
         "method": "graphcut",
         "pairwise_alpha": 1.0,
         "modelparams": 
            {
             "cvtype": "full",
             "params": 
                {
                 "covariance_type": "full",
                 "n_components": 1
                }
            },
         "return_only_object_with_seeds": True
        }
    }  
                
defaultAutoScribbleAndGCConfig2 = \
     {
      'method': 'graphcut',
      'use_boundary_penalties': False,
      'boundary_dilatation_distance': 2,
      'boundary_penalties_weight': 1,
      'modelparams': 
          {
           'type': 'gmmsame',
           'params':
               {
                'n_components': 2
               },
            'return_only_object_with_seeds': True
            # 'fv_type': 'fv_extern',
            # 'fv_extern': fv_function,
            # 'adaptation': 'original_data',
          },
     }
                     
def createGCInputUsingGT(patientName, srcFolder, expFolder,\
                  expPatName = "expPat", patDataConfig=defaultPatDataConfig,\
                  autoScribbleAndGCConfig = defaultAutoScribbleAndGCConfig1,\
                  verbose=False):
    #check existence of destination folder
    checkFolderExistenceAndCreate(expFolder)
    #Transfer file into experiment directory with fixed name
    shutil.copy(os.path.join(srcFolder, patientName + patDataConfig['ctSuffix']), \
                os.path.join(expFolder, expPatName + patDataConfig['ctSuffix']))
    shutil.copy(os.path.join(srcFolder, patientName + patDataConfig['ptSuffix']), \
                os.path.join(expFolder, expPatName + patDataConfig['ptSuffix']))
    shutil.copy(os.path.join(srcFolder, patientName + patDataConfig['predSuffix']), \
                os.path.join(expFolder, expPatName + patDataConfig['predSuffix']))
    shutil.copy(os.path.join(srcFolder, patientName + patDataConfig['softmaxSuffix']), \
                os.path.join(expFolder, expPatName + patDataConfig['softmaxSuffix']))
    #Since ground truth is present one can assume presence of ground truth file
    shutil.copy(os.path.join(srcFolder, patientName + patDataConfig['gtSuffix']), \
                os.path.join(expFolder, expPatName + patDataConfig['gtSuffix']))
        
    #Read np ndarray from the files, to scale or not to scale? 
    ctFileName = expPatName + patDataConfig['ctSuffix']
    ctData = readAndScaleImageData(fileName=ctFileName,\
        folderName=expFolder, clipFlag = patDataConfig['ct_clipImages'],\
        clipLow=patDataConfig['ct_low'], clipHigh =patDataConfig['ct_high'],\
        scaleFlag=patDataConfig['ct_scaleImages'], scaleFactor=patDataConfig['ct_scaleFactor'],\
        meanSDNormalizeFlag = patDataConfig['ct_normalizeImages'], finalDataType = np.float32,\
        isLabelData=False, labels_to_train_list=None, verbose=verbose)
    
    ptFileName = expPatName + patDataConfig['ptSuffix']
    ptData = readAndScaleImageData(fileName=ptFileName,\
        folderName=expFolder, clipFlag = patDataConfig['pt_clipImages'],\
        clipLow=patDataConfig['pt_low'], clipHigh =patDataConfig['pt_high'],\
        scaleFlag=patDataConfig['pt_scaleImages'], scaleFactor=patDataConfig['pt_scaleFactor'],\
        meanSDNormalizeFlag = patDataConfig['pt_normalizeImages'], finalDataType = np.float32,\
        isLabelData=False, labels_to_train_list=None, verbose=verbose)

    softmaxFileName = expPatName + patDataConfig['softmaxSuffix']
    softmaxData = readAndScaleImageData(fileName=softmaxFileName,\
        folderName=expFolder, clipFlag = False,\
        clipLow=0.0, clipHigh =1.0,\
        scaleFlag=False, scaleFactor=1.0, meanSDNormalizeFlag = False, finalDataType = np.float32,\
        isLabelData=False, labels_to_train_list=None, verbose=verbose)
    
    gtFileName = expPatName + patDataConfig['gtSuffix']
    gtData = readAndScaleImageData(fileName=gtFileName,\
        folderName=expFolder, clipFlag = False,\
        clipLow=0, clipHigh = 0,\
        scaleFlag=False, scaleFactor=1, meanSDNormalizeFlag = False, finalDataType = np.int8, \
        isLabelData=True, labels_to_train_list=patDataConfig['labels_to_train'], verbose=verbose)
    
    predFileName = expPatName + patDataConfig['predSuffix']
    predFromNN = readAndScaleImageData(fileName=predFileName,\
        folderName=expFolder, clipFlag = False,\
        clipLow=0, clipHigh = 0,\
        scaleFlag=False, scaleFactor=1, meanSDNormalizeFlag = False, finalDataType = np.int8, \
        isLabelData=True, labels_to_train_list=patDataConfig['labels_to_train'], verbose=verbose)
    


    binLimit, bbVolume, numFGS, numBGS, fgScribbleFromFGMissed, bgScribbleFromBGWrongC,\
        fgScribbleFromDefiniteFG, bgScribbleFromDefiniteBG, fgScribble, bgScribble\
        = autoGenerateScribbleAndBBox3D(gt=gtData, pred=predFromNN,\
            bbPad=autoScribbleAndGCConfig['bbPad'],fractionDivider=autoScribbleAndGCConfig['fractionDivider'],\
            dilation_diam=autoScribbleAndGCConfig['dilation_diam'],\
            useAtmostNScribblesPerRegion=autoScribbleAndGCConfig['useAtmostNScribblesPerRegion'])
            
    modelImage_nii_aff = nib.load(os.path.join(expFolder, ctFileName)).affine
    
    #fgScrM_name = expPatName + '_fgScrM_fr_{:>02d}_D_{:>02d}.nii.gz'.format(fractionDivider, dilation_diam)
    fgScribbleFileName = expPatName + patDataConfig['fgScribbleSuffix']
    fgScribbleFilePath = os.path.join(expFolder, fgScribbleFileName)
    nib.save(nib.Nifti1Image(np.transpose(fgScribble, axes=(2,1,0)),\
                             affine=modelImage_nii_aff),fgScribbleFilePath)
    
    bgScribbleFileName = expPatName + patDataConfig['bgScribbleSuffix']
    bgScribbleFilePath = os.path.join(expFolder, bgScribbleFileName)
    nib.save(nib.Nifti1Image(np.transpose(bgScribble, axes=(2,1,0)),\
                             affine=modelImage_nii_aff),bgScribbleFilePath)  
    #Create config and json file
    graphCutInputConfig = {}    
    graphCutInputConfig["patientName"] = patientName
    graphCutInputConfig["srcFolder"] = srcFolder
    graphCutInputConfig["expFolder"] = expFolder
    graphCutInputConfig["expPatName"] = expPatName
    graphCutInputConfig["binLimit"] = [ int(k) for k in binLimit]
    graphCutInputConfig["numFGS"] = int(numFGS) #convert int64 to int
    graphCutInputConfig["numBGS"] = int(numBGS)
    graphCutInputConfig["patDataConfig"] = patDataConfig
    graphCutInputConfig["autoScribbleAndGCConfig"] = autoScribbleAndGCConfig
    graphCutInputConfig_JsonFilePath =  os.path.join(expFolder, "graphCutInputConfig.json")
    with open(graphCutInputConfig_JsonFilePath, 'w') as fp:
        json.dump(graphCutInputConfig, fp, ) #, indent='' #, indent=4
        fp.close()        
    return  ctData,  ptData, gtData, softmaxData, predFromNN,\
            binLimit, bbVolume, numFGS, numBGS,\
            fgScribbleFromFGMissed, bgScribbleFromBGWrongC,\
            fgScribbleFromDefiniteFG, bgScribbleFromDefiniteBG,\
            fgScribble, bgScribble,\
            graphCutInputConfig, graphCutInputConfig_JsonFilePath
            


       

################## Original autoScribbleExperiment.py #################################
#import os
#import glob
#import sys
#
#import shutil
#import json
#
#import numpy as np
#import random
#import matplotlib.pyplot as plt
#import scipy
#from scipy import ndimage
#import nibabel as nib
#from scipy.ndimage import morphology
#import SimpleITK
#import pandas as pd
#
##Method to create 2D-disk and 3D ball to be used for fat scribble
#def disk(n):
#    struct = np.zeros((2 * n + 1, 2 * n + 1))
#    x, y = np.indices((2 * n + 1, 2 * n + 1))
#    mask = (x - n)**2 + (y - n)**2 <= n**2
#    struct[mask] = 1
#    return struct.astype(np.bool)
#
#def ball(n):
#    struct = np.zeros((2*n+1, 2*n+1, 2*n+1))
#    x, y, z = np.indices((2*n+1, 2*n+1, 2*n+1))
#    mask = (x - n)**2 + (y - n)**2 + (z - n)**2 <= n**2
#    struct[mask] = 1
#    return struct.astype(np.bool)
#
#def dice_coef_func(a,b):
#    a = a.astype(np.uint8).flatten()
#    b = b.astype(np.uint8).flatten()
#    dice = (2 * np.sum(np.multiply(a,b))) / (np.sum(a) + np.sum(b))
#    return dice
#
#def dice_multi_label(test, gt):
#    labels = np.unique(gt)
#    ti = labels > 0
#    unique_lbls = labels[ti]
#    dice = np.zeros(len(unique_lbls))
#    i = 0
#    for lbl_num in unique_lbls:
#            ti = (test == lbl_num)
#            ti2 = (gt == lbl_num)
#            test_mask = np.zeros(test.shape, dtype=np.uint8)
#            test_mask[ti] = 1
#            gt_mask = np.zeros(gt.shape, dtype=np.uint8)
#            gt_mask[ti2] = 1
#            dice[i] = dice_coef_func(test_mask, gt_mask)
#            i = i + 1
#    return dice
#
#def checkFolderExistenceAndCreate(folderPath):
#    #if folderPath does not exist create it
#    if os.path.exists(folderPath):
#        #Check if it is a directory or not
#        if os.path.isfile(folderPath): 
#            sys.exit(folderPath, ' is a file and not directory. Exiting.') 
#    else:
#        #create 
#        os.makedirs(folderPath)
#
#def readAndScaleImageData(fileName, folderName, clipFlag, clipLow, clipHigh, scaleFlag, scaleFactor,\
#                          meanSDNormalizeFlag, finalDataType, \
#                          isLabelData, labels_to_train_list, verbose=False): 
#    returnNow = False
#    #check file existence
#    filePath = os.path.join(folderName, fileName)            
#    if os.path.exists(filePath):
#        pass
#    else:
#        print(filePath, ' does not exist')  
#        returnNow = True  
#    if returnNow:
#        sys.exit() 
#    #We are here => returnNow = False
#    #Also note #axes: depth, height, width
#    fileData = np.transpose(nib.load(filePath).get_fdata(), axes=(2,1,0))  
#    #Debug code
#    if verbose:
#        dataMin = fileData.min()
#        dataMax = fileData.max()
#        print('fileName - shape - type -min -max: ', fileName, ' ', fileData.shape, ' ', fileData.dtype, ' ', dataMin, ' ', dataMax)
#    #Clamp                          
#    if True == clipFlag:
#        np.clip(fileData, clipLow, clipHigh, out= fileData)
#    #Scale   
#    if True == scaleFlag:
#        fileData = fileData / scaleFactor
#    #mean SD Normalization
#    if True == meanSDNormalizeFlag:
#        fileData = (fileData - np.mean(fileData))/np.std(fileData)
#    #Type conversion
#    fileData = fileData.astype(finalDataType)
#    if True == isLabelData:
#        # pick specific labels to train (if training labels other than 1s and 0s)
#        if labels_to_train_list != [1]:
#            temp = np.zeros(shape=fileData.shape, dtype=fileData.dtype)
#            new_label_value = 1
#            for lbl in labels_to_train_list: 
#                ti = (fileData == lbl)
#                temp[ti] = new_label_value
#                new_label_value += 1
#            fileData = temp
#    return fileData
#
##Understand Bounding box function
##Rewriting: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
#def bbox2_3D(img, expandBBoxFlag=False, pad=0):
#    """
#    In 3D np array assuming:
#    - first axis is slice, -> axial plane index
#    - 2nd axis row in each slice, -> coronal plane index
#    - last axis is column in each slice -> sagittal plane index
#    """
#    #np.any will search for non zero elemens in the axis mentioned 
#    #So if axis=(1,2): Axial plane, it will search over axis (1=row and 2=col) and return  result
#    #for each slice - axial plane
#    nonZeroAxialPlanes = np.any(img, axis=(1, 2))
#    #So if axis=(0,2) - Corronal plane, it will search over axis (0=slice and 2=col) and return  result
#    #for corronal plane
#    nonZeroCoronalPlanes = np.any(img, axis=(0, 2))
#    #So if axis=(0,1)- sagittal plane, it will search over axis (0=slice and 1=row) and return  result
#    #for each sagittal plane
#    nonZeroSagittalPlanes = np.any(img, axis=(0, 1))
#    
#    #result from np.any(): [False  True  True  True False]
#    #result of applying np.where() : (array([ 1,2,3]),)
#    #So its a tuple of 1-D array on which one applies [0][[0, -1]]
#    #The first [0] takes the first array element out of the tuple
#    #The next [[0,-1]] is using list based indexing and getting the first and last element out
#
#    axial_min, axial_max = np.where(nonZeroAxialPlanes)[0][[0, -1]]
#    coronal_min, coronal_max = np.where(nonZeroCoronalPlanes)[0][[0, -1]]
#    sagittal_min, sagittal_max = np.where(nonZeroSagittalPlanes)[0][[0, -1]]
#    
#    if True == expandBBoxFlag:
#        axial_min = max(axial_min-pad,0)
#        axial_max = min(axial_max+pad,img.shape[0]-1)
#        coronal_min = max(coronal_min-pad,0)
#        coronal_max = min(coronal_max+pad,img.shape[1]-1)        
#        sagittal_min = max(sagittal_min-pad,0)
#        sagittal_max = min(sagittal_max+pad,img.shape[2]-1)
#    return axial_min, axial_max, coronal_min, coronal_max, sagittal_min, sagittal_max
#
#def getUnionBoundingBoxWithPadding(gt, pred, bbPad):
#    """
#    gt: volume 1; 1st dim: slice; 2nd dim: row; 3rd dim col
#    pred: volume 2 of same shape as gt
#    bbPad: Padding amount to be added over union
#    
#    return: bounding box limits (inclusive on both end)
#    """
#    #In BB calculation: a : axial, c: corronal, s : sagittal
#    #BB around  GT
#    a_min_g, a_max_g, c_min_g, c_max_g, s_min_g, s_max_g = bbox2_3D(gt, expandBBoxFlag=False, pad=0)
#    #BB around  pred
#    a_min_p, a_max_p, c_min_p, c_max_p, s_min_p, s_max_p = bbox2_3D(pred, expandBBoxFlag=False, pad=0)
#    #common BB encompassing both GT and pred  and padding added
#    a_min, a_max, c_min, c_max, s_min, s_max = \
#            min(a_min_g, a_min_p), max(a_max_g, a_max_p),\
#            min(c_min_g, c_min_p), max(c_max_g, c_max_p),\
#            min(s_min_g, s_min_p), max(s_max_g, s_max_p)   
#    #After added padding: Note both GT and Pred has the same shape
#    a_min = max(a_min-bbPad,0)
#    a_max = min(a_max+bbPad,     gt.shape[0]-1)
#    c_min = max(c_min-bbPad,0)
#    c_max = min(c_max+bbPad,   gt.shape[1]-1)        
#    s_min = max(s_min-bbPad,0)
#    s_max = min(s_max+bbPad,  gt.shape[2]-1) 
#    return a_min, a_max, c_min, c_max, s_min, s_max
#
#
#
##Choose scribble from misclassified region (3D) : input includes fraction of misclassified pixels 
## to be added as initial scribble as well as scribble-brush diameter
#def chooseScribbleFromMissedFGOrWrongCBG3D(misclassifiedRegion, boundingBoxVol, fractionDivider, dilation_diam,\
#                                           useAtmostNScribbles):
#    """
#        misclassifiedRegion : int8 binary volume of fgMissed or bgWrongC with slice as first dimension
#        boundingBoxVol:  Binary (int0, 0-1) volume within which scribbles should be limited
#        fractionDivider : positive integer by which number of one pixels in a slice will be divide
#                to decide what fraction of them will be chosen. 
#                If fractionDivider=1, all of them get chosen
#        dilation_diam: dimeter of disk : 1,2,3: disk diameter of  scribble
#        useAtmostNScribbles: Integer, If <=0, ignored 
#    """
#    resultBinary = np.zeros_like(misclassifiedRegion)
#    #Constrain the misclassified region with boundary volume
#    misclassifiedRegion *= boundingBoxVol
#    onePixelsThisVol = np.where(misclassifiedRegion)
#    onePixelCoordsThisVol = list(zip(onePixelsThisVol[0], onePixelsThisVol[1], onePixelsThisVol[2]))
#    #print(onePixelCoordsThisVol)
#    numOnePixelCoordsThisVol = len(onePixelCoordsThisVol)
#    #Debug:
#    #print('numOnePixelCoordsThisVol ', numOnePixelCoordsThisVol)
#    numScribblesFromThisVol = numOnePixelCoordsThisVol // fractionDivider
#    if int(useAtmostNScribbles) > 0 :
#        numScribblesFromThisVol = min(numScribblesFromThisVol, int(useAtmostNScribbles))
#    chosenScribbleCoordsThisVol = random.sample(onePixelCoordsThisVol, numScribblesFromThisVol)
#    #print(chosenScribbleCoordsThisVol)
#    for coord in chosenScribbleCoordsThisVol : resultBinary[coord] = 1
#    #dilate in volume       
#    #print('result before dilation ')
#    #print(resultBinary)
#    resultBinary = \
#       scipy.ndimage.binary_dilation(resultBinary,structure=ball(dilation_diam)).astype(resultBinary.dtype)
#    #print('result after dilation ')
#    #print(resultBinary)
#    #But make sure it does not go beyond original binary 
#    resultBinary = resultBinary * misclassifiedRegion
#    #print('result after clipping ')
#    #print(resultBinary)
#    #Debug
#    #print('Debug: numScrVoxelsFromMissed-3D: ', np.sum(resultBinary))
#    return resultBinary, numScribblesFromThisVol
#
##Method to choose scribble in definitely correctly idenified region (3D): 
##Its chosen from a 3D shell within definite region
#def chooseScribbleFromDefiniteRegion3D(definiteRegion,  boundingBoxVol,   fractionDivider, dilation_diam,\
#                                           useAtmostNScribbles):
#    """
#        definiteRegion : int8 binary volume of definiteRegion with slice as first dimension 
#        boundingBoxVol:  Binary (int0, 0-1) volume within which scribbles should be limited  
#        fractionDivider : positive integer by which number of one pixels in a slice will be divide
#                to decide what fraction of them will be chosen. 
#                If fractionDivider=1, all of them get chosen
#        dilation_diam: dimeter of disk : 2, 3, 4 : a diam x diam window is placed to choose scribble
#        useAtmostNScribbles: Integer, If <=0, ignored 
#    """
#    resultBinary = np.zeros_like(definiteRegion)
#    #Erode the definite region 
#    erodedRegion = \
#          scipy.ndimage.binary_erosion(definiteRegion,structure=ball(dilation_diam)).astype(definiteRegion.dtype)
#    scribbleShell = definiteRegion - erodedRegion
#    #Constrain the scribbleShell region with boundary volume
#    scribbleShell *= boundingBoxVol
#    
#    onePixelsThisVol = np.where(scribbleShell)
#    onePixelCoordsThisVol = list(zip(onePixelsThisVol[0], onePixelsThisVol[1], onePixelsThisVol[2]))
#    #print(onePixelCoordsThisVol)
#    numOnePixelCoordsThisVol = len(onePixelCoordsThisVol)
#    #Debug:
#    #print('numOnePixelCoordsThisVol ', numOnePixelCoordsThisVol)
#    numScribblesFromThisVol = numOnePixelCoordsThisVol // fractionDivider
#    if int(useAtmostNScribbles) > 0 :
#        numScribblesFromThisVol = min(numScribblesFromThisVol, int(useAtmostNScribbles))
#    chosenScribbleCoordsThisVol = random.sample(onePixelCoordsThisVol, numScribblesFromThisVol)
#    #print(chosenScribbleCoordsThisVol)
#    for coord in chosenScribbleCoordsThisVol : resultBinary[coord] = 1
#    #dilate in volume       
#    #print('result before dilation ')
#    #print(resultBinary)
#    resultBinary = \
#       scipy.ndimage.binary_dilation(resultBinary,structure=ball(dilation_diam)).astype(resultBinary.dtype)
#    #print('result after dilation ')
#    #print(resultBinary)
#    #But make sure it does not go beyond original binary 
#    resultBinary = resultBinary * scribbleShell
#    #print('result after clipping ')
#    #print(resultBinary)
#    #Debug
#    #print('Debug: numScrVoxelsFromDefinite-3D: ', np.sum(resultBinary))
#    return resultBinary, numScribblesFromThisVol
#
#
#
########################################################################
#
##function to generate  different scribble regions automatically
#def autoGenerateScribbleRegionsAndSeeds3D(gt, pred, bbPad, fractionDivider, dilation_diam,\
#                                           useAtmostNScribblesPerRegion):
#    """
#        gt : int8 binary volume of ground truth  with slice as first dimension 
#        pred : int8 binary volume of prediction  with slice as first dimension
#        bbPad: padding to be used while creating  BB 
#        fractionDivider : positive integer by which number of one pixels in a slice will be divide
#                to decide what fraction of them will be chosen. 
#                If fractionDivider=1, all of them get chosen
#        dilation_diam: dimeter of disk : 2, 3, 4 : a diam x diam window is placed to choose scribble
#        useAtmostNScribblesPerRegion: Integer, If <=0, ignored 
#        
#        Return: 
#            binLimit: axial, corronal and sagittal limits of the bounding box to be used in graphcut, 
#            bbVolume: binary bounding box volume; 0 out side, 1 inside bounding box, 
#            numFGS: number foreground seeds (in missed and definite FG region)
#            numBGS: number of backrground seeds  (in wrongly classified and definite BG region) 
#            fgScribbleFromFGMissed: foreground scribbles from missed FG region 
#            bgScribbleFromBGWrongC: background scribbles from BG region wrongly classified as FG
#            fgScribbleFromDefiniteFG: foreground scribbles from definite FG region 
#            bgScribbleFromDefiniteBG: BG scribbles from definite BG region
#            fgSeeds: union of foreground scribbles / seeds 
#            bgSeeds: union of background scribbles / seeds
#            seedsForGC: seeds for imcut() : 0: unknown, 1 FG, 2: BG
#    """   
#    #In BB calculation: a : axial, c: corronal, s : sagittal
#    a_min, a_max, c_min, c_max, s_min, s_max = getUnionBoundingBoxWithPadding(gt, pred, bbPad)
#    binLimit = [a_min, a_max, c_min, c_max, s_min, s_max]
#    bbVolume = np.zeros_like(gt)
#    #Note the +1 without which we were notmaking the highest limit 1
#    bbVolume[a_min:a_max+1, c_min:c_max+1, s_min:s_max+1]=1
#
#    gt_comp = 1-gt 
#    pred_comp = 1-pred
#    gtMinusPred = gt * pred_comp 
#    predMinusGt = pred * gt_comp
#
#    definiteBG = gt_comp # This is also definite background
#    definiteFG = gt * pred #This is definite FG
#    fgMissed = gtMinusPred  #  FG missed by classified
#    bgWrongC = predMinusGt #  BG wrongly classified as FG
#
#    #misclassifiedRegion, boundingBoxVol, fractionDivider, dilation_diam
#    fgScribbleFromFGMissed, numFGSM = chooseScribbleFromMissedFGOrWrongCBG3D(misclassifiedRegion=fgMissed, \
#      boundingBoxVol=bbVolume, fractionDivider=fractionDivider,\
#    dilation_diam=dilation_diam, useAtmostNScribbles=useAtmostNScribblesPerRegion)
#    
#    #misclassifiedRegion, boundingBoxVol, fractionDivider, dilation_diam
#    bgScribbleFromBGWrongC, numBGSM = chooseScribbleFromMissedFGOrWrongCBG3D(misclassifiedRegion=bgWrongC, \
#      boundingBoxVol=bbVolume, fractionDivider=fractionDivider,\
#    dilation_diam=dilation_diam, useAtmostNScribbles=useAtmostNScribblesPerRegion)
#    
#    #definiteRegion,  boundingBox,  dilation_diam
#    fgScribbleFromDefiniteFG, numFGSD = chooseScribbleFromDefiniteRegion3D(definiteRegion=definiteFG,\
#       boundingBoxVol=bbVolume, fractionDivider=fractionDivider,\
#    dilation_diam=dilation_diam, useAtmostNScribbles=useAtmostNScribblesPerRegion)
#    
#    #definiteRegion,  boundingBox,  dilation_diam
#    bgScribbleFromDefiniteBG, numBGSD = chooseScribbleFromDefiniteRegion3D(definiteRegion=definiteBG,\
#       boundingBoxVol=bbVolume, fractionDivider=fractionDivider,\
#    dilation_diam=dilation_diam, useAtmostNScribbles=useAtmostNScribblesPerRegion)
#    
#    numFGS = numFGSM + numFGSD
#    numBGS = numBGSM + numFGSD
#    
#    #imCut requires fgSeeds to be 1 and bgSeeds to be 2
#    #Convert scribbles into seeds for imcut() 
#    # FG and BG seeds  are supposed to be disjoint; Still clip to 1    
#    fgSeeds = np.clip(fgScribbleFromFGMissed + fgScribbleFromDefiniteFG, 0, 1)
#    bgSeeds = 2 * np.clip(bgScribbleFromBGWrongC + bgScribbleFromDefiniteBG, 0, 1)
#    #Again fgSeeds and bgSeeds are supposed to be disjoint. Still clip to 2
#    seedsForGC = np.clip(fgSeeds + bgSeeds, 0, 2)    
#        
#    return binLimit, bbVolume, numFGS, numBGS, fgScribbleFromFGMissed, bgScribbleFromBGWrongC,\
#        fgScribbleFromDefiniteFG, bgScribbleFromDefiniteBG, fgSeeds, bgSeeds, seedsForGC
#
#import imcut.pycut    
#def generateGrahcutSegmentationFromSeeds(gt, predFromNN,
#        binLimit, seedsForGC, imgForGC, segparams):
#    """
#    gt : int8 binary volume of ground truth  with slice as first dimension 
#    
#    predFromNN : int8 binary volume of prediction from Neural Net with slice as first dimension    
#    
#    binLimit: axial, corronal and sagittal limits of the bounding box to be used in graphcut, 
#    
#    seedsForGC: seeds for imcut() : 0: unknown, 1 FG, 2: BG
#    
#    imgForGC: This image may be pet or CT or the softmax map or some layer to represent the
#     multimodal combination. It is the same shape as that of gt (and predFromNN)  
#     
#    segparams: graphCut configuration  as defined in :
#    1. https://github.com/mjirik/imcut#one-component-gaussian-model-and-one-object
#    One component Gaussian model and one object
#    pairwise_alpha control the complexity of the object shape. Higher pairwise_alpha => more compact shape.
#    segparams = {
#        'method': 'graphcut',
#        "pairwise_alpha": 20,
#        'modelparams': {
#                'cvtype': 'full',
#                "params": {"covariance_type": "full", "n_components": 1},
#        },
#        "return_only_object_with_seeds": True,
#    }
#    2. https://github.com/mjirik/imcut#gaussian-mixture-distribution-model-with-extern-feature-vector-function
#    Gaussian mixture distribution model with extern feature vector function
#    segparams = {
#    'method': 'graphcut',
#    'use_boundary_penalties': False,
#    'boundary_dilatation_distance': 2,
#    'boundary_penalties_weight': 1,
#    'modelparams': {
#        'type': 'gmmsame',
#        'fv_type': "fv_extern",
#        'fv_extern': fv_function,
#        'adaptation': 'original_data',
#    }
#    'mdl_stored_file': False,
#    }
#    mdl_stored_file: if this is set, load model from file, you can see more in 
#    function test_external_fv_with_save in pycut_test.py
#    
#    Returns  segmentation result
#    """    
#    #Setup graph cut by cropping everything to bounding box limit
#    [a_min, a_max, c_min, c_max, s_min, s_max] = binLimit
#    seedsForGC = seedsForGC[a_min:a_max+1, c_min:c_max+1, s_min:s_max+1]
#    imgForGC = imgForGC[a_min:a_max+1, c_min:c_max+1, s_min:s_max+1]
#    gc = imcut.pycut.ImageGraphCut(img=imgForGC, segparams=segparams)
#    gc.set_seeds(seedsForGC)
#    gc.run()
#    #segmentationResult with 1 as FG : note original gc.segmentation returns FG as 0
#    boundedSegmentationResult = 1 - gc.segmentation
#    gcSegResult = np.zeros_like(predFromNN)
#    gcSegResult[a_min:a_max+1, c_min:c_max+1, s_min:s_max+1] = boundedSegmentationResult
#    return gcSegResult
#
#
#    
#def runAutoGCExperimentOnPatient(patientName, srcFolder, expFolder,\
#                                 patDataConfig, autoScribbleAndGCConfig, 
#                                 numExperimentsPerPat, verbose=False):
#    #check existence of destination folder
#    checkFolderExistenceAndCreate(expFolder)
#    #Transfer file into experiment directory with fixed name
#    shutil.copy(os.path.join(srcFolder, patientName+patDataConfig['ctSuffix']), \
#                os.path.join(expFolder, patDataConfig['expPatName']+patDataConfig['ctSuffix']))
#    shutil.copy(os.path.join(srcFolder, patientName+patDataConfig['ptSuffix']), \
#                os.path.join(expFolder, patDataConfig['expPatName']+patDataConfig['ptSuffix']))
#    shutil.copy(os.path.join(srcFolder, patientName+patDataConfig['gtSuffix']), \
#                os.path.join(expFolder, patDataConfig['expPatName']+patDataConfig['gtSuffix']))
#    shutil.copy(os.path.join(srcFolder, patientName+patDataConfig['predSuffix']), \
#                os.path.join(expFolder, patDataConfig['expPatName']+patDataConfig['predSuffix']))
#    shutil.copy(os.path.join(srcFolder, patientName+patDataConfig['softmaxSuffix']), \
#                os.path.join(expFolder, patDataConfig['expPatName']+patDataConfig['softmaxSuffix']))    
#
#    #Read np ndarray from the files, to scale or not to scale? 
#    ctFileName = patDataConfig['expPatName']+patDataConfig['ctSuffix']
#    ctData = readAndScaleImageData(fileName=ctFileName,\
#        folderName=expFolder, clipFlag = patDataConfig['ct_clipImages'],\
#        clipLow=patDataConfig['ct_low'], clipHigh =patDataConfig['ct_high'],\
#        scaleFlag=patDataConfig['ct_scaleImages'], scaleFactor=patDataConfig['ct_scaleFactor'],\
#        meanSDNormalizeFlag = patDataConfig['ct_normalizeImages'], finalDataType = np.float32,\
#        isLabelData=False, labels_to_train_list=None, verbose=verbose)
#    
#    ptFileName = patDataConfig['expPatName']+patDataConfig['ptSuffix']
#    ptData = readAndScaleImageData(fileName=ptFileName,\
#        folderName=expFolder, clipFlag = patDataConfig['pt_clipImages'],\
#        clipLow=patDataConfig['pt_low'], clipHigh =patDataConfig['pt_high'],\
#        scaleFlag=patDataConfig['pt_scaleImages'], scaleFactor=patDataConfig['pt_scaleFactor'],\
#        meanSDNormalizeFlag = patDataConfig['pt_normalizeImages'], finalDataType = np.float32,\
#        isLabelData=False, labels_to_train_list=None, verbose=verbose)
#    
#    gtFileName = patDataConfig['expPatName']+patDataConfig['gtSuffix']
#    gtData = readAndScaleImageData(fileName=gtFileName,\
#        folderName=expFolder, clipFlag = False,\
#        clipLow=0, clipHigh = 0,\
#        scaleFlag=False, scaleFactor=1, meanSDNormalizeFlag = False, finalDataType = np.int8, \
#        isLabelData=True, labels_to_train_list=patDataConfig['labels_to_train'], verbose=verbose)
#    
#    predFileName = patDataConfig['expPatName']+patDataConfig['predSuffix']
#    predData = readAndScaleImageData(fileName=predFileName,\
#        folderName=expFolder, clipFlag = False,\
#        clipLow=0, clipHigh = 0,\
#        scaleFlag=False, scaleFactor=1, meanSDNormalizeFlag = False, finalDataType = np.int8, \
#        isLabelData=True, labels_to_train_list=patDataConfig['labels_to_train'], verbose=verbose)
#    
#    softmaxFileName = patDataConfig['expPatName']+patDataConfig['softmaxSuffix']
#    softmaxData = readAndScaleImageData(fileName=softmaxFileName,\
#        folderName=expFolder, clipFlag = False,\
#        clipLow=0.0, clipHigh =1.0,\
#        scaleFlag=False, scaleFactor=1.0, meanSDNormalizeFlag = False, finalDataType = np.float32,\
#        isLabelData=False, labels_to_train_list=None, verbose=verbose)
#   
#    experimentResultDetail = []
#    dice_org = dice_multi_label(predData, gtData)[0]
#    for expId in range(numExperimentsPerPat):
#        binLimit, bbVolume, numFGS, numBGS, fgScribbleFromFGMissed, bgScribbleFromBGWrongC,\
#            fgScribbleFromDefiniteFG, bgScribbleFromDefiniteBG, fgSeeds, bgSeeds, seedsForGC \
#            = autoGenerateScribbleRegionsAndSeeds3D(gt=gtData, pred=predData,\
#                bbPad=autoScribbleAndGCConfig['bbPad'],fractionDivider=autoScribbleAndGCConfig['fractionDivider'],\
#                dilation_diam=autoScribbleAndGCConfig['dilation_diam'],\
#                useAtmostNScribblesPerRegion=autoScribbleAndGCConfig['useAtmostNScribblesPerRegion']) 
#        
#        gcSegResult_softmax =  generateGrahcutSegmentationFromSeeds(\
#                gt=gtData, predFromNN=predData, binLimit=binLimit, seedsForGC=seedsForGC,\
#                imgForGC=softmaxData, segparams=autoScribbleAndGCConfig['segparams_ssgc'])
#        gcSegResult_ct =  generateGrahcutSegmentationFromSeeds(\
#                gt=gtData, predFromNN=predData, binLimit=binLimit, seedsForGC=seedsForGC,\
#                imgForGC=ctData, segparams=autoScribbleAndGCConfig['segparams_ssgc'])
#        gcSegResult_pet =  generateGrahcutSegmentationFromSeeds(\
#                gt=gtData, predFromNN=predData, binLimit=binLimit, seedsForGC=seedsForGC,\
#                imgForGC=ptData, segparams=autoScribbleAndGCConfig['segparams_ssgc'])
#            
#        dice_gc_softmax = dice_multi_label(gcSegResult_softmax, gtData)[0]
#        dice_gc_ct = dice_multi_label(gcSegResult_ct, gtData)[0]
#        dice_gc_pet = dice_multi_label(gcSegResult_pet, gtData)[0]
#        
#        # print('#FGScrb: ', numFGS, ' #BGScrb: ', numBGS,\
#        #       'd_org: ', round(dice_org,3), ' d_softmax: ', round(dice_gc_softmax,3),\
#        #       ' d_ct: ', round(dice_gc_ct,3), ' d_pet: ', round(dice_gc_pet,3))
#        experimentResultDetail.append([numFGS, numBGS, dice_org, dice_gc_softmax, dice_gc_ct, dice_gc_pet])
#    #create data frame
#    # Create the pandas DataFrame 
#    expResult_df = pd.DataFrame(experimentResultDetail, columns = ['#FGScrb', ' #BGScrb', 'd_org', ' d_softmax', ' d_ct', ' d_pet'])
#    return expResult_df
#
## ########### Test Code #############        
## #Run experiment on a patient
#
##Save / Read from json file
#patDataConfig = \
#{
# 'ctSuffix' :'_ct.nii.gz',
# 'ptSuffix': '_pt.nii.gz',
# 'gtSuffix': '_ct_gtvt.nii.gz',
# 'predSuffix' : '_segment.nii.gz',
# 'softmaxSuffix' : '_softMax.nii.gz',
# 'expPatName' : 'expPat',
# 'ct_low': -1000,
# 'ct_high': 3095,
# 'ct_clipImages' : False,
# 'ct_scaleImages' : False,
# 'ct_scaleFactor' : 1000,
# 'ct_normalizeImages' : False,
# 'pt_low' : -1000,
# 'pt_high' : 3095,
# 'pt_clipImages' : False,
# 'pt_scaleImages' : False,
# 'pt_scaleFactor' : 1,
# 'pt_normalizeImages': False,
# 'labels_to_train' : [1]
# }
#
#autoScribbleAndGCConfig = \
#{
# 'bbPad' : 2,
# 'fractionDivider' : 10,
# 'dilation_diam' : 2,
# 'useAtmostNScribblesPerRegion' : 4,
# 'segparams_ssgc' : 
# {
#  'method': 'graphcut',
#  'pairwise_alpha': 1.0,
#  'modelparams': 
#   {
#    'cvtype': 'full',
#    'params': {
#               'covariance_type': 'full', 
#               'n_components': 1
#              },
#   },
#   'return_only_object_with_seeds': True,
# },
# # 'segparams_ssgc2' : 
# # {
# #  'method': 'graphcut',
# #  'use_boundary_penalties': False,
# #  'boundary_dilatation_distance': 2,
# #  'boundary_penalties_weight': 1,
# #  'modelparams': 
# #  {
# #   'type': 'gmmsame',
# #   'params': 
# #    {
# #     'n_components': 2
# #    },
# #    'return_only_object_with_seeds': True,
# #    # 'fv_type': 'fv_extern',
# #    # 'fv_extern': fv_function,
# #    # 'adaptation': 'original_data',
# #  },
# # }     
#}
#    
############ Test Code #############        
##Run experiment on a patient
#
#srcFolder =\
#  '/home/user/DMML/Data/HeadNeck_PET_CT/nnUnet_3dfullres/validation_gtvs_withSoftmax'
#expFolder = '/home/user/DMML/Data/PlayDataManualSegmentation/AutoScribbleExperiment'
#numExperimentsPerPat = 15
#verbose = False
#medianExpResultList = []
##listOfPatients = ['CHGJ017', 'CHUM038', 'CHGJ008', 'CHUM002',  'CHUS041', 'CHUS067']
#listOfPatients = [(os.path.basename(f)).replace('_ct.nii.gz','') \
#      for f in glob.glob(srcFolder + '/*_ct.nii.gz', recursive=False) ]
##print(listOfPatients)
##patientName = 'CHUM038'
#for patientName in listOfPatients:
#    print('Experimenting graphCut on ', patientName)
#    expResult_df = runAutoGCExperimentOnPatient(patientName, srcFolder, expFolder,\
#                                      patDataConfig, autoScribbleAndGCConfig, 
#                                      numExperimentsPerPat, verbose)
#    #print(expResult_df)
#    medianExpResultForPatient = expResult_df.median(axis = 0) 
#    #print(medianExpResultForPatient) 
#    medianExpResultForPatientList = [patientName] + \
#        [medianExpResultForPatient[id] for id in range(len(medianExpResultForPatient))]
#    medianExpResultList.append(medianExpResultForPatientList)
#
##Result over patients
#medianExpResult_df = pd.DataFrame(medianExpResultList,\
#      columns = ['patientName', '#FGScrb', ' #BGScrb', 'd_org', 'd_softmax', 'd_ct', 'd_pet'])
## print(medianExpResult_df)
## medianExpResult_df.to_csv('/home/user/DMML/Data/PlayDataManualSegmentation/AutoScribbleExperiment/medianExpResult_df.csv', index = False)
#
#margin = 0.1
#softMaxGCImprovedOverOrg = medianExpResult_df['d_softmax'] > medianExpResult_df['d_org'] + margin
#softMaxGCWorenedBelowOrg = medianExpResult_df['d_softmax'] < medianExpResult_df['d_org'] - margin
#
#ctBest =  medianExpResult_df['d_ct'] == medianExpResult_df[['d_softmax', 'd_ct', 'd_pet']].max(axis=1)
#petBest =  medianExpResult_df['d_pet'] == medianExpResult_df[['d_softmax', 'd_ct', 'd_pet']].max(axis=1)
#softmaxBest =  medianExpResult_df['d_softmax'] == medianExpResult_df[['d_softmax', 'd_ct', 'd_pet']].max(axis=1)
#
## ctBest                   = medianExpResult_df['d_ct'] > medianExpResult_df['d_softmax'] and \
##                            medianExpResult_df['d_ct'] > medianExpResult_df['d_pet']
## petBest                  = medianExpResult_df['d_pet'] > medianExpResult_df['d_softmax'] and \
##                            medianExpResult_df['d_pet'] > medianExpResult_df['d_ct']
## softmaxBest              = medianExpResult_df['d_softmax'] > medianExpResult_df['d_ct'] and \
##                            medianExpResult_df['d_softmax'] > medianExpResult_df['d_pet']
#
## ctGCBetterThanSoftMaxGC = medianExpResult_df['d_ct'] > medianExpResult_df['d_softmax'] + margin
## ptGCBetterThanSoftMaxGC = medianExpResult_df['d_pet'] > medianExpResult_df['d_softmax'] + margin
## softmaxGCBetterThanCTGC= medianExpResult_df['d_softmax'] > medianExpResult_df['d_ct'] + margin
## softmaxGCBetterThanPETGC = medianExpResult_df['d_softmax'] > medianExpResult_df['d_pet'] + margin
#
#nP_softMaxGCImprovedOverOrg = len(medianExpResult_df[softMaxGCImprovedOverOrg])
#nP_softMaxGCWorsenedBelowOrg = len(medianExpResult_df[softMaxGCWorenedBelowOrg])
#nP_ctBest = len(medianExpResult_df[ctBest])
#nP_petBest = len(medianExpResult_df[petBest])
#nP_softmaxBest = len(medianExpResult_df[softmaxBest])
#
## nP_ctGCBetterThanSoftMaxGC = len(medianExpResult_df[ctGCBetterThanSoftMaxGC])
## nP_ptGCBetterThanSoftMaxGC = len(medianExpResult_df[ptGCBetterThanSoftMaxGC])
## nP_softmaxGCBetterThanCTGC= len(medianExpResult_df[softmaxGCBetterThanCTGC])
## nP_softmaxGCBetterThanPETGC = len(medianExpResult_df[softmaxGCBetterThanPETGC])
#
#print('numPatients', len(medianExpResult_df))
#print('Improvement / Worsening margin used ',  margin)
#print('nP_softMaxGCImprovedOverOrg', nP_softMaxGCImprovedOverOrg)
#print('nP_softMaxGCWorsenedBelowOrg', nP_softMaxGCWorsenedBelowOrg)
#print('nP_ctBest', nP_ctBest)
#print('nP_petBest', nP_petBest)
#print('nP_softmaxBest', nP_softmaxBest)
#
## print('nP_ctGCBetterThanSoftMaxGC', nP_ctGCBetterThanSoftMaxGC)
## print('nP_ptGCBetterThanSoftMaxGC', nP_ptGCBetterThanSoftMaxGC)
## print('nP_softmaxGCBetterThanCTGC', nP_softmaxGCBetterThanCTGC)
## print('nP_softmaxGCBetterThanPETGC', nP_softmaxGCBetterThanPETGC)
#
#
#pass     
##########################################################################################
