# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import glob
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



#######################################################################

#function to generate  different scribble regions automatically
def autoGenerateScribbleRegionsAndSeeds3D(gt, pred, bbPad, fractionDivider, dilation_diam,\
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
            numFGS: number foreground seeds (in missed and definite FG region)
            numBGS: number of backrground seeds  (in wrongly classified and definite BG region) 
            fgScribbleFromFGMissed: foreground scribbles from missed FG region 
            bgScribbleFromBGWrongC: background scribbles from BG region wrongly classified as FG
            fgScribbleFromDefiniteFG: foreground scribbles from definite FG region 
            bgScribbleFromDefiniteBG: BG scribbles from definite BG region
            fgSeeds: union of foreground scribbles / seeds 
            bgSeeds: union of background scribbles / seeds
            seedsForGC: seeds for imcut() : 0: unknown, 1 FG, 2: BG
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
    numBGS = numBGSM + numFGSD
    
    #imCut requires fgSeeds to be 1 and bgSeeds to be 2
    #Convert scribbles into seeds for imcut() 
    # FG and BG seeds  are supposed to be disjoint; Still clip to 1    
    fgSeeds = np.clip(fgScribbleFromFGMissed + fgScribbleFromDefiniteFG, 0, 1)
    bgSeeds = 2 * np.clip(bgScribbleFromBGWrongC + bgScribbleFromDefiniteBG, 0, 1)
    #Again fgSeeds and bgSeeds are supposed to be disjoint. Still clip to 2
    seedsForGC = np.clip(fgSeeds + bgSeeds, 0, 2)    
        
    return binLimit, bbVolume, numFGS, numBGS, fgScribbleFromFGMissed, bgScribbleFromBGWrongC,\
        fgScribbleFromDefiniteFG, bgScribbleFromDefiniteBG, fgSeeds, bgSeeds, seedsForGC

import imcut.pycut    
def generateGrahcutSegmentationFromSeeds(gt, predFromNN,
        binLimit, seedsForGC, imgForGC, segparams):
    """
    gt : int8 binary volume of ground truth  with slice as first dimension 
    
    predFromNN : int8 binary volume of prediction from Neural Net with slice as first dimension    
    
    binLimit: axial, corronal and sagittal limits of the bounding box to be used in graphcut, 
    
    seedsForGC: seeds for imcut() : 0: unknown, 1 FG, 2: BG
    
    imgForGC: This image may be pet or CT or the softmax map or some layer to represent the
     multimodal combination. It is the same shape as that of gt (and predFromNN)  
     
    segparams: graphCut configuration  as defined in :
    1. https://github.com/mjirik/imcut#one-component-gaussian-model-and-one-object
    One component Gaussian model and one object
    pairwise_alpha control the complexity of the object shape. Higher pairwise_alpha => more compact shape.
    segparams = {
        'method': 'graphcut',
        "pairwise_alpha": 20,
        'modelparams': {
                'cvtype': 'full',
                "params": {"covariance_type": "full", "n_components": 1},
        },
        "return_only_object_with_seeds": True,
    }
    2. https://github.com/mjirik/imcut#gaussian-mixture-distribution-model-with-extern-feature-vector-function
    Gaussian mixture distribution model with extern feature vector function
    segparams = {
    'method': 'graphcut',
    'use_boundary_penalties': False,
    'boundary_dilatation_distance': 2,
    'boundary_penalties_weight': 1,
    'modelparams': {
        'type': 'gmmsame',
        'fv_type': "fv_extern",
        'fv_extern': fv_function,
        'adaptation': 'original_data',
    }
    'mdl_stored_file': False,
    }
    mdl_stored_file: if this is set, load model from file, you can see more in 
    function test_external_fv_with_save in pycut_test.py
    
    Returns  segmentation result
    """    
    #Setup graph cut by cropping everything to bounding box limit
    [a_min, a_max, c_min, c_max, s_min, s_max] = binLimit
    seedsForGC = seedsForGC[a_min:a_max+1, c_min:c_max+1, s_min:s_max+1]
    imgForGC = imgForGC[a_min:a_max+1, c_min:c_max+1, s_min:s_max+1]
    gc = imcut.pycut.ImageGraphCut(img=imgForGC, segparams=segparams)
    gc.set_seeds(seedsForGC)
    gc.run()
    #segmentationResult with 1 as FG : note original gc.segmentation returns FG as 0
    boundedSegmentationResult = 1 - gc.segmentation
    gcSegResult = np.zeros_like(predFromNN)
    gcSegResult[a_min:a_max+1, c_min:c_max+1, s_min:s_max+1] = boundedSegmentationResult
    return gcSegResult


    
def runAutoGCExperimentOnPatient(patientName, srcFolder, expFolder,\
                                 patDataConfig, autoScribbleAndGCConfig, 
                                 numExperimentsPerPat, verbose=False):
    #check existence of destination folder
    checkFolderExistenceAndCreate(expFolder)
    #Transfer file into experiment directory with fixed name
    shutil.copy(os.path.join(srcFolder, patientName+patDataConfig['ctSuffix']), \
                os.path.join(expFolder, patDataConfig['expPatName']+patDataConfig['ctSuffix']))
    shutil.copy(os.path.join(srcFolder, patientName+patDataConfig['ptSuffix']), \
                os.path.join(expFolder, patDataConfig['expPatName']+patDataConfig['ptSuffix']))
    shutil.copy(os.path.join(srcFolder, patientName+patDataConfig['gtSuffix']), \
                os.path.join(expFolder, patDataConfig['expPatName']+patDataConfig['gtSuffix']))
    shutil.copy(os.path.join(srcFolder, patientName+patDataConfig['predSuffix']), \
                os.path.join(expFolder, patDataConfig['expPatName']+patDataConfig['predSuffix']))
    shutil.copy(os.path.join(srcFolder, patientName+patDataConfig['softmaxSuffix']), \
                os.path.join(expFolder, patDataConfig['expPatName']+patDataConfig['softmaxSuffix']))    

    #Read np ndarray from the files, to scale or not to scale? 
    ctFileName = patDataConfig['expPatName']+patDataConfig['ctSuffix']
    ctData = readAndScaleImageData(fileName=ctFileName,\
        folderName=expFolder, clipFlag = patDataConfig['ct_clipImages'],\
        clipLow=patDataConfig['ct_low'], clipHigh =patDataConfig['ct_high'],\
        scaleFlag=patDataConfig['ct_scaleImages'], scaleFactor=patDataConfig['ct_scaleFactor'],\
        meanSDNormalizeFlag = patDataConfig['ct_normalizeImages'], finalDataType = np.float32,\
        isLabelData=False, labels_to_train_list=None, verbose=verbose)
    
    ptFileName = patDataConfig['expPatName']+patDataConfig['ptSuffix']
    ptData = readAndScaleImageData(fileName=ptFileName,\
        folderName=expFolder, clipFlag = patDataConfig['pt_clipImages'],\
        clipLow=patDataConfig['pt_low'], clipHigh =patDataConfig['pt_high'],\
        scaleFlag=patDataConfig['pt_scaleImages'], scaleFactor=patDataConfig['pt_scaleFactor'],\
        meanSDNormalizeFlag = patDataConfig['pt_normalizeImages'], finalDataType = np.float32,\
        isLabelData=False, labels_to_train_list=None, verbose=verbose)
    
    gtFileName = patDataConfig['expPatName']+patDataConfig['gtSuffix']
    gtData = readAndScaleImageData(fileName=gtFileName,\
        folderName=expFolder, clipFlag = False,\
        clipLow=0, clipHigh = 0,\
        scaleFlag=False, scaleFactor=1, meanSDNormalizeFlag = False, finalDataType = np.int8, \
        isLabelData=True, labels_to_train_list=patDataConfig['labels_to_train'], verbose=verbose)
    
    predFileName = patDataConfig['expPatName']+patDataConfig['predSuffix']
    predData = readAndScaleImageData(fileName=predFileName,\
        folderName=expFolder, clipFlag = False,\
        clipLow=0, clipHigh = 0,\
        scaleFlag=False, scaleFactor=1, meanSDNormalizeFlag = False, finalDataType = np.int8, \
        isLabelData=True, labels_to_train_list=patDataConfig['labels_to_train'], verbose=verbose)
    
    softmaxFileName = patDataConfig['expPatName']+patDataConfig['softmaxSuffix']
    softmaxData = readAndScaleImageData(fileName=softmaxFileName,\
        folderName=expFolder, clipFlag = False,\
        clipLow=0.0, clipHigh =1.0,\
        scaleFlag=False, scaleFactor=1.0, meanSDNormalizeFlag = False, finalDataType = np.float32,\
        isLabelData=False, labels_to_train_list=None, verbose=verbose)
   
    experimentResultDetail = []
    dice_org = dice_multi_label(predData, gtData)[0]
    for expId in range(numExperimentsPerPat):
        binLimit, bbVolume, numFGS, numBGS, fgScribbleFromFGMissed, bgScribbleFromBGWrongC,\
            fgScribbleFromDefiniteFG, bgScribbleFromDefiniteBG, fgSeeds, bgSeeds, seedsForGC \
            = autoGenerateScribbleRegionsAndSeeds3D(gt=gtData, pred=predData,\
                bbPad=autoScribbleAndGCConfig['bbPad'],fractionDivider=autoScribbleAndGCConfig['fractionDivider'],\
                dilation_diam=autoScribbleAndGCConfig['dilation_diam'],\
                useAtmostNScribblesPerRegion=autoScribbleAndGCConfig['useAtmostNScribblesPerRegion']) 
        
        gcSegResult_softmax =  generateGrahcutSegmentationFromSeeds(\
                gt=gtData, predFromNN=predData, binLimit=binLimit, seedsForGC=seedsForGC,\
                imgForGC=softmaxData, segparams=autoScribbleAndGCConfig['segparams_ssgc'])
        gcSegResult_ct =  generateGrahcutSegmentationFromSeeds(\
                gt=gtData, predFromNN=predData, binLimit=binLimit, seedsForGC=seedsForGC,\
                imgForGC=ctData, segparams=autoScribbleAndGCConfig['segparams_ssgc'])
        gcSegResult_pet =  generateGrahcutSegmentationFromSeeds(\
                gt=gtData, predFromNN=predData, binLimit=binLimit, seedsForGC=seedsForGC,\
                imgForGC=ptData, segparams=autoScribbleAndGCConfig['segparams_ssgc'])
            
        dice_gc_softmax = dice_multi_label(gcSegResult_softmax, gtData)[0]
        dice_gc_ct = dice_multi_label(gcSegResult_ct, gtData)[0]
        dice_gc_pet = dice_multi_label(gcSegResult_pet, gtData)[0]
        
        # print('#FGScrb: ', numFGS, ' #BGScrb: ', numBGS,\
        #       'd_org: ', round(dice_org,3), ' d_softmax: ', round(dice_gc_softmax,3),\
        #       ' d_ct: ', round(dice_gc_ct,3), ' d_pet: ', round(dice_gc_pet,3))
        experimentResultDetail.append([numFGS, numBGS, dice_org, dice_gc_softmax, dice_gc_ct, dice_gc_pet])
    #create data frame
    # Create the pandas DataFrame 
    expResult_df = pd.DataFrame(experimentResultDetail, columns = ['#FGScrb', ' #BGScrb', 'd_org', ' d_softmax', ' d_ct', ' d_pet'])
    return expResult_df

# ########### Test Code #############        
# #Run experiment on a patient

#Save / Read from json file
patDataConfig = \
{
 'ctSuffix' :'_ct.nii.gz',
 'ptSuffix': '_pt.nii.gz',
 'gtSuffix': '_ct_gtvt.nii.gz',
 'predSuffix' : '_segment.nii.gz',
 'softmaxSuffix' : '_softMax.nii.gz',
 'expPatName' : 'expPat',
 'ct_low': -1000,
 'ct_high': 3095,
 'ct_clipImages' : False,
 'ct_scaleImages' : False,
 'ct_scaleFactor' : 1000,
 'ct_normalizeImages' : False,
 'pt_low' : -1000,
 'pt_high' : 3095,
 'pt_clipImages' : False,
 'pt_scaleImages' : False,
 'pt_scaleFactor' : 1,
 'pt_normalizeImages': False,
 'labels_to_train' : [1]
 }

autoScribbleAndGCConfig = \
{
 'bbPad' : 2,
 'fractionDivider' : 10,
 'dilation_diam' : 2,
 'useAtmostNScribblesPerRegion' : 4,
 'segparams_ssgc' : 
 {
  'method': 'graphcut',
  'pairwise_alpha': 1.0,
  'modelparams': 
   {
    'cvtype': 'full',
    'params': {
               'covariance_type': 'full', 
               'n_components': 1
              },
   },
   'return_only_object_with_seeds': True,
 },
 # 'segparams_ssgc2' : 
 # {
 #  'method': 'graphcut',
 #  'use_boundary_penalties': False,
 #  'boundary_dilatation_distance': 2,
 #  'boundary_penalties_weight': 1,
 #  'modelparams': 
 #  {
 #   'type': 'gmmsame',
 #   'params': 
 #    {
 #     'n_components': 2
 #    },
 #    'return_only_object_with_seeds': True,
 #    # 'fv_type': 'fv_extern',
 #    # 'fv_extern': fv_function,
 #    # 'adaptation': 'original_data',
 #  },
 # }     
}
    
########### Test Code #############        
#Run experiment on a patient

srcFolder =\
  '/home/user/DMML/Data/HeadNeck_PET_CT/nnUnet_3dfullres/validation_gtvs_withSoftmax'
expFolder = '/home/user/DMML/Data/PlayDataManualSegmentation/AutoScribbleExperiment'
numExperimentsPerPat = 15
verbose = False
medianExpResultList = []
#listOfPatients = ['CHGJ017', 'CHUM038', 'CHGJ008', 'CHUM002',  'CHUS041', 'CHUS067']
listOfPatients = [(os.path.basename(f)).replace('_ct.nii.gz','') \
      for f in glob.glob(srcFolder + '/*_ct.nii.gz', recursive=False) ]
#print(listOfPatients)
#patientName = 'CHUM038'
for patientName in listOfPatients:
    print('Experimenting graphCut on ', patientName)
    expResult_df = runAutoGCExperimentOnPatient(patientName, srcFolder, expFolder,\
                                      patDataConfig, autoScribbleAndGCConfig, 
                                      numExperimentsPerPat, verbose)
    #print(expResult_df)
    medianExpResultForPatient = expResult_df.median(axis = 0) 
    #print(medianExpResultForPatient) 
    medianExpResultForPatientList = [patientName] + \
        [medianExpResultForPatient[id] for id in range(len(medianExpResultForPatient))]
    medianExpResultList.append(medianExpResultForPatientList)

#Result over patients
medianExpResult_df = pd.DataFrame(medianExpResultList,\
      columns = ['patientName', '#FGScrb', ' #BGScrb', 'd_org', 'd_softmax', 'd_ct', 'd_pet'])
print(medianExpResult_df)

margin = 0.1
softMaxGCImprovedOverOrg = medianExpResult_df['d_softmax'] > medianExpResult_df['d_org'] + margin
softMaxGCWorenedBelowOrg = medianExpResult_df['d_softmax'] < medianExpResult_df['d_org'] - margin

ctBest =  medianExpResult_df['d_ct'] == medianExpResult_df[['d_softmax', 'd_ct', 'd_pet']].max(axis=1)
petBest =  medianExpResult_df['d_pet'] == medianExpResult_df[['d_softmax', 'd_ct', 'd_pet']].max(axis=1)
softmaxBest =  medianExpResult_df['d_softmax'] == medianExpResult_df[['d_softmax', 'd_ct', 'd_pet']].max(axis=1)

# ctBest                   = medianExpResult_df['d_ct'] > medianExpResult_df['d_softmax'] and \
#                            medianExpResult_df['d_ct'] > medianExpResult_df['d_pet']
# petBest                  = medianExpResult_df['d_pet'] > medianExpResult_df['d_softmax'] and \
#                            medianExpResult_df['d_pet'] > medianExpResult_df['d_ct']
# softmaxBest              = medianExpResult_df['d_softmax'] > medianExpResult_df['d_ct'] and \
#                            medianExpResult_df['d_softmax'] > medianExpResult_df['d_pet']

# ctGCBetterThanSoftMaxGC = medianExpResult_df['d_ct'] > medianExpResult_df['d_softmax'] + margin
# ptGCBetterThanSoftMaxGC = medianExpResult_df['d_pet'] > medianExpResult_df['d_softmax'] + margin
# softmaxGCBetterThanCTGC= medianExpResult_df['d_softmax'] > medianExpResult_df['d_ct'] + margin
# softmaxGCBetterThanPETGC = medianExpResult_df['d_softmax'] > medianExpResult_df['d_pet'] + margin

nP_softMaxGCImprovedOverOrg = len(medianExpResult_df[softMaxGCImprovedOverOrg])
nP_softMaxGCWorsenedBelowOrg = len(medianExpResult_df[softMaxGCWorenedBelowOrg])
nP_ctBest = len(medianExpResult_df[ctBest])
nP_petBest = len(medianExpResult_df[petBest])
nP_softmaxBest = len(medianExpResult_df[softmaxBest])

# nP_ctGCBetterThanSoftMaxGC = len(medianExpResult_df[ctGCBetterThanSoftMaxGC])
# nP_ptGCBetterThanSoftMaxGC = len(medianExpResult_df[ptGCBetterThanSoftMaxGC])
# nP_softmaxGCBetterThanCTGC= len(medianExpResult_df[softmaxGCBetterThanCTGC])
# nP_softmaxGCBetterThanPETGC = len(medianExpResult_df[softmaxGCBetterThanPETGC])

print('numPatients', len(medianExpResult_df))
print('Improvement / Worsening margin used ',  margin)
print('nP_softMaxGCImprovedOverOrg', nP_softMaxGCImprovedOverOrg)
print('nP_softMaxGCWorsenedBelowOrg', nP_softMaxGCWorsenedBelowOrg)
print('nP_ctBest', nP_ctBest)
print('nP_petBest', nP_petBest)
print('nP_softmaxBest', nP_softmaxBest)

# print('nP_ctGCBetterThanSoftMaxGC', nP_ctGCBetterThanSoftMaxGC)
# print('nP_ptGCBetterThanSoftMaxGC', nP_ptGCBetterThanSoftMaxGC)
# print('nP_softmaxGCBetterThanCTGC', nP_softmaxGCBetterThanCTGC)
# print('nP_softmaxGCBetterThanPETGC', nP_softmaxGCBetterThanPETGC)
#########################################
# ctSuffix = '_ct.nii.gz'
# ptSuffix = '_pt.nii.gz'
# gtSuffix = '_ct_gtvt.nii.gz'
# predSuffix = '_segment.nii.gz'
# softmaxSuffix = '_softMax.nii.gz'
# expPatName = 'expPat'

# ct_low = -1000 
# ct_high = 3095 
# pt_low = 0.0 
# pt_high =  20.0 
# labels_to_train = [1]
# clipImages=False
# scaleImages=False
# normalizeImages=False
# verbose=True


# #Following code will go into a function

# #check existence of destination folder
# checkFolderExistenceAndCreate(expFolder)
# #Transfer file into experiment directory with fixed name
# shutil.copy(os.path.join(srcFolder, patientName+ctSuffix), \
#             os.path.join(expFolder, expPatName+ctSuffix))
# shutil.copy(os.path.join(srcFolder, patientName+ptSuffix), \
#             os.path.join(expFolder, expPatName+ptSuffix))
# shutil.copy(os.path.join(srcFolder, patientName+patDataConfig['gtSuffix']), \
#             os.path.join(expFolder, expPatName+patDataConfig['gtSuffix']))
# shutil.copy(os.path.join(srcFolder, patientName+predSuffix), \
#             os.path.join(expFolder, expPatName+predSuffix))
# shutil.copy(os.path.join(srcFolder, patientName+softmaxSuffix), \
#             os.path.join(expFolder, expPatName+softmaxSuffix))

# #Read np ndarray from the files, to scale or not to scale? 
# ctFileName = patDataConfig['expPatName']+patDataConfig['ctSuffix']
# ctData = readAndScaleImageData(fileName=ctFileName,\
#     folderName=expFolder, clipFlag = clipImages,\
#     clipLow=ct_low, clipHigh =ct_high,\
#     scaleFlag=scaleImages, scaleFactor=1000, meanSDNormalizeFlag = normalizeImages, finalDataType = np.float32,\
#     isLabelData=False, labels_to_train_list=None, verbose=verbose)

# ptFileName = patDataConfig['expPatName']+patDataConfig['ptSuffix']
# ptData = readAndScaleImageData(fileName=ptFileName,\
#     folderName=expFolder, clipFlag = clipImages,\
#     clipLow=pt_low, clipHigh =pt_high,\
#     scaleFlag=scaleImages, scaleFactor=1, meanSDNormalizeFlag = normalizeImages, finalDataType = np.float32,\
#     isLabelData=False, labels_to_train_list=None, verbose=verbose)

# gtFileName = patDataConfig['expPatName']+patDataConfig['gtSuffix']
# gtData = readAndScaleImageData(fileName=gtFileName,\
#     folderName=expFolder, clipFlag = False,\
#     clipLow=0, clipHigh = 0,\
#     scaleFlag=False, scaleFactor=1, meanSDNormalizeFlag = False, finalDataType = np.int8, \
#     isLabelData=True, labels_to_train_list=labels_to_train, verbose=verbose)

# predFileName = patDataConfig['expPatName']+patDataConfig['predSuffix']
# predData = readAndScaleImageData(fileName=predFileName,\
#     folderName=expFolder, clipFlag = False,\
#     clipLow=0, clipHigh = 0,\
#     scaleFlag=False, scaleFactor=1, meanSDNormalizeFlag = False, finalDataType = np.int8, \
#     isLabelData=True, labels_to_train_list=labels_to_train, verbose=verbose)

# softmaxFileName = patDataConfig['expPatName']+patDataConfig['softmaxSuffix']
# softmaxData = readAndScaleImageData(fileName=softmaxFileName,\
#     folderName=expFolder, clipFlag = False,\
#     clipLow=0.0, clipHigh =1.0,\
#     scaleFlag=False, scaleFactor=1.0, meanSDNormalizeFlag = False, finalDataType = np.float32,\
#     isLabelData=False, labels_to_train_list=None, verbose=verbose)

# #Testing graph cut
# #One component Gaussian model and one object
# #pairwise_alpha control the complexity of the object shape. Higher pairwise_alpha => more compact shape.
# segparams_ssgc = {
#     'method': 'graphcut',
#     "pairwise_alpha": 1.0,
#     'modelparams': {
#             'cvtype': 'full',
#             "params": {"covariance_type": "full", "n_components": 1},
#     },
#     "return_only_object_with_seeds": True,
# }

# # segparams_ssgc = {
# #     # 'method':'graphcut',
# #     "method": "graphcut",
# #     "use_boundary_penalties": False,
# #     "boundary_dilatation_distance": 2,
# #     "boundary_penalties_weight": 1,
# #     "modelparams": {
# #         "type": "gmmsame",
# #         "params": {"n_components": 2},
# #         # "return_only_object_with_seeds": True,
# #         # 'fv_type': "fv_extern",
# #         # 'fv_extern': fv_function,
# #         # 'adaptation': 'original_data',
# #     },
# # }

# bbPad=2
# fractionDivider=10
# dilation_diam=2
# useAtmostNScribblesPerRegion=4

# binLimit, bbVolume, numFGS, numBGS, fgScribbleFromFGMissed, bgScribbleFromBGWrongC,\
#     fgScribbleFromDefiniteFG, bgScribbleFromDefiniteBG, seedsForGC \
#     = autoGenerateScribbleRegionsAndSeeds3D(gt=gtData, pred=predData, bbPad=bbPad,
#         fractionDivider=fractionDivider, dilation_diam=dilation_diam,\
#         useAtmostNScribblesPerRegion=useAtmostNScribblesPerRegion) 

# gcSegResult_softmax =  generateGrahcutSegmentationFromSeeds(\
#         gt=gtData, predFromNN=predData, binLimit=binLimit, seedsForGC=seedsForGC,\
#         imgForGC=softmaxData, segparams=segparams_ssgc)
# gcSegResult_ct =  generateGrahcutSegmentationFromSeeds(\
#         gt=gtData, predFromNN=predData, binLimit=binLimit, seedsForGC=seedsForGC,\
#         imgForGC=ctData, segparams=segparams_ssgc)
# gcSegResult_pet =  generateGrahcutSegmentationFromSeeds(\
#         gt=gtData, predFromNN=predData, binLimit=binLimit, seedsForGC=seedsForGC,\
#         imgForGC=ptData, segparams=segparams_ssgc)
# dice_org = dice_multi_label(predData, gtData)
# dice_gc_softmax = dice_multi_label(gcSegResult_softmax, gtData)
# dice_gc_ct = dice_multi_label(gcSegResult_ct, gtData)
# dice_gc_pet = dice_multi_label(gcSegResult_pet, gtData)

# print('#FGScrb: ', numFGS, ' #BGScrb: ', numBGS)
# print('d_org: ', dice_org, ' d_softmax: ', dice_gc_softmax,\
#       ' d_ct: ', dice_gc_ct, ' d_pet: ', dice_gc_pet)    

# medianExpResultList= [['CHUM013', 8.0, 8.0, 0.4094309251017896, 0.09460536805740101, 0.027138908793121096, 0.03202846975088968], ['CHGJ089', 4.0, 4.0, 0.0, 0.13222144714312892, 0.21995225670349192, 0.4240802068683936], ['CHUM055', 8.0, 8.0, 0.9564574433125206, 0.9576148677560744, 0.5881558682672631, 0.9053304062553957], ['CHGJ048', 8.0, 8.0, 0.8849823592575549, 0.9010767583905296, 0.18293572762463053, 0.9125090883910951], ['CHGJ057', 8.0, 8.0, 0.8036938309215537, 0.8271943573667712, 0.42832338578404777, 0.8066339992461364], ['CHUS090', 8.0, 8.0, 0.8334956183057449, 0.8285538461538462, 0.4113701393123505, 0.6747041750390712], ['CHMR005', 8.0, 8.0, 0.8384956369009451, 0.8396975037532287, 0.5458245409556394, 0.5636280746834657], ['CHGJ072', 8.0, 8.0, 0.8654171775559059, 0.8527862489603548, 0.3442996025295481, 0.6919912762609391], ['CHUM062', 8.0, 8.0, 0.9104689257249035, 0.9183220829315333, 0.5307034976152624, 0.7960305180600757], ['CHGJ088', 8.0, 8.0, 0.6878202358170258, 0.7339200283503632, 0.42747823468901325, 0.6233907920576042], ['CHUS008', 8.0, 8.0, 0.7772461456671983, 0.828427065026362, 0.15157351167171065, 0.7996172772006561], ['CHUM002', 8.0, 8.0, 0.8988727332135272, 0.9002528341897073, 0.40149334449852586, 0.8412877590644331], ['CHGJ038', 8.0, 8.0, 0.8771493378210288, 0.892401033669151, 0.48553380644155036, 0.8030749977652633], ['CHUS053', 8.0, 8.0, 0.7158914728682171, 0.020452278017958098, 0.24909210238459856, 0.7307692307692307], ['CHUS089', 8.0, 8.0, 0.8391019644527596, 0.8451788139429606, 0.4452008168822328, 0.6626882637112816], ['CHUS050', 4.0, 4.0, 0.0, 0.09388742304309587, 0.15358874254579777, 0.7419782551047468], ['CHUS021', 8.0, 8.0, 0.6674981810622597, 0.13697245387920143, 0.33535739070090215, 0.20460917082442384], ['CHUM030', 8.0, 8.0, 0.9590462356869329, 0.9559234898314054, 0.5601918870655861, 0.9001311313403904], ['CHUS043', 8.0, 8.0, 0.9399469620765784, 0.9375271092651253, 0.5328244619588537, 0.9054153361035768], ['CHUS047', 8.0, 8.0, 0.7991498405951116, 0.8285229202037352, 0.4981253347616497, 0.770962296004502], ['CHMR001', 8.0, 8.0, 0.8249035245690765, 0.836200409953037, 0.3367020817137428, 0.2814776224518345], ['CHUS069', 8.0, 8.0, 0.6116492640583722, 0.13751017087062653, 0.24650571791613723, 0.350385423966363], ['CHUM042', 8.0, 8.0, 0.9033574066927614, 0.9272639433141876, 0.4630497592295345, 0.6593630573248408], ['CHMR030', 8.0, 8.0, 0.5348277139681363, 0.1483118574501253, 0.2214392729547737, 0.35595753047581596], ['CHUM007', 4.0, 8.0, 0.6876705532126023, 0.10762942779291552, 0.290448343079922, 0.9130275229357798], ['CHMR016', 8.0, 8.0, 0.8582093662226085, 0.7991035257834348, 0.6127695791113323, 0.3254987595617118], ['CHGJ053', 8.0, 8.0, 0.7522981366459627, 0.28955882352941176, 0.11197324414715719, 0.8764224686420651], ['CHUM040', 8.0, 8.0, 0.6455853615206963, 0.03728020526752698, 0.28790302213991076, 0.21831051504144547], ['CHGJ026', 8.0, 8.0, 0.9042372881355932, 0.8828652029927634, 0.48043656207366986, 0.8839681133746679], ['CHUS095', 7.0, 8.0, 0.7706919945725916, 0.844311377245509, 0.4377554812337421, 0.634600465477114], ['CHGJ091', 8.0, 8.0, 0.9245120313095968, 0.9421980498301438, 0.5726255701944395, 0.9440607361188754], ['CHUS016', 8.0, 8.0, 0.9012864763100095, 0.8910143161742309, 0.46411871064280813, 0.7572022684310019], ['CHUS041', 8.0, 8.0, 0.8416769697650893, 0.8528646123039271, 0.5571690792072307, 0.5953019308818104], ['CHUM016', 8.0, 8.0, 0.9108363163032867, 0.9149070280491648, 0.49480705902220723, 0.8215633423180593], ['CHUM061', 8.0, 8.0, 0.6667884545122397, 0.026552836784081744, 0.2898271071318308, 0.8682721410711935], ['CHUM006', 8.0, 8.0, 0.5661820964039785, 0.588637226369733, 0.5172805933250927, 0.6311710193765796], ['CHMR034', 8.0, 8.0, 0.5903252910815028, 0.6038319647744853, 0.40939446236962557, 0.7146881287726358], ['CHUS086', 8.0, 8.0, 0.9255582285772371, 0.9042794759825328, 0.49550752792617775, 0.704895883007471], ['CHUS067', 4.0, 8.0, 0.6570757486788021, 0.06735751295336788, 0.34221675550088837, 0.9137466307277629], ['CHMR014', 4.0, 4.0, 0.0, 0.13884506808369312, 0.15402724335557613, 0.17664071190211347], ['CHUM050', 8.0, 8.0, 0.8396790862115855, 0.8649119282030674, 0.4623030866709295, 0.6350349197206422], ['CHUS101', 8.0, 8.0, 0.8479898774946799, 0.32256833428035925, 0.45596860364590786, 0.811349810175797], ['CHGJ015', 8.0, 8.0, 0.8529862174578867, 0.8799698511400038, 0.45465707770566566, 0.8647081561654426], ['CHGJ085', 8.0, 8.0, 0.9050395771817077, 0.894175996857199, 0.44740546119921176, 0.7815728893409711], ['CHMR028', 8.0, 8.0, 0.8242143078672594, 0.16804381801342905, 0.5985954588352856, 0.41686006825938565], ['CHUS073', 8.0, 8.0, 0.8874700718276137, 0.9041782729805014, 0.4548366431774504, 0.7033036848792884], ['CHUM001', 8.0, 8.0, 0.8105174654131315, 0.8132051184977663, 0.3436393822635717, 0.27424995398490704], ['CHGJ018', 8.0, 8.0, 0.8999611247894259, 0.9123155975468258, 0.3664142073648472, 0.8260449290492209], ['CHUS094', 8.0, 8.0, 0.9319383766383077, 0.9287287055278711, 0.5177241584748287, 0.9351198871650211], ['CHUM021', 8.0, 8.0, 0.8906373431920878, 0.8824776714001157, 0.5433706186989626, 0.8394437420986094], ['CHUS019', 8.0, 8.0, 0.9179834462001505, 0.9287295607255214, 0.4771457882138379, 0.8186187255250978], ['CHMR024', 8.0, 8.0, 0.6547827017005954, 0.6563233933527863, 0.3898241385119811, 0.7511739915713426], ['CHUS056', 8.0, 8.0, 0.9553831437984908, 0.9562172003855628, 0.5680538623178195, 0.8411271417868967], ['CHUS006', 8.0, 8.0, 0.8721350233138504, 0.8661967276725085, 0.4220315631262525, 0.8085305105853051], ['CHUM034', 8.0, 8.0, 0.8810578031176078, 0.8855597423118998, 0.5544449767879859, 0.7842093865049655], ['CHUS005', 8.0, 8.0, 0.7468210717529519, 0.7201797385620915, 0.28017320241507593, 0.6888316553500351], ['CHUS096', 4.0, 8.0, 0.7045871559633028, 0.06578947368421052, 0.328851841892982, 0.9174418604651163], ['CHMR023', 8.0, 8.0, 0.6358891485226355, 0.6599333788862316, 0.35718971651732306, 0.5572718094003828], ['CHUS057', 8.0, 8.0, 0.8074890532991091, 0.8465863453815261, 0.3795336787564767, 0.8827774527013849], ['CHUM008', 8.0, 8.0, 0.8513546899448242, 0.8760556583286415, 0.4915139681586062, 0.6067356138955184], ['CHUS088', 8.0, 8.0, 0.9278365247213489, 0.9112135633551457, 0.391314268329784, 0.8930663003398164], ['CHUS085', 7.0, 8.0, 0.8682380871629642, 0.9068977841315226, 0.4085816448152563, 0.9195314000384098], ['CHUS048', 8.0, 8.0, 0.7682560137457045, 0.7931916568280205, 0.2912028725314183, 0.7634720327421555], ['CHUM064', 8.0, 8.0, 0.9019573228589712, 0.9153724521844607, 0.49816277400008446, 0.9094483086361084], ['CHGJ065', 8.0, 8.0, 0.8906687907484439, 0.9052429052429053, 0.42896885182719596, 0.8608759598447147], ['CHUS020', 4.0, 8.0, 0.7407801182802844, 0.029344175357963586, 0.42106100373816907, 0.904537473431291], ['CHUS081', 8.0, 8.0, 0.9367958331794171, 0.9411522016352861, 0.5057560747754498, 0.8998667581862962], ['CHMR013', 4.0, 8.0, 0.30446503791069923, 0.08377518557794274, 0.23095415734062838, 0.8477284073889166], ['CHUM051', 8.0, 8.0, 0.8483547925608012, 0.8977157873458662, 0.4595830860016062, 0.8813409141253784], ['CHUM012', 8.0, 8.0, 0.9054988072664995, 0.9253617669459253, 0.11984266465813039, 0.9258188824662813], ['CHGJ036', 8.0, 8.0, 0.9195910068311806, 0.9197481026965929, 0.5080752477839425, 0.8916999236410263], ['CHUM044', 8.0, 6.0, 0.8923570595099183, 0.919167061050639, 0.5165075887967454, 0.9336023005279197], ['CHUS049', 8.0, 8.0, 0.8886517943743938, 0.8976424361493124, 0.4650575357844513, 0.8684764098968246], ['CHGJ086', 8.0, 8.0, 0.7819022272026894, 0.8086301167430175, 0.3235170823726682, 0.3003003003003003], ['CHGJ046', 8.0, 8.0, 0.9214487802274062, 0.9219082740352152, 0.13483674870085302, 0.9045892351274788], ['CHUM049', 8.0, 8.0, 0.8653869083104805, 0.8374448195273955, 0.5703626943005181, 0.8741217101345718], ['CHUM056', 8.0, 8.0, 0.8163119042499821, 0.8256358768406962, 0.3747238179407866, 0.782376298106292], ['CHGJ062', 8.0, 8.0, 0.8068495854717727, 0.8382585878625942, 0.6224711619435577, 0.8496408300133823], ['CHUM063', 8.0, 8.0, 0.737800467985535, 0.2530050214085549, 0.29007686328698506, 0.7921859602341893], ['CHUM018', 8.0, 8.0, 0.2470978441127695, 0.2458295525930062, 0.23929982945087266, 0.25613732517718074], ['CHUM058', 6.0, 8.0, 0.4854368932038835, 0.5872122762148337, 0.18503118503118504, 0.4357414448669201], ['CHUM027', 8.0, 8.0, 0.9218930176734992, 0.9312611796408576, 0.5628786534512915, 0.9351691690306511], ['CHUM015', 8.0, 4.0, 0.43500164275544845, 0.7210078424828967, 0.5085636154289004, 0.7091038406827881], ['CHUM036', 8.0, 8.0, 0.7074896316174677, 0.6861575178997613, 0.39981550316086495, 0.6122497759187332], ['CHUM053', 8.0, 4.0, 0.8752202341333543, 0.9321346989904944, 0.5748297181179922, 0.9450139162310773], ['CHUS028', 4.0, 4.0, 0.0, 0.13904288036984075, 0.22836843440751078, 0.15865728142055735], ['CHGJ008', 6.0, 8.0, 0.7562977235424249, 0.29547347958900305, 0.2620868000899483, 0.8794909446891825], ['CHUS060', 8.0, 8.0, 0.8784016249041431, 0.8643143883372069, 0.6149137451307735, 0.5059743454577403], ['CHMR040', 8.0, 8.0, 0.8481209056430296, 0.8657344314747881, 0.5157543063814136, 0.6745562130177515], ['CHUS039', 8.0, 8.0, 0.75, 0.7966112600536193, 0.37844974192200975, 0.6487098600663269], ['CHUM033', 8.0, 8.0, 0.8946857558844941, 0.9294251998477351, 0.44640081163783935, 0.9174928046050528], ['CHUM046', 8.0, 8.0, 0.21461988304093568, 0.2859159901164843, 0.22047413022490792, 0.2267515923566879], ['CHGJ013', 8.0, 8.0, 0.843348623853211, 0.8648587570621469, 0.48424289008455035, 0.8763576522216128], ['CHUS076', 8.0, 8.0, 0.8199248670537151, 0.8222721061877806, 0.530700969235097, 0.6342397548831865], ['CHUM011', 8.0, 8.0, 0.8690478020879779, 0.8716203732216542, 0.49116940119134705, 0.5934982122555239], ['CHUS040', 8.0, 8.0, 0.890330018734688, 0.8969987318585317, 0.4371837501807142, 0.856973293768546], ['CHUS045', 8.0, 8.0, 0.7570125623816899, 0.7487830319888734, 0.4074894181089671, 0.7186047503304763], ['CHGJ035', 8.0, 8.0, 0.9286528750610848, 0.9298352160006478, 0.5457303000329706, 0.9032231507018482], ['CHUS031', 8.0, 8.0, 0.8693845240451217, 0.9097245114198258, 0.5426966009020586, 0.7141216991963261], ['CHUS074', 8.0, 8.0, 0.9007592913280937, 0.906902935731288, 0.42952684542783925, 0.7592995014700243], ['CHUS015', 8.0, 5.0, 0.7871559633027523, 0.8752312435765673, 0.4558006096467635, 0.888713807366555], ['CHUS087', 8.0, 8.0, 0.9095251101259884, 0.9042528484865546, 0.5663232631504044, 0.5505571262265092], ['CHMR020', 8.0, 8.0, 0.6512275981357806, 0.18969836656979658, 0.3608952800649426, 0.733105612998523], ['CHUM060', 8.0, 8.0, 0.943194352525921, 0.9313860252004582, 0.482989277834769, 0.70371402905554], ['CHUM019', 8.0, 8.0, 0.8623057462956271, 0.906949806949807, 0.4027638291889286, 0.9081349775356085], ['CHGJ010', 8.0, 8.0, 0.8516788970424728, 0.8732519422863485, 0.3961229435021043, 0.8768562942654786], ['CHUM032', 4.0, 8.0, 0.7362604360622259, 0.013444561431926783, 0.39976428992339424, 0.9334490740740741], ['CHUS033', 8.0, 8.0, 0.7446314567614626, 0.7491620860705188, 0.34912891986062716, 0.6470161874848998], ['CHUS066', 4.0, 8.0, 0.7948919009745715, 0.8614067278287462, 0.41040516515392533, 0.9162058698426202], ['CHUS022', 8.0, 8.0, 0.8887091954225043, 0.907388351512657, 0.41409106974127424, 0.6626599383871029], ['CHUM038', 8.0, 8.0, 0.48517890685847803, 0.5058387141090809, 0.3907711090762785, 0.34392513044198436], ['CHUM017', 8.0, 8.0, 0.9215341626104201, 0.9243697478991597, 0.5798993608051135, 0.5266608784168139], ['CHUM037', 8.0, 8.0, 0.8916956109883171, 0.9177814029363784, 0.4688617277791636, 0.889739383998542], ['CHGJ067', 4.0, 4.0, 0.0, 0.07667050979998997, 0.09324393431444969, 0.387603964908283], ['CHMR004', 8.0, 8.0, 0.8661597654818615, 0.890075460041481, 0.4567859411391065, 0.6390998281691702], ['CHGJ017', 8.0, 8.0, 0.8969021943712495, 0.9033763072484674, 0.436083338104334, 0.40751853844404634], ['CHGJ087', 8.0, 8.0, 0.8383675351625548, 0.3284982567899109, 0.34088912317694603, 0.42282784673502427], ['CHGJ081', 8.0, 8.0, 0.8046320508031378, 0.8823529411764706, 0.32668099478047286, 0.715323787134003], ['CHGJ077', 8.0, 8.0, 0.819325328759291, 0.8358121901428988, 0.3475842891901286, 0.7107215104517869], ['CHUM041', 8.0, 6.0, 0.5944206008583691, 0.6030795551753636, 0.377734316232887, 0.8267862666254253], ['CHUM054', 8.0, 8.0, 0.2897300750678805, 0.21791044776119403, 0.08325081600469432, 0.21114369501466276], ['CHUM035', 8.0, 8.0, 0.8155099515314015, 0.32925575865057133, 0.047110141766630316, 0.3790569175473419], ['CHGJ082', 8.0, 8.0, 0.8274833328259612, 0.8490662805080399, 0.5720113693362314, 0.715646352405273], ['CHMR025', 8.0, 8.0, 0.8658232628398792, 0.8771026433230347, 0.46784204099345034, 0.4021741479544423], ['CHGJ043', 8.0, 8.0, 0.9180318527593379, 0.9252163500789201, 0.46565984400355964, 0.8836418033027308], ['CHGJ039', 5.0, 8.0, 0.7385242049193169, 0.8011675824175825, 0.32632063882063883, 0.7139272271016311], ['CHUS083', 8.0, 8.0, 0.9461393362121377, 0.9473469579810845, 0.5994051023910307, 0.941093879634948], ['CHGJ070', 8.0, 8.0, 0.7914419899112889, 0.8089488636363636, 0.35510556858871467, 0.6965554948059048], ['CHUM045', 7.0, 8.0, 0.8489720204050085, 0.8574132492113564, 0.4444444444444444, 0.8644444444444445], ['CHUS036', 8.0, 8.0, 0.598835656678951, 0.16379483356747612, 0.3497899550169151, 0.504412978601816], ['CHUS026', 8.0, 8.0, 0.5259284497444634, 0.28392146852998107, 0.46415197281948267, 0.3236967509025271], ['CHUS042', 8.0, 8.0, 0.9180561863362097, 0.9230169050715215, 0.44794912559618444, 0.9153321170349348], ['CHUS078', 8.0, 8.0, 0.4566601689408707, 0.1440803300381412, 0.16368012837657128, 0.43926896328643056], ['CHUM057', 8.0, 8.0, 0.8186218972087753, 0.8490064817766444, 0.3583691939133135, 0.9266324284666178], ['CHGJ080', 8.0, 8.0, 0.8598337134868663, 0.8530272685256509, 0.39137973742878374, 0.7253963011889035], ['CHGJ090', 4.0, 4.0, 0.0, 0.08437983297247138, 0.1344311699494134, 0.7091043671354552], ['CHUS027', 8.0, 8.0, 0.9611419104059135, 0.9588953197387164, 0.5877905137744573, 0.9497422847720811], ['CHUM029', 8.0, 8.0, 0.7220338983050848, 0.7207858800669293, 0.5207921263970641, 0.7327874761831197], ['CHUS051', 8.0, 8.0, 0.3670886075949367, 0.2036271078587337, 0.34024746726464583, 0.2249043516556109], ['CHUM048', 8.0, 4.0, 0.635667107001321, 0.7255481645725548, 0.4644037807887508, 0.8000421851929973], ['CHUS077', 8.0, 8.0, 0.920437864362706, 0.9213141058608134, 0.47435457184778196, 0.8400985956898306], ['CHUM026', 5.0, 8.0, 0.5085435313262815, 0.5496149614961496, 0.11703229451087468, 0.6186965496903568], ['CHGJ007', 8.0, 8.0, 0.9213869692488759, 0.9224201294843783, 0.5244842068994197, 0.48420597144093463], ['CHGJ055', 7.0, 8.0, 0.8249113270993597, 0.02090899086607241, 0.23243903858464762, 0.8994195132842152], ['CHGJ076', 8.0, 8.0, 0.9002370548082141, 0.9200930040910027, 0.3915429467391837, 0.8699205900920357], ['CHUS097', 8.0, 8.0, 0.8628457501661434, 0.8487590968464266, 0.5016207455429498, 0.7182060367793498], ['CHGJ066', 8.0, 8.0, 0.8908296943231441, 0.8999128955991539, 0.4853281495880774, 0.8993033017439093], ['CHUS061', 8.0, 8.0, 0.5881192459447611, 0.6081845403127426, 0.2928783145077305, 0.3129559896446223], ['CHUS058', 4.0, 8.0, 0.35911330049261087, 0.21542227662178703, 0.04042772186642269, 0.8248337028824834], ['CHUS010', 8.0, 8.0, 0.6148257463957316, 0.6141069130892503, 0.2975614996240198, 0.3326605151864667], ['CHUS035', 6.0, 8.0, 0.7548337200309359, 0.06229508196721312, 0.08521253242865695, 0.8952380952380953], ['CHUS055', 8.0, 8.0, 0.8677867056245434, 0.8852022486157488, 0.5483393638278339, 0.5378517542809245], ['CHUS046', 8.0, 8.0, 0.8906624102154829, 0.9108457385898904, 0.4360863159719213, 0.9091530054644809], ['CHUM014', 8.0, 8.0, 0.6747922437673131, 0.679839304747069, 0.39364236976064115, 0.4868740229637216], ['CHGJ032', 8.0, 8.0, 0.9140625, 0.8663009941720946, 0.5241823783814342, 0.897807830733995], ['CHUS091', 8.0, 8.0, 0.9083698256630588, 0.9202474226804124, 0.59683578832515, 0.9289402507991148], ['CHUS007', 8.0, 8.0, 0.8211462450592886, 0.8171293414664509, 0.4218463317163898, 0.6403483586946119], ['CHGJ074', 8.0, 8.0, 0.7352727020561007, 0.2682810897355715, 0.06439241917502787, 0.8871699374492671], ['CHGJ030', 8.0, 8.0, 0.891533180778032, 0.8811229999018356, 0.32798246234354006, 0.848119090726212], ['CHMR021', 8.0, 8.0, 0.7911854318713243, 0.7609713282621416, 0.48096026490066224, 0.6767314332137296], ['CHUS064', 5.0, 8.0, 0.6957078795643818, 0.7872340425531915, 0.2679916653309825, 0.8460854092526691], ['CHUM023', 8.0, 8.0, 0.8685348278622899, 0.8856063208519409, 0.44917056571671626, 0.7806149035956227], ['CHGJ073', 8.0, 8.0, 0.827710961181187, 0.0041771308492645175, 0.04666133323410668, 0.3631917482760619], ['CHUS080', 8.0, 8.0, 0.8904972624315608, 0.8907708521529659, 0.49514456374363297, 0.7161833169606628], ['CHUM065', 8.0, 8.0, 0.6248523815772208, 0.19954066748049906, 0.13044583396878653, 0.76162215628091], ['CHGJ025', 8.0, 8.0, 0.9235581000848176, 0.919317938085365, 0.48707960908426945, 0.8335357550329395], ['CHUS013', 8.0, 8.0, 0.8811155378486055, 0.8955642023346303, 0.5092762253779203, 0.9099482811073928], ['CHUM039', 8.0, 8.0, 0.9200860489204048, 0.9348526133267278, 0.15722492909606575, 0.9451834279420487], ['CHGJ071', 8.0, 8.0, 0.7411489645958583, 0.7588225593667546, 0.5771579846997269, 0.4706211527700056], ['CHUS038', 8.0, 8.0, 0.9495299548704952, 0.9526343519494205, 0.49222814755152544, 0.925458084435197], ['CHGJ058', 8.0, 8.0, 0.8739564243534922, 0.9005796784425243, 0.37587787543630935, 0.8262278733482922], ['CHGJ052', 8.0, 8.0, 0.793766917843295, 0.7794422678036413, 0.3647605120706021, 0.5798165137614679], ['CHMR012', 8.0, 8.0, 0.6080898006580221, 0.15944211110230605, 0.34590497319282676, 0.38893462868434053], ['CHUS098', 8.0, 7.0, 0.843346086237204, 0.8758222045046841, 0.5390663879765104, 0.9065637065637066], ['CHUM043', 8.0, 8.0, 0.8832358674463937, 0.867394082374914, 0.44878530692891466, 0.7171387073347858], ['CHMR011', 8.0, 8.0, 0.7910625745308479, 0.15655333764746876, 0.44501461901068085, 0.47480660673217645], ['CHUM024', 8.0, 8.0, 0.908477729968601, 0.9116344136162275, 0.40195408408636446, 0.8620224112326178], ['CHUM059', 8.0, 8.0, 0.8371250251660962, 0.8618447785453068, 0.43714488636363635, 0.7246954595791805], ['CHGJ078', 8.0, 8.0, 0.8849774866966844, 0.8997003349197955, 0.5873412450551738, 0.8462301587301587], ['CHGJ050', 8.0, 8.0, 0.9333052808752367, 0.9336493723114739, 0.5365531700120356, 0.9230072634657108], ['CHGJ092', 8.0, 8.0, 0.848808198671938, 0.8590891397430906, 0.408259413301925, 0.8660203357767794], ['CHUS030', 8.0, 8.0, 0.30894771799981025, 0.4729609929078014, 0.22359905868510074, 0.5504853488161118], ['CHGJ031', 8.0, 8.0, 0.8693823915900132, 0.8916621286379863, 0.5427380064082868, 0.6460492112666025], ['CHUS003', 8.0, 8.0, 0.7845650752125573, 0.7920623671155209, 0.4345258770406391, 0.5769850402761795], ['CHGJ034', 8.0, 8.0, 0.7995522995522996, 0.8485378412513359, 0.5444832750873689, 0.7640278174716586], ['CHUS065', 8.0, 5.0, 0.5173608523954996, 0.6244847983168641, 0.5181877044145236, 0.800175387313651], ['CHUS009', 8.0, 4.0, 0.6408885803540437, 0.749473146433361, 0.46393578743426517, 0.908744833587122], ['CHUM047', 8.0, 4.0, 0.48221698642432403, 0.643107418159976, 0.4369238960433929, 0.6673387096774194], ['CHGJ016', 8.0, 8.0, 0.9387981168651343, 0.9403782894736842, 0.5841267928166807, 0.9136169826725936], ['CHMR029', 8.0, 8.0, 0.8242872028038268, 0.7188054714419965, 0.37794068643785406, 0.7753049907578559], ['CHGJ029', 8.0, 8.0, 0.7253210657465977, 0.743978243978244, 0.35920101997450066, 0.33404649835972045], ['CHGJ028', 8.0, 8.0, 0.8042967916340613, 0.320471297744025, 0.05343047965998786, 0.823153473934082], ['CHUS004', 8.0, 8.0, 0.8253544620517097, 0.8341384863123994, 0.4285846097065879, 0.8175705178587023], ['CHUM010', 8.0, 8.0, 0.4604510265903736, 0.5994375251104861, 0.042212011313158763, 0.11948790896159317], ['CHGJ037', 8.0, 8.0, 0.8534618473895582, 0.8449322898061248, 0.5526985952517868, 0.7156056110062045], ['CHGJ083', 8.0, 8.0, 0.8168421052631579, 0.8378531073446328, 0.3318950518477278, 0.6932594644506002], ['CHGJ069', 8.0, 8.0, 0.6535428667300487, 0.03913186451825058, 0.303552984920471, 0.8087912087912088], ['CHUS052', 8.0, 8.0, 0.9366042724514017, 0.9380759444556369, 0.49752616698748797, 0.9238994095511978], ['CHUS100', 8.0, 8.0, 0.7729934464628342, 0.785985803737283, 0.4507697419426293, 0.6617915482544303], ['CHUS068', 8.0, 8.0, 0.8850671140939598, 0.9048575527772085, 0.543867364439949, 0.7268011527377521], ['CHUM022', 8.0, 8.0, 0.7028798068632522, 0.16621655483475792, 0.43373493975903615, 0.387598944591029]]
# #Result over patients
# medianExpResult_df = pd.DataFrame(medianExpResultList,\
#       columns = ['patientName', '#FGScrb', ' #BGScrb', 'd_org', 'd_softmax', 'd_ct', 'd_pet'])
# print(medianExpResult_df)
# medianExpResult_df.to_csv('/home/user/DMML/Data/PlayDataManualSegmentation/AutoScribbleExperiment/medianExpResult_df.csv', index = False)

pass
