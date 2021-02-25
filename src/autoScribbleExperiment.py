# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import glob
import pandas as pd
import scribbleAndGCHelper


# ########### Test Code #############        
# #Run experiment on a patient

#Save / Read from json file
patDataConfig = \
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


autoScribbleAndGCConfig = \
{
 'bbPad' : 2,
 'fractionDivider' : 10,
 'dilation_diam' : 2,
 'useAtmostNScribblesPerRegion' : 4,
 'segparams_ssgc' : 
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


    
########### Test Code #############        
#Run experiment on a patient

srcFolder =\
 '/home/user/DMML/Data/HeadNeck_PET_CT/nnUnet_3dfullres/validation_gtvs_withSoftmax'
expFolder = '/home/user/DMML/Data/PlayDataManualSegmentation/AutoScribbleExperiment'
# srcFolder = 'J:/HecktorData/nnUnet_3dfullres/validation_gtvs_withSoftmax'
# expFolder = 'J:/PlayDataManualSegmentation/AutoScribbleExperiment'
numExperimentsPerPat = 100
verbose = False

medianExpResultList = []
listOfPatients = [(os.path.basename(f)).replace('_ct.nii.gz','') \
      for f in glob.glob(srcFolder + '/*_ct.nii.gz', recursive=False) ]
print(listOfPatients)
#listOfPatients = ['CHGJ017', 'CHUM038', 'CHGJ008']
#patientName = 'CHUM038'
for patientName in listOfPatients:
    print('Experimenting graphCut on ', patientName)
    expResult_df = scribbleAndGCHelper.runAutoGCExperimentOnPatient(patientName, srcFolder, expFolder,\
                                      patDataConfig, autoScribbleAndGCConfig, 
                                      numExperimentsPerPat, verbose)
    print(expResult_df)
    numeric_expResult_df = expResult_df[['#FGScrb', '#BGScrb', 'd_org', ' d_softmax', ' d_ct', ' d_pet']]
    medianExpResultForPatient = numeric_expResult_df.median(axis = 0) 
    print('Median result of ', numExperimentsPerPat, ' experiment over ', patientName, ' is: ')
    print(medianExpResultForPatient) 
    medianExpResultForPatientList = [patientName] + \
        [medianExpResultForPatient[id] for id in range(len(medianExpResultForPatient))]
    medianExpResultList.append(medianExpResultForPatientList)

#Result over patients
medianExpResult_df = pd.DataFrame(medianExpResultList,\
      columns = ['patientName', '#FGScrb', ' #BGScrb', 'd_org', 'd_softmax', 'd_ct', 'd_pet'])
# print(medianExpResult_df)
# medianExpResult_df.to_csv('/home/user/DMML/Data/PlayDataManualSegmentation/AutoScribbleExperiment/medianExpResult_df.csv', index = False)

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


pass
