import sys
import os
rootPath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg'
sys.path.append('/home/user/DMML/CodeAndRepositories/MMGTVSeg')
import src
from src import  DSSENet
from DSSENet import DSSE_VNet
#import pprint
#pprint.pprint(sys.path)
#pprint.pprint(sys.modules)

import glob
resampledTestDataLocation = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/data/hecktor_test/resampled_test'
#First get  file list, then drop the '_ct.nii.gz' to get patient name 
listOfTestPatientNames = [(os.path.basename(f)).replace('_ct.nii.gz','') \
      for f in glob.glob(resampledTestDataLocation + '/*_ct.nii.gz', recursive=False) ]


DSSE_VNet.ensembleBasedPrediction(listOfTestPatientNames,
            resampledTestDataLocation = resampledTestDataLocation,
            groundTruthPresent = False,
            trainConfigFilePath = os.path.join(rootPath, 'input/trainInput_DSSENet.json'),
            trainModelEnsembleJsonPath_in = os.path.join(rootPath, 'output/trainModelCVEval_DSSENet.json'),
            out_dir = os.path.join(rootPath, 'output/blind_test')
            )