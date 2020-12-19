import sys
#import pprint
#pprint.pprint(sys.path)
#pprint.pprint(sys.modules)
import  DSSENet
from DSSENet import DSSE_VNet
DSSE_VNet.train(trainConfigFilePath = '/home/user/DMML/CodeAndRepositories/MMGTVSeg/input/trainInput_DSSENet.json', 
             data_format='channels_last',
             cpuOnlyFlag=True)