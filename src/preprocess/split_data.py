import sys
from os import path, makedirs
import glob
import random
import shutil
import ntpath


if len(sys.argv) != 3:
    print('USAGE: python split_data.py <data path> <train data percentage>')
    print('Example 80/20 split: python split_data.py /mnt/data/data/pancreas/2mm 80')
else:
    input_dir = sys.argv[1]
    perc_train = float(sys.argv[2])

    train_dir = input_dir + '/train/'
    test_dir = input_dir + '/test/'
    
    if not path.exists(train_dir):    
        makedirs(train_dir)
        
    if not path.exists(test_dir):    
        makedirs(test_dir)

    img_fnames = sorted(glob.glob(input_dir + '/img*.nii.gz'))
    lbl_fnames = sorted(glob.glob(input_dir + '/lbl*.nii.gz'))    
    assert len(img_fnames) == len(lbl_fnames)
    
    indeces = list(range(0, len(img_fnames)))
    random.shuffle( indeces )
    #print('Randomized indeces: ', indeces)

    train_indeces = range(0, int((perc_train/100.0) * len(img_fnames)) )
    test_indeces = range(int((perc_train/100.0) * len(img_fnames)), len(img_fnames) )

    print('Copying to TRAIN:')
    for i in train_indeces:
        img_fname = img_fnames[indeces[i]]
        lbl_fname = lbl_fnames[indeces[i]]
        print(img_fname)
        shutil.copyfile(img_fname, train_dir + ntpath.basename(img_fname))
        print(lbl_fname)
        shutil.copyfile(lbl_fname, train_dir + ntpath.basename(lbl_fname))

    print('Copying to TEST:')
    for i in test_indeces:
        img_fname = img_fnames[indeces[i]]
        lbl_fname = lbl_fnames[indeces[i]]
        print(img_fname)
        shutil.copyfile(img_fname, test_dir + ntpath.basename(img_fname))
        print(lbl_fname)
        shutil.copyfile(lbl_fname, test_dir + ntpath.basename(lbl_fname))

    print('Number of train datasets: ', len(train_indeces))
    print('Number of test datasets: ', len(test_indeces))
