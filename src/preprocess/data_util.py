import os
import re
from subprocess import call, check_output, check_call #, run
from math import ceil, floor
import shlex
from sys import platform
import numpy as np
import nibabel as nib
from sys import platform
from os.path import expanduser
import pickle
import SimpleITK
import glob
import sys
sys.path.append('/home/user/DMML/CodeAndRepositories/MMGTVSeg')
import src
from src.DSSENet import volume #from vmsseg import volume

from scipy import ndimage
from shutil import copyfile
import pydicom


def get_data_paths(src_folder, dst_folder):
    """Get the full paths of source and destination data folders."""
    home_folder = expanduser('~')
    if platform == 'linux' or platform == 'linux2':
        src = home_folder + '/' + src_folder + '/'
        dst = home_folder + '/' + dst_folder + '/'
    elif platform == 'darwin':
        src = home_folder + '/' + src_folder + '/'
        dst = home_folder + '/' + dst_folder + '/'
    elif platform == 'win32':
        home_folder = home_folder.replace('\\','/')
        src = home_folder + '/' + src_folder + '/'
        dst = home_folder + '/' + dst_folder + '/'
    return src, dst

def get_full_path(folder_path):
    """Get the full path of a data folder."""
    home_folder = expanduser('~')
    if platform == 'linux' or platform == 'linux2':
        full_path = home_folder + '/' + folder_path + '/'
    elif platform == 'darwin':
        full_path = home_folder + '/' + folder_path + '/'
    elif platform == 'win32':
        home_folder = home_folder.replace('\\','/')
        full_path = home_folder + '/' + folder_path + '/'
    return full_path

def get_c3d_command():
    if platform == 'linux' or platform == 'linux2':
        c3d = 'util/c3d'
    elif platform == 'darwin':
        c3d = 'util/c3d'
    elif platform == 'win32':
        c3d = 'util/c3d.exe'
    return c3d

c3d = get_c3d_command()

def resample_PDDCA_data(inDir, outDir, resMM, paddedVolSize, padValue, croppedVolSize, structureName):
    if os.path.isdir(outDir) == False:
        os.mkdir(outDir)
    
    caseList = os.listdir(inDir)
    shortList = [caseList[0]]
    caseNum = 1;

    for case in caseList:
        print('Processing case: %s' % (case))

        # make sure all structures exist
        allStructsAvailable = True
        for struct in structureName:
            structureFileName = '%s%s/structures/%s.nrrd' % (inDir, case, struct)
            if os.path.isfile(structureFileName) == False: allStructsAvailable = False

        if allStructsAvailable == True:
            # resample the grayscale volume
            cmd = '%s %s%s/img.nrrd -interpolation Linear -resample-mm %.1fx%.1fx%.1fmm -type short -o %simg%d.nii' \
                % (c3d, inDir, case, resMM[0], resMM[1], resMM[2], outDir, caseNum)       
            print(cmd)
            run(shlex.split(cmd))

            # pad the grayscale volume to a consistent volume size       
            PadNiiVolumeToSize('%simg%d.nii' % (outDir, caseNum), '%simg%d.nii' % (outDir, caseNum), paddedVolSize, padValue, 'short')

            # crop the grayscale volume to a smaller subvolume 
            CropNiiVolumeToSize('%simg%d.nii' % (outDir, caseNum), '%simg%d.nii' % (outDir, caseNum), croppedVolSize, 'short')
            
            # --------------- process the label volume ---------------

            # combine the label volumes into one volume
            CreateCombinedLabelVolume('%s%s/structures/' % (inDir, case), '%slbl%d.nii' % (outDir, caseNum), structureName, 'uchar')

            # resample the labels
            cmd = '%s %s -interpolation NearestNeighbor -resample-mm %.1fx%.1fx%.1fmm -type uchar -o %slbl%d.nii' \
                % (c3d, '%slbl%d.nii' % (outDir, caseNum), resMM[0], resMM[1], resMM[2], outDir, caseNum)
            print(cmd)
            run(shlex.split(cmd))

            # pad the label volume to a consistent volume size       
            PadNiiVolumeToSize('%slbl%d.nii' % (outDir, caseNum), '%slbl%d.nii' % (outDir, caseNum), paddedVolSize, 0, 'uchar')

            # crop the label volume to a smaller subvolume 
            CropNiiVolumeToSize('%slbl%d.nii' % (outDir, caseNum), '%slbl%d.nii' % (outDir, caseNum), croppedVolSize, 'uchar')

            # increment case number only if structure file was present
            caseNum = caseNum + 1
        else:
            print('WARNING: Structure file %s does not exist!' % (structureFileName))

def resample_PDDCA_data2(in_dir, out_dir, res, cropped_vol_size, structure_name):
    if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)
    
    #case_list = os.listdir(in_dir)
    case_list = glob.glob(in_dir + '/0*')
    short_list = [case_list[0]]
    case_num = 1;

    for case in case_list:
        case = os.path.basename(case)
        print('Processing case: %s' % (case))

        # make sure all structures exist
        all_structs_available = True
        # for struct in structure_name:
        #     structure_filename = '%s%s/structures/%s.nrrd' % (in_dir, case, struct)
        #     if os.path.isfile(structure_filename) == False: all_structs_available = False

        if all_structs_available == True:
            convert_nrrd_to_nii('%s%s/img.nrrd' % ((in_dir, case)), '%simg%d.nii.gz' % (out_dir, case_num))
            print(in_dir)
            print(case)
            create_combined_label_volume('%s%s/structures/' % (in_dir, case), '%slbl%d.nii.gz' % (out_dir, case_num), structure_name, 'uint8')
            resample_and_crop_volumes('%simg%d.nii.gz' % (out_dir, case_num), '%slbl%d.nii.gz' % (out_dir, case_num), cropped_vol_size, resolution=res, padding=[100, 100, 100])
            case_num = case_num + 1
        else:
            print('WARNING: Structure file %s does not exist!' % (structure_filename))

def resample_MICCAI_data(in_dir, out_dir, res, padding, pad_value, cropped_vol_size, clamp_range, center_label=-1):
    if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)
    
    img_list = sorted(os.listdir(in_dir + '/img/'))
    lbl_list = sorted(os.listdir(in_dir + '/label/'))
    case_num = 1;

    for img_fname, lbl_fname in zip(img_list, lbl_list):
        print('Processing %s and %s' % (img_fname, lbl_fname))
        copyfile('%s/img/%s' %(in_dir, img_fname), '%s/img%d.nii.gz' % (out_dir, case_num))
        convert_nii_to_rai('%s/img%d.nii.gz' % (out_dir, case_num), '%s/img%d.nii.gz' % (out_dir, case_num), dtype=np.int16)
        flip_nii_axis('%s/img%d.nii.gz' % (out_dir, case_num), '%s/img%d.nii.gz' % (out_dir, case_num), img_flip=[False,True,False], dtype=np.int16)
        copyfile('%s/label/%s' %(in_dir, lbl_fname), '%s/lbl%d.nii.gz' % (out_dir, case_num))
        convert_nii_to_rai('%s/lbl%d.nii.gz' % (out_dir, case_num), '%s/lbl%d.nii.gz' % (out_dir, case_num), dtype=np.uint8)
        flip_nii_axis('%s/lbl%d.nii.gz' % (out_dir, case_num), '%s/lbl%d.nii.gz' % (out_dir, case_num), img_flip=[False,True,False], dtype=np.uint8)
        #resample_and_crop_volumes('%simg%d.nii.gz' % (out_dir, case_num), '%slbl%d.nii.gz' % (out_dir, case_num), cropped_vol_size, res, padding, center_label)
        resample_and_crop_volumes(img_fname = '%simg%d.nii.gz' % (out_dir, case_num), lbl_fname = '%slbl%d.nii.gz' % (out_dir, case_num), 
            cropped_vol_size = cropped_vol_size, resolution = res, padding = padding, pad_value = pad_value, clamp_range = clamp_range, center_label=center_label)
        case_num = case_num + 1

def resample_PancreasTCIA_data(in_dir, out_dir, res, padding, pad_value, cropped_vol_size, clamp_range):
    if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)
    
    imgdir_list = sorted(os.listdir(in_dir + '/DOI/'))
    lbl_list = sorted(os.listdir(in_dir + '/TCIA_pancreas_labels-02-05-2017/'))
    case_num = 1;

    for img_dir, lbl_fname in zip(imgdir_list, lbl_list):
        print('Processing %s and %s' % (img_dir, lbl_fname))
        dicom_path = in_dir + '/DOI/' + img_dir
        dicom_path = dicom_path + '/' + os.listdir(dicom_path)[0]
        dicom_path = dicom_path + '/' + os.listdir(dicom_path)[0] + '/'
        print(dicom_path)
        convert_dicom_to_nii(dicom_path, '%s/img%d.nii.gz' % (out_dir, case_num))
        copyfile('%s/TCIA_pancreas_labels-02-05-2017/%s' %(in_dir, lbl_fname), '%s/lbl%d.nii.gz' % (out_dir, case_num))
        print('pad_value1: ', pad_value)
        resample_and_crop_volumes(img_fname = '%simg%d.nii.gz' % (out_dir, case_num), lbl_fname = '%slbl%d.nii.gz' % (out_dir, case_num), 
            cropped_vol_size = cropped_vol_size, resolution = res, padding = padding, pad_value = pad_value, clamp_range = clamp_range, center_label=1)
        case_num = case_num + 1    

def resample_tcia_parallel(src_path, dst_path, resolution, padding, pad_value, cropped_vol_size, clamp_range):
    import threading
    thr1 = threading.Thread(target=resample_tcia, 
                        args=(src_path, dst_path, resolution, padding, pad_value, cropped_vol_size, clamp_range, np.arange(0,13)), 
                        kwargs={})
    thr2 = threading.Thread(target=resample_tcia, 
                            args=(src_path, dst_path, resolution, padding, pad_value, cropped_vol_size, clamp_range, np.arange(13,26)), 
                            kwargs={})
    thr3 = threading.Thread(target=resample_tcia, 
                            args=(src_path, dst_path, resolution, padding, pad_value, cropped_vol_size, clamp_range, np.arange(26,40)), 
                            kwargs={})
    thr4 = threading.Thread(target=resample_tcia, 
                            args=(src_path, dst_path, resolution, padding, pad_value, cropped_vol_size, clamp_range, np.arange(40,54)), 
                            kwargs={})
    thr5 = threading.Thread(target=resample_tcia, 
                            args=(src_path, dst_path, resolution, padding, pad_value, cropped_vol_size, clamp_range, np.arange(54,68)), 
                            kwargs={})
    thr6 = threading.Thread(target=resample_tcia, 
                            args=(src_path, dst_path, resolution, padding, pad_value, cropped_vol_size, clamp_range, np.arange(68,82)), 
                            kwargs={})   
    # start parallel processing threads
    thr1.start()
    thr2.start()
    thr3.start()
    thr4.start()
    thr5.start()
    thr6.start()    
    # wait on threads to finish
    thr1.join() 
    thr2.join() 
    thr3.join() 
    thr4.join() 
    thr5.join() 
    thr6.join() 

def resample_tcia(in_dir, out_dir, res, padding, pad_value, cropped_vol_size, clamp_range, case_indeces):
    if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)
    
    imgdir_list = sorted(os.listdir(in_dir + '/DOI/'))
    lbl_list = sorted(os.listdir(in_dir + '/TCIA_pancreas_labels-02-05-2017/'))
    imgdir_list = [imgdir_list[i] for i in case_indeces]
    lbl_list = [lbl_list[i] for i in case_indeces]
    case_num = 0

    for img_dir, lbl_fname in zip(imgdir_list, lbl_list):
        print('Processing %s and %s' % (img_dir, lbl_fname))
        dicom_path = in_dir + '/DOI/' + img_dir
        dicom_path = dicom_path + '/' + os.listdir(dicom_path)[0]
        dicom_path = dicom_path + '/' + os.listdir(dicom_path)[0] + '/'
        print(dicom_path)
        convert_dicom_to_nii(dicom_path, '%s/img%d.nii.gz' % (out_dir, case_indeces[case_num]))
        convert_nii_to_rai('%s/img%d.nii.gz' % (out_dir, case_indeces[case_num]), '%s/img%d.nii.gz' % (out_dir, case_indeces[case_num]), dtype=np.int16)
        copyfile('%s/TCIA_pancreas_labels-02-05-2017/%s' %(in_dir, lbl_fname), '%s/lbl%d.nii.gz' % (out_dir, case_indeces[case_num]))
        convert_nii_to_rai('%s/lbl%d.nii.gz' % (out_dir, case_indeces[case_num]), '%s/lbl%d.nii.gz' % (out_dir, case_indeces[case_num]), dtype=np.uint8)
        #print('pad_value1: ', pad_value)
        resample_and_crop_volumes(img_fname = '%simg%d.nii.gz' % (out_dir, case_indeces[case_num]), lbl_fname = '%slbl%d.nii.gz' % (out_dir, case_indeces[case_num]), 
            cropped_vol_size = cropped_vol_size, resolution = res, padding = padding, pad_value = pad_value, clamp_range = clamp_range, center_label=1)
        case_num = case_num + 1    

def resample_pancreas_manual(in_dir, out_dir, res, padding, pad_value, cropped_vol_size, clamp_range, case_indeces):
    if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)
    
    imgdir_list = sorted(os.listdir(in_dir))
    print(imgdir_list)
    imgdir_list = [imgdir_list[i] for i in case_indeces]
    case_num = 0

    for img_dir in imgdir_list:
        print('Processing %s' % (img_dir))
        dicom_path = in_dir + img_dir
        print(dicom_path)
        convert_dicom_to_nii(dicom_path, '%s/img%d.nii.gz' % (out_dir, case_indeces[case_num]))
        convert_nii_to_rai('%s/img%d.nii.gz' % (out_dir, case_indeces[case_num]), '%s/img%d.nii.gz' % (out_dir, case_indeces[case_num]), dtype=np.int16)
        copyfile('%s/pancreas.nii.gz' %(dicom_path), '%s/lbl%d.nii.gz' % (out_dir, case_indeces[case_num]))
        convert_nii_to_rai('%s/lbl%d.nii.gz' % (out_dir, case_indeces[case_num]), '%s/lbl%d.nii.gz' % (out_dir, case_indeces[case_num]), dtype=np.uint8)
        resample_and_crop_volumes(img_fname = '%simg%d.nii.gz' % (out_dir, case_indeces[case_num]), lbl_fname = '%slbl%d.nii.gz' % (out_dir, case_indeces[case_num]), 
            cropped_vol_size = cropped_vol_size, resolution = res, padding = padding, pad_value = pad_value, clamp_range = clamp_range, center_label=1)
        case_num = case_num + 1 


def PadNiiVolumeToSize(inFile, outFile, paddedVolSize, padValue, dataType):
    # get the dimensions of the existing volume
    cmd = [c3d, '%s' % (inFile), '-info']
    cmdOutput = check_output(cmd)
    regexp = re.compile(r'dim = \[(\d+), (\d+), (\d+)\]') 
    result = regexp.search(str(cmdOutput))
    volSize = [int(result.group(1)), int(result.group(2)), int(result.group(3))]   

    # calculate how many voxels to add on each size of the volume
    padBelow = [0, 0, 0]
    padAbove = [0, 0, 0]
    padBelow[0] = floor((paddedVolSize[0] - volSize[0]) / 2.0)
    padAbove[0] = ceil((paddedVolSize[0] - volSize[0]) / 2.0)
    padBelow[1] = floor((paddedVolSize[1] - volSize[1]) / 2.0)
    padAbove[1] = ceil((paddedVolSize[1] - volSize[1]) / 2.0)
    padBelow[2] = floor((paddedVolSize[2] - volSize[2]) / 2.0)
    padAbove[2] = ceil((paddedVolSize[2] - volSize[2]) / 2.0)  

    cmd = '%s %s -pad %dx%dx%dvox %dx%dx%dvox %.1f -type %s -o %s' \
        % (c3d, inFile, padBelow[0], padBelow[1], padBelow[2], padAbove[0], padAbove[1], padAbove[2], padValue, dataType, outFile)
    print(cmd)
    if (padBelow[0]>=0 and padBelow[1]>=0 and padBelow[2]>=0 and \
        padAbove[0]>=0 and padAbove[1]>=0 and padAbove[2]>=0):
        run(shlex.split(cmd))
    else:
        print('Error: Increase requested padded volume dimensions!!!')
        exit()

def CropNiiVolumeToSize(inFile, outFile, croppedVolSize, dataType):
    # get the dimensions of the existing volume
    cmd = [c3d, '%s' % (inFile), '-info']
    cmdOutput = check_output(cmd)
    regexp = re.compile(r'dim = \[(\d+), (\d+), (\d+)\]') 
    result = regexp.search(str(cmdOutput))
    volSize = [int(result.group(1)), int(result.group(2)), int(result.group(3))]   

    # calculate how many voxels to add on each size of the volume
    cropBoxCorner = [0, 0, 0]
    cropBoxSize = [0, 0, 0]
    cropBoxCorner[0] = floor((volSize[0] - croppedVolSize[0]) / 2.0)
    cropBoxCorner[1] = floor((volSize[1] - croppedVolSize[1]) / 2.0)
    cropBoxCorner[2] = floor((volSize[2] - croppedVolSize[2]) / 1.0) # crop from the top of the image

    cmd = '%s %s -region %dx%dx%dvox %dx%dx%dvox -type %s -o %s' \
        % (c3d, inFile, cropBoxCorner[0], cropBoxCorner[1], cropBoxCorner[2], croppedVolSize[0], croppedVolSize[1], croppedVolSize[2], dataType, outFile)
    print(cmd)
    run(shlex.split(cmd))

def CreateCombinedLabelVolume(inDir, outFile, structureNames, datType):
    for struct in structureNames:
        cmd = '%s %s -type %s -o %s' \
        % (c3d, '%s/%s.nrrd' % (inDir, struct), datType, '%s/%s.nii' % (inDir, struct))
        print(cmd)
        run(shlex.split(cmd))

    # get the dimensions of the existing volume
    cmd = [c3d, ('%s/%s.nii' % (inDir, struct)), '-info']
    cmdOutput = check_output(cmd)
    regexp = re.compile(r'dim = \[(\d+), (\d+), (\d+)\]') 
    result = regexp.search(str(cmdOutput))
    volSize = [int(result.group(1)), int(result.group(2)), int(result.group(3))]   
    
    lblVol = np.zeros(volSize, dtype=np.uint8)
    lblNum = 1

    for struct in structureNames:
        curVol = nib.load('%s/%s.nii' % (inDir, struct))
        Taffine = curVol.get_affine()
        ti = (curVol.get_data() > 0.5)
        lblVol[ti] = lblNum
        lblNum = lblNum + 1 
    
    # save the combined label volume
    nib.Nifti1Image(lblVol, Taffine).to_filename(outFile)


def create_combined_label_volume(in_dir, out_file, structure_names, dat_type):
    cur_vol = SimpleITK.ReadImage('%s/%s.nrrd' % (in_dir, structure_names[0]))
    lbl_vol = np.zeros(cur_vol.GetSize(), dtype=dat_type)
    lbl_num = 1
    for struct in structure_names:
        fname = '%s/%s.nrrd' % (in_dir, struct)
        if os.path.isfile(fname):
            cur_vol = SimpleITK.ReadImage(fname)
            lbl = SimpleITK.GetArrayFromImage(cur_vol)
            lbl = np.transpose(lbl, (2,1,0))
            lbl = np.flip(lbl, axis=1)
            ti = (lbl > 0.5)
            lbl_vol[ti] = lbl_num
        lbl_num = lbl_num + 1     
    # save the combined label volume
    volume.save_to_nii(lbl_vol.astype(dat_type), out_file, cur_vol.GetSpacing(), cur_vol.GetOrigin())

def resample_and_crop_volumes(img_fname, lbl_fname, cropped_vol_size, resolution=[1,1,1], padding=[100,100,100], pad_value=-1024, clamp_range=[-1024,16535], center_label=-1):

    cropped_vol_size = np.array(cropped_vol_size, dtype=np.int32)
    resolution = np.array(resolution, dtype=np.float)

    lbl_img = SimpleITK.ReadImage(lbl_fname)
    img_img = SimpleITK.ReadImage(img_fname)

    caster = SimpleITK.CastImageFilter()
    caster.SetOutputPixelType(SimpleITK.sitkUInt8)
    lbl_img = caster.Execute(lbl_img)    
    caster.SetOutputPixelType(SimpleITK.sitkInt16)
    img_img = caster.Execute(img_img)

    # clamp
    clamper = SimpleITK.ClampImageFilter()
    clamper.SetLowerBound(clamp_range[0])
    clamper.SetUpperBound(clamp_range[1])
    img_img = clamper.Execute(img_img)

    # resample
    factor = np.asarray(lbl_img.GetSpacing()) / resolution
    factor_size = np.asarray(lbl_img.GetSize() * factor, dtype=np.float)
    new_size = np.max([factor_size, cropped_vol_size], axis=0)    
    new_size = new_size.astype(dtype=np.int32)
    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(lbl_img)
    resampler.SetOutputSpacing(resolution)
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(SimpleITK.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    lbl_resampled = resampler.Execute(lbl_img)
    resampler.SetInterpolator(SimpleITK.sitkLinear)
    resampler.SetDefaultPixelValue(pad_value)
    img_resampled = resampler.Execute(img_img)

    # pad
    padder = SimpleITK.ConstantPadImageFilter()
    padder.SetPadLowerBound(padding)
    padder.SetPadUpperBound(padding)
    padder.SetConstant(0)
    lbl_resampled = padder.Execute(lbl_resampled)
    
    #Bose reverting back to ConstantPadImageFiler because in the generated image seeing truncated volume on the border
    padder2 = SimpleITK.ConstantPadImageFilter()
    #padder2 = SimpleITK.MirrorPadImageFilter() 
    padder2.SetPadLowerBound(padding)
    padder2.SetPadUpperBound(padding)
    print('Pad value: ', pad_value)
    #padder2.SetConstant(pad_value)
    img_resampled = padder2.Execute(img_resampled)
    print('padded img size: ', img_resampled.GetSize())
    
    # compute the centroid of labels
    lbl = np.transpose(SimpleITK.GetArrayFromImage(lbl_resampled))
    mask = np.zeros(lbl.shape)
    if center_label == -1:
        mask[lbl > 0.5] = 1
    else:
        mask[lbl == center_label] = 1
    lbl_centroid = np.array(ndimage.measurements.center_of_mass(mask))
    print('Label centroid: ', lbl_centroid)
    print('# of nonzeros: ', np.count_nonzero(mask))
 
    region_extractor = SimpleITK.RegionOfInterestImageFilter()
    region_start_voxel = (lbl_centroid - np.array(cropped_vol_size) / 2.0).astype(dtype=int)
    region_extractor.SetSize(cropped_vol_size.astype(dtype=np.int32).tolist())
    region_extractor.SetIndex(region_start_voxel.tolist())
    lbl_resampled_cropped = region_extractor.Execute(lbl_resampled)
    img_resampled_cropped = region_extractor.Execute(img_resampled)
    
    SimpleITK.WriteImage(lbl_resampled_cropped, lbl_fname)
    #SimpleITK.WriteImage(img_resampled_cropped, img_fname)

    caster.SetOutputPixelType(SimpleITK.sitkInt16)
    img_out = caster.Execute(img_resampled_cropped)
    SimpleITK.WriteImage(img_out, img_fname)

def resample_and_crop_volumes_in_memory(img_img, lbl_img, cropped_vol_size, resolution=[1,1,1], padding=[100,100,100], pad_value=-1024, clamp_range=[-1024,16535], center_label=-1):

    cropped_vol_size = np.array(cropped_vol_size, dtype=np.int32)
    resolution = np.array(resolution, dtype=np.float)

    caster = SimpleITK.CastImageFilter()
    caster.SetOutputPixelType(SimpleITK.sitkUInt8)
    lbl_img = caster.Execute(lbl_img)    
    caster.SetOutputPixelType(SimpleITK.sitkInt16)
    img_img = caster.Execute(img_img)

    # clamp
    clamper = SimpleITK.ClampImageFilter()
    clamper.SetLowerBound(clamp_range[0])
    clamper.SetUpperBound(clamp_range[1])
    img_img = clamper.Execute(img_img)

    # resample
    factor = np.asarray(lbl_img.GetSpacing()) / resolution
    factor_size = np.asarray(lbl_img.GetSize() * factor, dtype=np.float)        
    new_size = np.max([factor_size, cropped_vol_size], axis=0)    
    new_size = new_size.astype(dtype=np.int32)
    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(lbl_img)
    resampler.SetOutputSpacing(resolution)
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(SimpleITK.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    lbl_resampled = resampler.Execute(lbl_img)
    resampler.SetInterpolator(SimpleITK.sitkLinear)
    resampler.SetDefaultPixelValue(pad_value)
    img_resampled = resampler.Execute(img_img)

    # pad
    padder = SimpleITK.ConstantPadImageFilter()
    padder.SetPadLowerBound(padding)
    padder.SetPadUpperBound(padding)
    padder.SetConstant(0)
    lbl_resampled = padder.Execute(lbl_resampled)
    print('Pad value: ', pad_value)
    padder.SetConstant(pad_value)
    img_resampled = padder.Execute(img_resampled)
    print('padded img size: ', img_resampled.GetSize())
    
    # compute the centroid of labels
    lbl = np.transpose(SimpleITK.GetArrayFromImage(lbl_resampled))
    mask = np.zeros(lbl.shape)
    if center_label == -1:
        mask[lbl > 0.5] = 1
    else:
        mask[lbl == center_label] = 1
    lbl_centroid = np.array(ndimage.measurements.center_of_mass(mask))
    print('Label centroid: ', lbl_centroid)
 
    region_extractor = SimpleITK.RegionOfInterestImageFilter()
    region_start_voxel = (lbl_centroid - np.array(cropped_vol_size) / 2.0).astype(dtype=int)
    region_extractor.SetSize(cropped_vol_size.astype(dtype=np.int32).tolist())
    region_extractor.SetIndex(region_start_voxel.tolist())
    lbl_resampled_cropped = region_extractor.Execute(lbl_resampled)
    img_resampled_cropped = region_extractor.Execute(img_resampled)
    
    caster.SetOutputPixelType(SimpleITK.sitkInt16)
    img_out = caster.Execute(img_resampled_cropped)

    return img_out, lbl_resampled_cropped

def resample_and_crop_volumes_IEV(img_fname, lbl_fname, cropped_vol_size, resolution=[1,1,1], padding=[100,100,100], pad_value=-1024, clamp_range=[-1024,16535], center_label=True, center_label_coordinates=[0,0,0]):
    #Same as resample_and_crop_volumes, except for a few things:
    # 1) It returns the variable lbl_centroid,
    # 2) The 'if' statement when determining to center the label can handle arrays as input (and correspondingly return False)
    cropped_vol_size = np.array(cropped_vol_size, dtype=np.int32)
    resolution = np.array(resolution, dtype=np.float)

    lbl_img = SimpleITK.ReadImage(lbl_fname)
    img_img = SimpleITK.ReadImage(img_fname)

    caster = SimpleITK.CastImageFilter()
    caster.SetOutputPixelType(SimpleITK.sitkUInt8)
    lbl_img = caster.Execute(lbl_img)    
    caster.SetOutputPixelType(SimpleITK.sitkInt16)
    img_img = caster.Execute(img_img)

    # clamp
    clamper = SimpleITK.ClampImageFilter()
    clamper.SetLowerBound(clamp_range[0])
    clamper.SetUpperBound(clamp_range[1])
    img_img = clamper.Execute(img_img)

    # resample
    factor = np.asarray(lbl_img.GetSpacing()) / resolution
    factor_size = np.asarray(lbl_img.GetSize() * factor, dtype=np.float)
    new_size = np.max([factor_size, cropped_vol_size], axis=0)    
    new_size = new_size.astype(dtype=np.int32)
    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(lbl_img)
    resampler.SetOutputSpacing(resolution)
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(SimpleITK.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    lbl_resampled = resampler.Execute(lbl_img)
    resampler.SetInterpolator(SimpleITK.sitkLinear)
    resampler.SetDefaultPixelValue(pad_value)
    img_resampled = resampler.Execute(img_img)

    # pad
    padder = SimpleITK.ConstantPadImageFilter()
    padder.SetPadLowerBound(padding)
    padder.SetPadUpperBound(padding)
    padder.SetConstant(0)
    lbl_resampled = padder.Execute(lbl_resampled)
    print('Pad value: ', pad_value)
    padder.SetConstant(pad_value)
    img_resampled = padder.Execute(img_resampled)
    print('padded img size: ', img_resampled.GetSize())
    
    # compute the centroid of labels
    lbl = np.transpose(SimpleITK.GetArrayFromImage(lbl_resampled))
    mask = np.zeros(lbl.shape)
    # If the value of center_label is True, set the label coordinates to its centroid 
    # If False, then pass then center on the specified label coordinates in the input
    if center_label==True:
        print('Setting label center to its centroid')
        mask[lbl > 0.5] = 1
        center_label_coordinates = np.array(ndimage.measurements.center_of_mass(mask))
    else:
        print('Setting label center to the specified coordinates')
    
    print('Label centroid: ', center_label_coordinates)
 
    region_extractor = SimpleITK.RegionOfInterestImageFilter()
    region_start_voxel = (center_label_coordinates - np.array(cropped_vol_size) / 2.0).astype(dtype=int)
    region_extractor.SetSize(cropped_vol_size.astype(dtype=np.int32).tolist())
    region_extractor.SetIndex(region_start_voxel.tolist())
    lbl_resampled_cropped = region_extractor.Execute(lbl_resampled)
    img_resampled_cropped = region_extractor.Execute(img_resampled)
    
    SimpleITK.WriteImage(lbl_resampled_cropped, lbl_fname)
    #SimpleITK.WriteImage(img_resampled_cropped, img_fname)

    caster.SetOutputPixelType(SimpleITK.sitkInt16)
    img_out = caster.Execute(img_resampled_cropped)
    SimpleITK.WriteImage(img_out, img_fname)
    return center_label_coordinates


def pickle_data(data_dir):

    imgFileList = sorted(glob.glob(data_dir + '/' + "img*.nii.gz"))
    lblFileList = sorted(glob.glob(data_dir + '/' + "lbl*.nii.gz"))

    caseNum = 1;
    for case in imgFileList:
        print("Processing: " + case)
        img = SimpleITK.ReadImage(case)
        X = SimpleITK.GetArrayFromImage(img)
        X = X.astype(float)

        # normalize 12 bit CT HU range <-1024..3072> to <0..1>
        X = (X + 1024) / 4096

        # reshape to casenum_first, channel_last dimension order
        X = np.transpose(X, (2,1,0))
        Xshape = X.shape
        X = np.reshape(X, (1, Xshape[0], Xshape[1], Xshape[2]))
        print("Dataset size: " + str(X.shape))

        if caseNum==1:
            Xall = X
        else:
            Xall = np.concatenate((Xall, X), axis=0)
        caseNum = caseNum + 1

    print("Xall dataset size: " + str(Xall.shape))
    print("Xall max value accross all cases: " + str(Xall.max()))
    print("Xall min value accross all cases: " + str(Xall.min()))

    caseNum = 1;
    for case in lblFileList:
        print("Processing: " + case)
        img = SimpleITK.ReadImage(case)
        Y = SimpleITK.GetArrayFromImage(img)
        Y = Y.astype(float)

        # reshape to casenum_first, channel_last dimension order
        Y = np.transpose(Y, (2,1,0))
        Yshape = Y.shape
        Y = np.reshape(Y, (1, Yshape[0], Yshape[1], Yshape[2]))
        print("Dataset size: " + str(Y.shape))

        if caseNum==1:
            Yall = Y
        else:
            Yall = np.concatenate((Yall, Y), axis=0)
        caseNum = caseNum + 1

    print("Yall dataset size: " + str(Yall.shape))
    print("Yall max value accross all cases: " + str(Yall.max()))
    print("Yall min value accross all cases: " + str(Yall.min()))

    # dump all data into a single pickle to make future reloading for training and evaluation more efficient
    X = Xall.astype('float32')
    y = Yall.astype('uint8')
    print(X.shape)
    print(y.shape)
    with open(data_dir + '/' + "train_data.p3", 'wb') as f:
        pickle.dump([X, y], f, protocol=4)

def convert_dicom_to_nii(dicomdir, nii_fname):
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicomdir)
    reader.SetFileNames(dicom_names)
    #print(dicom_names)
    image = reader.Execute() 
    size = image.GetSize()
    print( "Converting DICOM to NII. Image size:", size[0], size[1], size[2] ) 
    SimpleITK.WriteImage(image, nii_fname)

def convert_dicom_to_nii_fast(dicomdir, nii_fname):
    import dicom2nifti
    dicom2nifti.dicom_series_to_nifti(dicomdir, output_file=nii_fname, reorient_nifti=False)
    return nii_fname  


def convert_nii_to_rai(fin, fout, dtype):
    img = SimpleITK.ReadImage(fin)

    # standardize image orientation
    direction = img.GetDirection()
    print('Image direction cosines: ', direction)
    X = np.transpose(SimpleITK.GetArrayFromImage(img))
    if (direction[0] > -1.05) and (direction[0] < -0.95):
        X = np.flip(X, axis=0)
    if (direction[4] > -1.05) and (direction[4] < -0.95):
        X = np.flip(X, axis=1)
    if (direction[8] > -1.05) and (direction[8] < -0.95):
        X = np.flip(X, axis=2)
    
    imgnew = SimpleITK.GetImageFromArray(np.transpose(X.astype(dtype)), isVector=False)
    direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    imgnew.SetDirection(direction)
    imgnew.SetOrigin(img.GetOrigin())
    imgnew.SetSpacing(img.GetSpacing())
    SimpleITK.WriteImage(imgnew, fout)


def flip_nii_axis(fin, fout, img_flip, dtype):
    img = SimpleITK.ReadImage(fin)
    X = np.transpose(SimpleITK.GetArrayFromImage(img))
    if img_flip[0]==True:
        X = np.flip(X, axis=0)
    if img_flip[1]==True:
        X = np.flip(X, axis=1)
    if img_flip[2]==True:
        X = np.flip(X, axis=2)
    imgnew = SimpleITK.GetImageFromArray(np.transpose(X.astype(dtype)), isVector=False)
    #imgnew = SimpleITK.GetImageFromArray(X.astype(dtype), isVector=False)
    imgnew.SetDirection(img.GetDirection())
    imgnew.SetOrigin(img.GetOrigin())
    imgnew.SetSpacing(img.GetSpacing())
    SimpleITK.WriteImage(imgnew, fout)

# ----------------------------------------------------------------------------------------


def convert_resample_crop_dicom_to_nii(dicom_dir, nii_fname, cropped_vol_size, shift=[0,0,0], img_flip=[False,False,False], resolution=[1,1,1], padding=[0,0,0], clamp_range=[-900,4096], pad_value=-1000):
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute() 
    size = image.GetSize()
    print( "Converting DICOM to NII. Image size:",size[0],size[1],size[2]) 

    cropped_vol_size = np.array(cropped_vol_size, dtype=np.int32)
    resolution = np.array(resolution, dtype=np.float)

    caster = SimpleITK.CastImageFilter()
    caster.SetOutputPixelType(SimpleITK.sitkInt16)
    image = caster.Execute(image)

    # clamp
    clamper = SimpleITK.ClampImageFilter()
    clamper.SetLowerBound(clamp_range[0])
    clamper.SetUpperBound(clamp_range[1])
    image = clamper.Execute(image)

    # resample
    factor = np.asarray(image.GetSpacing()) / resolution
    factor_size = np.asarray(image.GetSize() * factor, dtype=np.float)
    new_size = np.max([factor_size, cropped_vol_size], axis=0)    
    new_size = new_size.astype(dtype=np.int32)
    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetOutputSpacing(resolution)
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(SimpleITK.sitkLinear)
    resampler.SetDefaultPixelValue(pad_value)
    image = resampler.Execute(image)

    # pad
    padder = SimpleITK.ConstantPadImageFilter()
    padder.SetPadLowerBound(padding)
    padder.SetPadUpperBound(padding)
    padder.SetConstant(pad_value)
    image = padder.Execute(image)

    size = image.GetSize()
    print( "Converting DICOM to NII. Image size:",size[0],size[1],size[2]) 
    region_extractor = SimpleITK.RegionOfInterestImageFilter()
    region_start_voxel = (np.array(size)/2.0 - np.array(cropped_vol_size)/2.0 + np.array(shift)).astype(dtype=int)
    print( "Region start voxel:",region_start_voxel[0],region_start_voxel[1],region_start_voxel[2]) 
    region_extractor.SetSize(cropped_vol_size.astype(dtype=np.int32).tolist())
    region_extractor.SetIndex(region_start_voxel.tolist())
    image = region_extractor.Execute(image)
        
    # flip the image axes to match model expected orientation
    img = SimpleITK.GetArrayFromImage(image)
    #img = np.transpose(img, (2,1,0))
    if img_flip[0]==True:
        img = np.flip(img, axis=0)
    if img_flip[1]==True:
        img = np.flip(img, axis=1)
    if img_flip[2]==True:
        img = np.flip(img, axis=2)
    image = SimpleITK.GetImageFromArray(img, isVector=False)

    SimpleITK.WriteImage(image, nii_fname)

def convert_resample_crop_dicom_slice_to_nii_2D(dicom_fname, nii_fname, cropped_slice_size, shift=[0,0], img_flip=[False,False], resolution=[1,1], padding=[0,0], clamp_range=[-900,4096], pad_value=-1000):
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute() 
    size = image.GetSize()
    print( "Converting DICOM to NII. Image size:",size[0],size[1],size[2]) 

    cropped_vol_size = np.array(cropped_vol_size, dtype=np.int32)
    resolution = np.array(resolution, dtype=np.float)

    caster = SimpleITK.CastImageFilter()
    caster.SetOutputPixelType(SimpleITK.sitkInt16)
    image = caster.Execute(image)

    # clamp
    clamper = SimpleITK.ClampImageFilter()
    clamper.SetLowerBound(clamp_range[0])
    clamper.SetUpperBound(clamp_range[1])
    image = clamper.Execute(image)

    # resample
    factor = np.asarray(image.GetSpacing()) / resolution
    factor_size = np.asarray(image.GetSize() * factor, dtype=np.float)
    new_size = np.max([factor_size, cropped_vol_size], axis=0)    
    new_size = new_size.astype(dtype=np.int32)
    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetOutputSpacing(resolution)
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(SimpleITK.sitkLinear)
    resampler.SetDefaultPixelValue(pad_value)
    image = resampler.Execute(image)

    # pad
    padder = SimpleITK.ConstantPadImageFilter()
    padder.SetPadLowerBound(padding)
    padder.SetPadUpperBound(padding)
    padder.SetConstant(pad_value)
    image = padder.Execute(image)

    size = image.GetSize()
    print( "Converting DICOM to NII. Image size:",size[0],size[1],size[2]) 
    region_extractor = SimpleITK.RegionOfInterestImageFilter()
    region_start_voxel = (np.array(size)/2.0 - np.array(cropped_vol_size)/2.0 + np.array(shift)).astype(dtype=int)
    print( "Region start voxel:",region_start_voxel[0],region_start_voxel[1],region_start_voxel[2]) 
    region_extractor.SetSize(cropped_vol_size.astype(dtype=np.int32).tolist())
    region_extractor.SetIndex(region_start_voxel.tolist())
    image = region_extractor.Execute(image)
        
    # flip the image axes to match model expected orientation
    img = SimpleITK.GetArrayFromImage(image)
    #img = np.transpose(img, (2,1,0))
    if img_flip[0]==True:
        img = np.flip(img, axis=0)
    if img_flip[1]==True:
        img = np.flip(img, axis=1)
    if img_flip[2]==True:
        img = np.flip(img, axis=2)
    image = SimpleITK.GetImageFromArray(img, isVector=False)

    SimpleITK.WriteImage(image, nii_fname)
# ----------------------------------------------------------------------------------------

def convert_nrrd_to_nii(fin, fout):
    img = SimpleITK.ReadImage(fin)
    SimpleITK.WriteImage(img, fout)

def reindex_data(data_dir, start_index):    
    img_list = sorted(glob.glob(data_dir + '/' + "img*.nii.gz"))
    lbl_list = sorted(glob.glob(data_dir + '/' + "lbl*.nii.gz"))
    new_indeces = range(start_index, start_index+len(img_list))
    i = 0
    for img_fname, lbl_fname in zip(img_list, lbl_list):
        print('Reindexing: ', img_fname)
        print('Reindexing: ', lbl_fname)
        os.rename(img_fname, data_dir + '/img%d.nii.gz' % (new_indeces[i]))
        os.rename(lbl_fname, data_dir + '/lbl%d.nii.gz' % (new_indeces[i]))
        i = i + 1

def convert_to_single_label(data_dir, target_label):
    lbl_list = sorted(glob.glob(data_dir + '/' + "lbl*.nii.gz"))
    for lbl_fname in zip(lbl_list):
        print('Relabeling: ', lbl_fname[0])
        input_lbl_vol = SimpleITK.ReadImage(lbl_fname[0])
        lbl = SimpleITK.GetArrayFromImage(input_lbl_vol)
        ti = (lbl == target_label)
        lbl = np.zeros(lbl.shape, dtype=np.uint8)
        lbl[ti] = 1
        output_lbl_vol = SimpleITK.GetImageFromArray(lbl, isVector=False)
        output_lbl_vol.SetOrigin(input_lbl_vol.GetOrigin())
        output_lbl_vol.SetSpacing(input_lbl_vol.GetSpacing())
        output_lbl_vol.SetDirection(input_lbl_vol.GetDirection())
        SimpleITK.WriteImage(output_lbl_vol, lbl_fname[0])

def convert_nii_to_single_label(input_fname, output_fname, target_label):
        print('Relabeling: ', input_fname)
        input_lbl_vol = SimpleITK.ReadImage(input_fname)
        lbl = SimpleITK.GetArrayFromImage(input_lbl_vol)
        ti = (lbl == target_label)
        lbl = np.zeros(lbl.shape, dtype=np.uint8)
        lbl[ti] = 1
        output_lbl_vol = SimpleITK.GetImageFromArray(lbl, isVector=False)
        output_lbl_vol.SetOrigin(input_lbl_vol.GetOrigin())
        output_lbl_vol.SetSpacing(input_lbl_vol.GetSpacing())
        output_lbl_vol.SetDirection(input_lbl_vol.GetDirection())
        SimpleITK.WriteImage(output_lbl_vol, output_fname)

def combine_label_volumes(gt_dir, autoseg_dir, out_dir, highest_label = 10):
    gt_list = sorted(glob.glob(gt_dir + '/lbl*.nii.gz'))
    autoseg_list = sorted(glob.glob(autoseg_dir + '/lbl*.nii.gz'))
    
    i = 0
    for gt_fname, as_fname in zip(gt_list, autoseg_list):        
        print('Processing: ', gt_fname)
        print('Processing: ', as_fname)
        y_gt = volume.load_nii(gt_fname)
        y_as = volume.load_nii(as_fname)
        y = np.zeros(y_gt.shape, dtype=np.uint8)
        
        for label_num in range(1, highest_label):
            if label_num in y_gt:
                print('y_gt contains label ', label_num)
                ti = (y_gt == label_num)
            else:
                print('Getting autoseg result for label ', label_num)
                ti = (y_as == label_num)
            y[ti] = label_num
                
        volume.save_to_nii2(y, out_dir + '/' + os.path.basename(gt_fname))
        i += 1        

# ----------------------------------- Phil's data_util functions ----------------------------------

def append_file_extension(directory,ext):
    #Add the .dcm extension to files that don't already have it.
    folder=os.path.join(directory)
    for filename in os.listdir(folder):
        infilename = os.path.join(folder,filename)
        if not infilename.endswith(ext):
            newname = infilename+ext
            os.rename(infilename, newname)

def nii_slices_to_volume(slice_dir, vol_file):
        
        # Sort paths by integer in the filename rather than string (such that 10 comes after 9 rather than 1)
        # https://stackoverflow.com/questions/33159106/sort-filenames-in-directory-in-ascending-order with solution for Python3 comment
        slice_paths = sorted(glob.glob(os.path.join(slice_dir, '*.nii.gz')), key=lambda f: int(''.join(filter(str.isdigit, f))))

        imgs = []

        for i in range(0,len(slice_paths)):

            img_path= slice_paths[i]

            img = nib.load(img_path)
            img_array = np.array(img.dataobj)
            img_array = img_array[0,:,:,0]

            imgs.append(img_array)


        imgs_3D=np.array(imgs)

        #Move z-axis from first index to last index
        imgs_3D = np.rollaxis(imgs_3D, 0, 3)


        
        print(imgs_3D.shape)

        print('Saving 3D image')
        data = nib.nifti1.Nifti1Image(imgs_3D, None)
        nib.save(data, vol_file)
    
def combine_rtss_files(rtss_fname_inputs, rtss_fname_output):

    RTROIObservationsSequences = []
    StructureSetROISequences = []
    ROIContourSequences = []

    for rtss_fname_input in rtss_fname_inputs:

        # load RTSS and verify it is proper RTSS object
        rtss_input = pydicom.read_file(rtss_fname_input) 

        if not (rtss_input.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3'):
            print('Error: Not a DICOM RTSS file.')
            exit()

        #Copy all elements from the 3 attributes associated with rtss objects. Store in a temporary list.
        print('Copying rtss attributes from %s'%rtss_fname_input)

        if hasattr(rtss_input, 'RTROIObservationsSequence'):   
            for i in range(0, len(rtss_input.RTROIObservationsSequence)):
                RTROIObservationsSequence = rtss_input.RTROIObservationsSequence[i]
                RTROIObservationsSequences.append(RTROIObservationsSequence)

        if hasattr(rtss_input, 'StructureSetROISequence'):   
            for i in range(0, len(rtss_input.StructureSetROISequence)):
                StructureSetROISequence = rtss_input.StructureSetROISequence[i]
                StructureSetROISequences.append(StructureSetROISequence)

        if hasattr(rtss_input, 'ROIContourSequence'):
            for i in range(0, len(rtss_input.ROIContourSequence)):
                ROIContourSequence = rtss_input.ROIContourSequence[i]
                ROIContourSequences.append(ROIContourSequence)
    
    # "Copy" all information from the first rtss input file to a new rtss dataset
    rtss_output = pydicom.read_file(rtss_fname_input[0])

    #Overwrite the above 3 rtss attributes in the new dataset
    for i in range(0, len(RTROIObservationsSequences)):
        print(i)
        rtss_output.RTROIObservationsSequence[i] = RTROIObservationsSequences[i]

    for j in range(0, len(StructureSetROISequences)):
        rtss_output.StructureSetROISequence[j] = StructureSetROISequences[j]

    for k in range(0, len(ROIContourSequences)):
        rtss_output.ROIContourSequence[k] = ROIContourSequences[k]
    
    #save output rtss dataset as an rtss file
    rtss_output.save_as(rtss_fname_output)
    


                