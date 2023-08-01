from __future__ import annotations
from os.path import join, exists
from os import sep, mkdir,getcwd
import sys
parent_dir = getcwd()
sys.path.append(parent_dir)

from tifffile import imread, imwrite
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pystackreg import StackReg
from ImageAnalysis_pipeline.pipeline.classes import Experiment
from ImageAnalysis_pipeline.pipeline.loading_data import load_stack
from typing import Iterable


def _chan_shift_file_name(file_list: list, channel_list: list, reg_channel: str)-> list:
    """Return a list of tuples of file names to be registered. 
    The first element of the tuple is the reference image and the second is the image to be registered.
    """
    d = {chan: [file for file in file_list if chan in file] for chan in channel_list}
    chan_2b_process = channel_list.copy()
    chan_2b_process.remove(reg_channel)
    tuples_list = []
    for chan in chan_2b_process:
        ref_list = sorted(d[reg_channel])
        chan_list = sorted(d[chan])
        for i in range(len(ref_list)):
            tuples_list.append((ref_list[i],chan_list[i]))
    return tuples_list

def _reg_mtd(reg_mtd: str)-> StackReg:
    mtd_list = ['translation','rigid_body','scaled_rotation','affine','bilinear']
    if reg_mtd not in mtd_list:
        raise ValueError(f"{reg_mtd} is not valid. Please only put {mtd_list}")
        
    if reg_mtd=='translation':       stackreg = StackReg(StackReg.TRANSLATION)
    elif reg_mtd=='rigid_body':      stackreg = StackReg(StackReg.RIGID_BODY)
    elif reg_mtd=='scaled_rotation': stackreg = StackReg(StackReg.SCALED_ROTATION)
    elif reg_mtd=='affine':          stackreg = StackReg(StackReg.AFFINE)
    elif reg_mtd=='bilinear':        stackreg = StackReg(StackReg.BILINEAR)
    return stackreg

def _correct_chan_shift(input_data: list)-> None:
    # Unpack input data
    stackreg,file_list = input_data
    
    # Load ref_img (which is not contained in chan_list)
    ref_img = imread(file_list[0])
    # Load img
    img_file = file_list[1]
    img = imread(img_file)
    # Apply transfo
    reg_img = stackreg.register_transform(ref_img,img)
    # Save
    reg_img[reg_img<0] = 0
    imwrite(img_file,reg_img.astype(np.uint16))

def _register_with_first(stackreg: StackReg, exp_set: Experiment, reg_channel: str, img_folder: str)-> None:
    # Load ref image
    img_ref = load_stack(img_list=exp_set.processed_images_list,channel_list=[reg_channel],frame_range=[0])
    if exp_set.img_properties.n_slices>1: img_ref = np.amax(img_ref,axis=0)
    
    serie = int(exp_set.exp_path.split(sep)[-1].split('_')[-1][1:])
    for f in range(exp_set.img_properties.n_frames):
        # Load image to register
        img = load_stack(img_list=exp_set.processed_images_list,channel_list=[reg_channel],frame_range=[f])
        if exp_set.img_properties.n_slices>1: img = np.amax(img,axis=0)
        # Get the transfo matrix
        tmats = stackreg.register(ref=img_ref,mov=img)
        
        for chan in exp_set.active_channel_list:
            # Load image to transform
            img = load_stack(img_list=exp_set.processed_images_list,channel_list=[chan],frame_range=[f])
            for z in range(exp_set.img_properties.n_slices):
                # Apply transfo
                if exp_set.img_properties.n_slices==1: reg_img = stackreg.transform(mov=img,tmat=tmats)
                else: reg_img = stackreg.transform(mov=img[z,...],tmat=tmats)
                # Save
                reg_img[reg_img<0] = 0
                imwrite(join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(f+1)+'_z%04d.tif'%(z+1)),reg_img.astype(np.uint16))

def _register_with_mean(stackreg: StackReg, exp_set: Experiment, reg_channel: str, img_folder: str)-> None:
    # Load ref image
    if exp_set.img_properties.n_slices==1: img_ref = np.mean(load_stack(img_list=exp_set.processed_images_list,channel_list=[reg_channel],frame_range=range(exp_set.img_properties.n_frames)),axis=0)
    else: img_ref = np.mean(np.amax(load_stack(img_list=exp_set.processed_images_list,channel_list=[reg_channel],frame_range=range(exp_set.img_properties.n_frames)),axis=1),axis=0)

    serie = int(exp_set.exp_path.split(sep)[-1].split('_')[-1][1:])
    for f in range(exp_set.img_properties.n_frames):
        # Load image to register
        img = load_stack(img_list=exp_set.processed_images_list,channel_list=[reg_channel],frame_range=[f])
        if exp_set.img_properties.n_slices>1: img = np.amax(img,axis=0)
        # Get the transfo matrix
        tmats = stackreg.register(ref=img_ref,mov=img)
        
        for chan in exp_set.active_channel_list:
            # Load image to transform
            img = load_stack(img_list=exp_set.processed_images_list,channel_list=[chan],frame_range=[f])
            for z in range(exp_set.img_properties.n_slices):
                # Apply transfo
                if exp_set.img_properties.n_slices==1: reg_img = stackreg.transform(mov=img,tmat=tmats)
                else: reg_img = stackreg.transform(mov=img[z,...],tmat=tmats)
                # Save
                reg_img[reg_img<0] = 0
                imwrite(join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(f+1)+'_z%04d.tif'%(z+1)),reg_img.astype(np.uint16))

def _register_with_previous(stackreg: StackReg, exp_set: Experiment, reg_channel: str, img_folder: str)-> None:
    for f in range(1,exp_set.img_properties.n_frames):
        # Load ref image
        if f==1:
            img_ref = load_stack(img_list=exp_set.processed_images_list,channel_list=[reg_channel],frame_range=[f-1])
            if exp_set.img_properties.n_slices>1: img_ref = np.amax(img_ref,axis=0)
        else:
            img_ref = load_stack(img_list=exp_set.register_images_list,channel_list=[reg_channel],frame_range=[f-1])
            if exp_set.img_properties.n_slices>1: img_ref = np.amax(img_ref,axis=0)
        
        # Load image to register
        img = load_stack(img_list=exp_set.processed_images_list,channel_list=[reg_channel],frame_range=[f])
        if exp_set.img_properties.n_slices>1: img = np.amax(img,axis=0)
        
        # Get the transfo matrix
        serie = int(exp_set.exp_path.split(sep)[-1].split('_')[-1][1:])
        tmats = stackreg.register(ref=img_ref,mov=img)
        for chan in exp_set.active_channel_list:
            # Load image to transform
            img = load_stack(img_list=exp_set.processed_images_list,channel_list=[chan],frame_range=[f])
            fst_img = load_stack(img_list=exp_set.processed_images_list,channel_list=[chan],frame_range=[f-1])
            for z in range(exp_set.img_properties.n_slices):
                # Copy the first image to the reg_folder
                if f==1:
                    if exp_set.img_properties.n_slices==1: imwrite(join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(1)+'_z%04d.tif'%(z+1)),fst_img.astype(np.uint16))
                    else: imwrite(join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(1)+'_z%04d.tif'%(z+1)),fst_img[z,...].astype(np.uint16))
                # Apply transfo
                if exp_set.img_properties.n_slices==1: reg_img = stackreg.transform(mov=img,tmat=tmats)
                else: reg_img = stackreg.transform(mov=img[z,...],tmat=tmats)
                # Save
                reg_img[reg_img<0] = 0
                imwrite(join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(f+1)+'_z%04d.tif'%(z+1)),reg_img.astype(np.uint16))

# # # # # # # # main functions # # # # # # # # # 
def channel_shift_register(exp_set_list: list[Experiment], reg_mtd: str, reg_channel: str, chan_shift_overwrite: bool=False)-> list[Experiment]:
    for exp_set in exp_set_list:
        if exp_set.process.channel_shift_corrected and not chan_shift_overwrite:
            print(f"--> Channel shift was already apply on {exp_set.exp_path}")
            continue
        stackreg = _reg_mtd(reg_mtd)
        print(f"--> Applying channel shift correction on {exp_set.exp_path}")
        
        # Generate input data for parallel processing
        img_group_list = _chan_shift_file_name(exp_set.processed_images_list,exp_set.active_channel_list,reg_channel)
        input_data = [(stackreg,img_list) for img_list in img_group_list]
                
        with ProcessPoolExecutor() as executor:
            executor.map(_correct_chan_shift,input_data)
        # Save settings
        exp_set.process.channel_shift_corrected = [f"reg_channel={reg_channel}",f"reg_mtd={reg_mtd}"]
        exp_set.save_as_json()
    return exp_set_list

def register_img(exp_set_list: list[Experiment], reg_channel: str, reg_mtd: str, reg_ref: int, reg_overwrite: bool=False)-> list[Experiment]:
    for exp_set in exp_set_list:
        img_folder = join(sep,exp_set.exp_path+sep,'Images_Registered')
        if not exists(img_folder):
            mkdir(img_folder)
        
        if exp_set.process.img_registered and not reg_overwrite:
            print(f"--> Registration was already apply on {exp_set.exp_path}")
            continue
        
        stackreg = _reg_mtd(reg_mtd)
        if reg_ref=='first':
            print(f"--> Registering {exp_set.exp_path} with first image and {reg_mtd} method")
            _register_with_first(stackreg,exp_set,reg_channel,img_folder)
        elif reg_ref=='previous':
            print(f"--> Registering {exp_set.exp_path} with previous image and {reg_mtd} method")
            _register_with_previous(stackreg,exp_set,reg_channel,img_folder)
        elif reg_ref=='mean':
            print(f"--> Registering {exp_set.exp_path} with mean image and {reg_mtd} method")
            _register_with_mean(stackreg,exp_set,reg_channel,img_folder)
        exp_set.process.img_registered = [f"reg_channel={reg_channel}",f"reg_mtd={reg_mtd}",f"reg_ref={reg_ref}"]
        exp_set.save_as_json()
    return exp_set_list