from __future__ import annotations
from os import sep,mkdir,scandir,getcwd
from os.path import join, exists
import sys
parent_dir = getcwd() 
# Add the parent to sys.pah
sys.path.append(parent_dir)
from ImageAnalysis_pipeline.pipeline.classes import init_from_dict,init_from_json,Experiment

from nd2reader import ND2Reader
from tifffile import imwrite, imread
import numpy as np
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from ImageAnalysis_pipeline.pipeline.pre_process.metadata import get_metadata

 
def _name_img_list(meta_dict: dict)-> list[str]:
    """Return a list of generated image names based on the metadata of the experiment"""
    # Create a name for each image
    img_name_list = []
    for serie in range(meta_dict['n_series']):
        for t in range(meta_dict['n_frames']):
            for z in range(meta_dict['n_slices']):
                for chan in meta_dict['active_channel_list']:
                    img_name_list.append(chan+'_s%02d'%(serie+1)+'_f%04d'%(t+1)+'_z%04d'%(z+1))
    return img_name_list

def _write_ND2(img_data: list)-> None:
    # Unpack img_data
    meta,img_name = img_data
    img_obj = ND2Reader(meta['img_path'])
    serie,frame,z_slice = [int(i[1:])-1 for i in img_name.split('_')[1:]]
    chan = meta['full_channel_list'].index(img_name.split('_')[0])
    
    # Get the image       
    if meta['n_slices']>1: 
        img = img_obj.get_frame_2D(c=chan,t=frame,z=z_slice,x=meta['img_width'],y=meta['img_length'],v=serie)
    else: img = img_obj.get_frame_2D(c=chan,t=frame,x=meta['img_width'],y=meta['img_length'],v=serie)
    # Save
    im_folder = join(sep,meta['exp_path_list'][serie]+sep,'Images')
    imwrite(join(sep,im_folder+sep,img_name)+".tif",img.astype(np.uint16))
    
def _expand_dim_tif(img_path:str, axes: str)-> np.ndarray:
    """Adjust the dimension of the image to TZCYX"""
    # Open tif file
    img = imread(img_path)
    ref_axes = 'TZCYX'
    
    if len(axes)<len(ref_axes):
        missing_axes = [ref_axes.index(x) for x in ref_axes if x not in axes]
        # Add missing axes
        for ax in missing_axes:
            img = np.expand_dims(img,axis=ax)
    return img

def _write_tif(img_data: list)-> None:
    # Unpack img_data
    meta,img_name,img = img_data
    _,frame,z_slice = [int(i[1:])-1 for i in img_name.split('_')[1:]]             
    chan = meta['full_channel_list'].index(img_name.split('_')[0])
    
    im_folder = join(sep,meta['exp_path_list'][0]+sep,'Images')
    imwrite(join(sep,im_folder+sep,img_name)+".tif",img[frame,z_slice,chan,...].astype(np.uint16))
    
def write_img(meta_dict: dict)-> None:
    # Create all the names for the images+metadata
    img_name_list = _name_img_list(meta_dict)
    
    if meta_dict['file_type'] == '.nd2':
        # Add metadata and img_obj to img_name_list
        img_name_list = [(meta_dict,x) for x in img_name_list]
        with ProcessPoolExecutor() as executor: # nd2 file are messed up with multithreading
            executor.map(_write_ND2,img_name_list)
    elif meta_dict['file_type'] == '.tif':
        # Add metadata and img to img_name_list
        img_arr = _expand_dim_tif(meta_dict['img_path'],meta_dict['axes'])
        img_name_list = [(meta_dict,x,img_arr) for x in img_name_list]
        with ThreadPoolExecutor() as executor:
            executor.map(_write_tif,img_name_list)

def _init_exp_settings(exp_path: str, meta_dict: dict)-> dict:
    """Initialize Settings object from json file or metadata"""
    
    if exists(join(sep,exp_path+sep,'exp_settings.json')):
        exp_set = init_from_json(join(sep,exp_path+sep,'exp_settings.json'))
    else:
        meta_dict['exp_path'] = exp_path
        exp_set = init_from_dict(meta_dict)
    return exp_set

def img_seq_exp(img_path: str, active_channel_list: list[str], full_channel_list: list[str]=None, img_seq_overwrite: bool=False)-> list[Experiment]:
    """Create an image seq for individual image files (.nd2 or .tif), based on the number of field of view and return a list of Settings objects"""
    # Get metadata
    meta_dict = get_metadata(img_path,active_channel_list,full_channel_list)
    
    # If img are already processed
    exp_set_list = []
    for serie in range(meta_dict['n_series']):
        exp_path = meta_dict['exp_path_list'][serie]
        meta_dict['exp_path'] = exp_path
        
        img_folder = join(sep,exp_path+sep,'Images')
        if not exists(img_folder):
            mkdir(img_folder)
        
        if exists(join(sep,exp_path+sep,'REMOVED_EXP.txt')):
            print(f"-> Exp.: {exp_path} has been removed\n")
            continue
        
        if any(scandir(img_folder)) and not img_seq_overwrite:
            print(f"-> Exp.: {exp_path} has already been processed\n")
            exp_set_list.append(_init_exp_settings(exp_path,meta_dict))
            continue
        
        # If images are not processed
        print(f"-> Exp.: {exp_path} is being processed\n")
        write_img(meta_dict)
        
        exp_set = init_from_dict(meta_dict)
        exp_set.save_as_json()
        exp_set_list.append(exp_set)
    return exp_set_list
    
# # # # # # # main function # # # # # # #
def img_seq_all(img_path_list: list[str], active_channel_list: list, 
                          full_channel_list: list=None, img_seq_overwrite: bool=False)-> list[Experiment]:
    """Process all the images files (.nd2 or .tif) found in parent_folder and return a list of Settings objects"""
    exp_set_list = []
    for img_path in img_path_list:
        exp_set_list.extend(img_seq_exp(img_path,active_channel_list,full_channel_list,img_seq_overwrite))
    return exp_set_list
