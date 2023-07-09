from dataclasses import dataclass
import json
from os import mkdir,sep,scandir,walk
import re
from time import time
from typing import Protocol
from nd2reader import ND2Reader
from tifffile import imwrite,imread
from os.path import join,isdir,exists
import numpy as np
from metadata import get_metadata
from settings import Settings
from smo import SMO

def _name_img_list(meta: dict)-> list:
    # Create a name for each image
    img_name = []
    for t in range(meta['n_frames']):
        for z in range(meta['n_slices']):
            for chan in meta['active_channel_list']:
                img_name.append(chan+'_f%04d'%(t+1)+'_z%04d'%(z+1))
    return img_name

def _write_ND2(meta: dict, img_obj: ND2Reader, img_name: list, serie: int)-> None:
    # Unpack img_name
    t,z = [int(i[1:])-1 for i in img_name.split('_')[1:]]             
    c = meta['full_channel_list'].index(img_name.split('_')[0])
    
    # Get the image       
    if meta['n_slices']>1: 
        img = img_obj.get_frame_2D(c=c,t=t,z=z,x=meta['img_width'],y=meta['img_length'],v=serie)
    else: img = img_obj.get_frame_2D(c=c,t=t,x=meta['img_width'],y=meta['img_length'],v=serie)
    # Save
    im_folder = join(sep,meta['exp_path_list'][serie]+sep,'Images')
    imwrite(join(sep,im_folder+sep,img_name)+".tif",img.astype(np.uint16))
    
def _expand_dim_tif(img_path:str, axes: str)-> np.ndarray:
    # Open tif file
    img = imread(img_path)
    ref_axes = 'TZCYX'
    
    if len(axes)<len(ref_axes):
        missing_axes = [ref_axes.index(x) for x in ref_axes if x not in axes]
        # Add missing axes
        for ax in missing_axes:
            img = np.expand_dims(img,axis=ax)
    return img

def _write_tif(meta: dict,img_name: list,img: np.array)-> None:
    # Unpack img_name
    t,z = [int(i[1:])-1 for i in img_name.split('_')[1:]]             
    c = meta['full_channel_list'].index(img_name.split('_')[0])
    
    im_folder = join(sep,meta['exp_path_list'][0]+sep,'Images')
    imwrite(join(sep,im_folder+sep,img_name)+".tif",img[t,z,c,...].astype(np.uint16))
    
def write_img(meta: dict,img_name_list: list, serie: int)-> None:
    if meta['file_type'] == '.tif':
        img = _expand_dim_tif(meta['img_path'],meta['axes'])
        meta['exp_path'] = meta['exp_path_list'][0]
    elif meta['file_type'] == '.nd2':
        img_obj = ND2Reader(meta['img_path'])
        meta['exp_path'] = meta['exp_path_list'][serie]
    
    for img_name in img_name_list:
        if meta['file_type'] == '.nd2':  
            _write_ND2(meta,img_obj,img_name,serie)
        elif meta['file_type'] == '.tif':
            _write_tif(meta,img_name,img)
    return meta

# def smo_bg_sub(imgFold_path,sigma=0.0,size=7):
#         # Log
#         print(f"--> Applying 'Auto' background substraction on {imgFold_path} with: sigma={sigma} and size={size}")
        
#         # Get the exp_path and load exp_para
#         exp_path = sep.join(imgFold_path.split(sep)[:-1])
#         exp_prop = Utility.open_exp_prop(exp_path=exp_path)
#         exp_para = exp_prop['metadata']

#         # Initiate SMO
#         smo = SMO(shape=(exp_para['x'],exp_para['y']),sigma=sigma,size=size)

#         # Run one image at a time
#         for file in sorted(listdir(imgFold_path)):
#             if file.endswith('.tif'):
#                 # Load image
#                 img = imread(join(sep,imgFold_path+sep,file))
#                 # Get mean of the bg as rounded to closest intenger
#                 bg_img = smo.bg_corrected(img)
#                 # Reset neg val to 0
#                 bg_img[bg_img<0] = 0
#                 # Resave stack
#                 imwrite(join(sep,imgFold_path+sep,file),bg_img.astype(np.uint16))
        
#         # Save pickle
#         exp_prop['fct_inputs']['smo_bg_sub'] = {'imgFold_path':imgFold_path,'smo_sigma':sigma,'smo_size':size}
#         exp_prop['img_preProcess']['bg_sub'] = 'Auto'
#         Utility.save_exp_prop(exp_path=exp_path,exp_prop=exp_prop)
#         return exp_prop

def _check_exp_state(exp_path_list: list)-> list:
    # Check if all images exist
    processed_exp_list = []
    for exp_path in exp_path_list:
        if exists(join(sep,exp_path+sep,'REMOVED_EXP.txt')):
            print(f"-> Exp.: {exp_path} has been removed\n")
            continue
        
        if any(scandir(join(sep,exp_path+sep,'Images'))):
            processed_exp_list.append(exp_path)
    return processed_exp_list

def _load_settings(exp_path: str, meta: dict)-> dict:
    if exists(join(sep,exp_path+sep,'exp+settings.json')):
        settings = Settings.from_json(Settings,join(sep,exp_path+sep,'exp_settings.json'))
    else:
        meta['exp_path'] = exp_path
        settings = Settings.from_metadata(Settings,meta)
    return settings

def create_img_seq(img_path: str, active_channel_list: list, full_channel_list: list=None, img_seq_overwrite: bool=False)-> list:
    # Get metadata
    meta = get_metadata(img_path,active_channel_list,full_channel_list)
    
    # If img are already processed
    settings_list = []
    processed_exp_list = _check_exp_state(meta['exp_path_list'])
    if processed_exp_list and not img_seq_overwrite:
        for exp_path in processed_exp_list:
            print(f"-> Exp.: {exp_path} has already been processed\n")
            settings_list.append(_load_settings(exp_path,meta))
        return settings_list 
    
    # Create a name for each image
    img_name_list = _name_img_list(meta)
    
    for serie in range(meta['n_series']):
        # Write ND2
        meta = write_img(meta,img_name_list,serie)
        settings_list.append(Settings.from_metadata(Settings,meta))
    return settings_list
    
def gather_all_images(parent_folder: str, file_type: str=None)-> list:
    # look through the folder and collect all image files
    if not isdir(parent_folder):
        raise ValueError(f"{parent_folder} is not a correct path. Try a full path")
    
    if file_type: extension = (file_type,)
    else: extension = ('.nd2','.tif','.tiff')
    
    # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
    imgS_path = []
    for root , _, files in walk(parent_folder):
        for f in files:
            # Look for all files with selected extension and that are not already processed 
            if not re.search(r'f\d\d\d_z\d\d\d',f) and f.endswith(extension):
                imgS_path.append(join(sep,root+sep,f))
    return imgS_path.sort()

def process_imgs(imgS_path: list, full_channel_list: list=None, img_seq_overwrite: bool=False)-> list:
    settings_list = []
    for img_path in imgS_path:
        settings_list.extend(create_img_seq(img_path,active_channel_list,full_channel_list,img_seq_overwrite))
    return settings_list

def background_sub(settings_list: list, sigma: float=0.0, size: int=7)->None:
    for settings in settings_list:
        if settings.background_sub:
            print(f"--> Background substraction was already apply on {settings.img_path}")
            continue
        
        _apply_bg_sub(settings,sigma,size)

def _apply_bg_sub(settings: Settings,sigma: float=0.0, size: int=7)->None:
    # Initiate SMO
    smo = SMO(shape=(settings.img_width,settings.img_length),sigma=sigma,size=size)
    print(f"--> Applying background substraction on {settings.img_path}, with sigma={sigma} and size={size}")
    for proc_img_path in settings.processed_image_list:
        img = imread(proc_img_path)
        bg_img = smo.bg_corrected(img)
        # Reset neg val to 0
        bg_img[bg_img<0] = 0
        imwrite(proc_img_path,bg_img.astype(np.uint16))


if __name__ == "__main__":
    

    # Test
    # img_path = '/Users/benhome/BioTool/GitHub/cp_dev/c4z1t91v1.tif'
    active_channel_list = ['BFP','GFP','RFP','iRed']
    # t1 = time()
    # settings_list = create_img_seq(img_path,active_channel_list)
    # t2 = time()
    # print(f"Time to get meta: {t2-t1}")
    # print(settings_list)

    img_path2 = '/Users/benhome/BioTool/GitHub/cp_dev/c4z1t91v1.nd2'
    t1 = time()
    settings_list = create_img_seq(img_path2,active_channel_list=active_channel_list)
    t2 = time()
    print(f"Time to get meta: {t2-t1}")
    print(settings_list)