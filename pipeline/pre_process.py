from dataclasses import dataclass
from os import mkdir,sep
from time import time
from typing import Protocol
from nd2reader import ND2Reader
from tifffile import imwrite,imread
from os.path import join,isdir
import numpy as np
from metadata import get_metadata
from joblib import Parallel,delayed

def _create_exp_folder(meta: dict) -> dict:
    meta['exp_path_list'] = []
    for serie in range(meta['n_series']):
        # Create subfolder in parent folder to save the image sequence with a serie's tag
        path_split = meta['img_path'].split(sep)
        path_split[-1] = path_split[-1].split('.')[0]+f"_s{serie+1}"
        exp_path =  sep.join(path_split)
        if not isdir(exp_path):
            mkdir(exp_path)
        meta['exp_path_list'].append(exp_path)
        # Get tags
        meta['level_1_tag'] = path_split[-3]
        meta['level_0_tag'] = path_split[-2]
    return meta

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
        img = img_obj.get_frame_2D(c=c,t=t,z=z,x=meta['ImageWidth'],y=meta['ImageLength'],v=serie)
    else: img = img_obj.get_frame_2D(c=c,t=t,x=meta['ImageWidth'],y=meta['ImageLength'],v=serie)
    # Save
    im_folder = meta['exp_path_list'][serie]
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
    
    imwrite(join(sep,meta['exp_path_list'][0]+sep,img_name)+".tif",img[t,z,c,...].astype(np.uint16))
    
def write_img(meta: dict,img_name_list: list, serie: int)-> None:
    if meta['file_type'] == '.tif':
        img = _expand_dim_tif(meta['img_path'],meta['axes'])
        Parallel(n_jobs=-1)(delayed(_write_tif)(meta,img_name,img) for img_name in img_name_list)
    elif meta['file_type'] == '.nd2':
        img_obj = ND2Reader(meta['img_path'])
        Parallel(n_jobs=-1)(delayed(_write_ND2)(meta,img_obj,img_name,serie) for img_name in img_name_list)
    
    # for img_name in img_name_list:
    #     if meta['file_type'] == '.nd2':  
    #         _write_ND2(meta,img_obj,img_name,serie)
    #     elif meta['file_type'] == '.tif':
    #         _write_tif(meta,img_name,img)

def main(img_path:str,active_channel_list:list):
    # Get metadata
    meta = get_metadata(img_path,active_channel_list)
    # Create experiment folder
    meta = _create_exp_folder(meta)
    # Create a name for each image
    img_name_list = _name_img_list(meta)
    
    for serie in range(meta['n_series']):
        # Write ND2
        write_img(meta,img_name_list,serie)
    








if __name__ == "__main__":
    

    # Test
    img_path = '/Users/benhome/BioTool/GitHub/cp_dev/c4z1t91v1.tif'
    active_channel_list = ['BFP','GFP','RFP','iRed']
    t1 = time()
    img_meta = main(img_path,active_channel_list)
    t2 = time()
    print(f"Time to get meta: {t2-t1}")

    img_path2 = '/Users/benhome/BioTool/GitHub/cp_dev/c4z1t91v1.nd2'
    t1 = time()
    img_meta2 = main(img_path2,active_channel_list=active_channel_list)
    t2 = time()
    print(f"Time to get meta: {t2-t1}")