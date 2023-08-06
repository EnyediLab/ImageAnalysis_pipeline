from __future__ import annotations
from os import getcwd, sep, mkdir, listdir
import sys

parent_dir = getcwd()
sys.path.append(parent_dir)

from dataclasses import fields
from os import sep
from os.path import join
import pandas as pd
import numpy as np
from ImageAnalysis_pipeline.pipeline.classes import Experiment, Masks
from ImageAnalysis_pipeline.pipeline.loading_data import load_stack, img_list_src, is_processed, mask_list_src
from tifffile import imread,imsave
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor


def masks_process_dict(masks: Masks)-> dict:
    masks_dict ={}
    for field in fields(masks):
        name = field.name
        value = list(getattr(masks,name).keys())
        if value:
            masks_dict[name] = value
    return masks_dict

def trim_masks_list(masks_dict: dict)-> dict:
    if 'iou_tracking' in masks_dict:
        del masks_dict['cellpose_seg']
    return masks_dict

def get_masks_path(exp_set: Experiment)-> dict:
    masks_dict = masks_process_dict(exp_set.masks)
    masks_dict = trim_masks_list(masks_dict)
    
    new_masks_dict = {}
    for mask_name,mask_channels in masks_dict.items():
        if mask_name == 'iou_tracking':
            mask_files = exp_set.mask_iou_track_list
        elif mask_name == 'cellpose_seg':
            mask_files = exp_set.mask_cellpose_list
        
        for chan in mask_channels:
            chan_mask_list = [file for file in mask_files if chan in file]
            new_masks_dict[mask_name] = {chan:[[file for file in chan_mask_list if f"_f{(i+1):04d}" in file] for i in range(exp_set.img_properties.n_frames)]}
    return new_masks_dict

def gen_input_data(exp_set: Experiment, img_folder_src: str)-> list[dict]:
    masks_dict = get_masks_path(exp_set)
    img_path_list = img_list_src(exp_set,img_folder_src)
    img_path_input = [[file for file in img_path_list if file.__contains__(f"_f{(i+1):04d}")] for i in range(exp_set.img_properties.n_frames)]
    
    new_masks_dict = {}
    for mask_name, mask_channelsNfiles_dict in masks_dict.items():
        for chan, mask_files_list in mask_channelsNfiles_dict.items():
            new_masks_dict[mask_name] = {chan:list(zip(mask_files_list,img_path_input))}
    return new_masks_dict
    
def get_mask_keys(mask_name: str, exp_set: Experiment)-> list:
    default_keys = ['cell','frames','time','mask_name','mask_chan','exp']  
    
    if mask_name == 'iou_tracking':
        specific_keys = exp_set.active_channel_list
        return default_keys + specific_keys

def change_df_dtype(df: pd.DataFrame, exp_set: Experiment)-> pd.DataFrame:
    dtype_default = {'cell':'string','frames':'int','time':'float','mask_name':'category','mask_chan':'category',
                  'exp':'category','level_0_tag':'category','level_1_tag':'category'}
    dtype_channels = {chan:'float' for chan in exp_set.active_channel_list}
    
    dtype_final = {**dtype_default,**dtype_channels}
    df = df.astype(dtype_final)
    return df

def extract_mask_data(mask_name: str, mask_channelsNfiles_dict: dict, exp_set: Experiment)-> pd.DataFrame:
    
    mask_keys = get_mask_keys(mask_name,exp_set)
    df = pd.DataFrame(columns=mask_keys)
    for mask_chan,input_list in mask_channelsNfiles_dict.items():
        for input_imgs in input_list:
            mask_path, img_path = input_imgs
            frame = int(mask_path[0].split(sep)[-1].split('_')[2][1:])-1
            mask = load_stack(mask_path,channel_list=[mask_chan],frame_range=[frame])
            if mask.ndim==3:
                mask = np.amax(mask,axis=0).astype('uint16')
            temp_dict = {k:[] for k in mask_keys}
            for obj in list(np.unique(mask))[1:]:
                temp_dict['cell'].append(f"{exp_set.exp_path.split(sep)[-1]}_{mask_name}_{mask_chan}_cell{obj}")
                temp_dict['frames'].append(frame+1)
                temp_dict['time'].append(exp_set.time_seq[frame])
                temp_dict['mask_name'].append(mask_name)
                temp_dict['mask_chan'].append(mask_chan)
                temp_dict['exp'].append(exp_set.exp_path.split(sep)[-1])
                for chan in exp_set.active_channel_list:
                    img = load_stack(img_path,channel_list=[chan],frame_range=[frame])
                    temp_dict[chan].append(np.nanmean(a=img,where=mask==obj))
            df = pd.concat([df,pd.DataFrame.from_dict(temp_dict)],ignore_index=True) 
    return df  
    
def extract_mask_data_para(mask_name: str, mask_channelsNfiles_dict: dict, exp_set: Experiment)-> pd.DataFrame:
    
    mask_keys = get_mask_keys(mask_name,exp_set)
    df = pd.DataFrame(columns=mask_keys)
    for mask_chan,input_list in mask_channelsNfiles_dict.items():
        input_list = [[mask_path,img_path,mask_chan,mask_keys,exp_set,mask_name] for mask_path,img_path in input_list]
        
        with ProcessPoolExecutor() as executor:
            temp_dicts = executor.map(_extract_mask_data_para,input_list)
            for temp_dict in temp_dicts:
                df = pd.concat([df,pd.DataFrame.from_dict(temp_dict)],ignore_index=True) 
    return df  

def _extract_mask_data_para(input_imgs: list)-> pd.DataFrame:
    mask_path, img_path, mask_chan, mask_keys, exp_set, mask_name = input_imgs
    frame = int(mask_path[0].split(sep)[-1].split('_')[2][1:])-1
    mask = load_stack(mask_path,channel_list=[mask_chan],frame_range=[frame])
    if mask.ndim==3:
        mask = np.amax(mask,axis=0).astype('uint16')
    temp_dict = {k:[] for k in mask_keys}
    for obj in list(np.unique(mask))[1:]:
        temp_dict['cell'].append(f"{exp_set.exp_path.split(sep)[-1]}_{mask_name}_{mask_chan}_cell{obj:03d}")
        temp_dict['frames'].append(frame+1)
        temp_dict['time'].append(exp_set.time_seq[frame])
        temp_dict['mask_name'].append(mask_name)
        temp_dict['mask_chan'].append(mask_chan)
        temp_dict['exp'].append(exp_set.exp_path.split(sep)[-1])
        for chan in exp_set.active_channel_list:
            img = load_stack(img_path,channel_list=[chan],frame_range=[frame])
            temp_dict[chan].append(np.nanmean(a=img,where=mask==obj))
    return temp_dict

# # # # # # # # main functions # # # # # # # # # 
def extract_channel_data(exp_set_list: list[Experiment], img_folder_src: str, data_overwrite: bool=False)-> list[Experiment]:
    for exp_set in exp_set_list:
        # Load df
        df_analysis = exp_set.load_df_analysis(data_overwrite)
        if not df_analysis.empty:
            print(f" --> Cell data have already been extracted")
            df_analysis = change_df_dtype(df_analysis,exp_set)
            continue
        
        print(f" --> Extracting cell data")   
        # Pre-load masks and images path
        masks_dict = gen_input_data(exp_set,img_folder_src)
        
        for mask_name,mask_channelsNfiles_dict in masks_dict.items():
            # Don't use parallel processing
            # df = extract_mask_data(mask_name,mask_channelsNfiles_dict,exp_set)
            # Use parallel processing
            df = extract_mask_data_para(mask_name,mask_channelsNfiles_dict,exp_set)
            
            # Add tags
            df['level_0_tag'] = exp_set.analysis.level_0_tag
            df['level_1_tag'] = exp_set.analysis.level_1_tag
            
            # Concat all df
            df_analysis = pd.concat([df_analysis,df],ignore_index=True)        
            df_analysis = change_df_dtype(df_analysis,exp_set)
            df_analysis = df_analysis.sort_values(by=['frames','cell'])
        
        # Save df
        exp_set.save_df_analysis(df_analysis)
        exp_set.save_as_json()
    return exp_set_list

