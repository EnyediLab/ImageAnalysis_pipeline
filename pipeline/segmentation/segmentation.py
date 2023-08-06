from __future__ import annotations
from os import getcwd,sep,mkdir
import sys
parent_dir = getcwd()
sys.path.append(parent_dir)

from os.path import join, isdir
import cv2
from skimage.morphology import remove_small_objects,remove_small_holes
import numpy as np
from tifffile import imread,imsave
from concurrent.futures import ThreadPoolExecutor
from ImageAnalysis_pipeline.pipeline.classes import Experiment
from ImageAnalysis_pipeline.pipeline.loading_data import img_list_src, is_processed

def _determine_threshold(img: np.ndarray, manual_threshold: float=None)-> float:
    # Set the threshold's value. Either as input or automatically if thres==None
    threshold_value = manual_threshold
    if not manual_threshold:
        threshold_value,_ = cv2.threshold(img.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshold_value

def _clean_mask(mask: np.ndarray)-> np.ndarray:
    mask = remove_small_holes(mask.astype(bool),50)
    return remove_small_objects(mask,1000).astype(np.uint16)

def _apply_threshold(img_data: list)-> float:
    img_path,manual_threshold = img_data
    img = imread(img_path)
    savedir = img_path.replace("Images","Masks_Threshold").replace('_Registered','').replace('_Blured','')
    
    threshold_value = _determine_threshold(img,manual_threshold)
    
    # Apply the threshold
    _,mask = cv2.threshold(img.astype(np.uint16),threshold_value,255,cv2.THRESH_BINARY)
    
    # Clean and save
    imsave(savedir,_clean_mask(mask))
    return threshold_value

# # # # # # # # main functions # # # # # # # # # 
def simple_threshold(exp_set_list: list[Experiment], channel_seg: str, simple_thresold_overwrite: bool=False, manual_threshold: int=None, img_fold_src: str=None)-> list[Experiment]:
    for exp_set in exp_set_list:
        # Check if exist
        if is_processed(exp_set.masks.simple_threshold,channel_seg,simple_thresold_overwrite):
                # Log
            print(f" --> Object has already been segmented with {exp_set.process.simple_threshold}")
            continue
        
        # If not, Generate list of image source
        img_path_list = img_list_src(exp_set, img_fold_src)
        img_data = [(img_path,manual_threshold) for img_path in img_path_list if channel_seg in img_path]
        
        # Create blur dir and apply blur
        if not isdir(join(sep,exp_set.exp_path+sep,'Masks_Threshold')):
            mkdir(join(sep,exp_set.exp_path+sep,'Masks_Threshold'))
        
        print(f" --> Segmenting object...")
        # Determine threshold value
        threshold_value_list = []
        with ThreadPoolExecutor() as executor:
            results = executor.map(_apply_threshold,img_data)
            for result in results:
                threshold_value_list.append(result)

        # log
        log_value = "MANUAL"
        threshold_value = manual_threshold
        if not manual_threshold:
            log_value = "AUTOMATIC"
            threshold_value = round(np.mean(threshold_value_list),ndigits=2)
        print(f"\t---> Threshold created with: {log_value} threshold of {threshold_value}")

        # Save settings
        if exp_set.masks.threshold_seg:
            exp_set.masks.threshold_seg.update({channel_seg:{'method':log_value,'threshold':threshold_value}})
        else:
            exp_set.masks.threshold_seg = {channel_seg:{'method':log_value,'threshold':threshold_value}}
        exp_set.save_as_json()    
    return exp_set_list  

