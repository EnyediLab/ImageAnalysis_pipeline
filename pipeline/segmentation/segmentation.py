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
from ImageAnalysis_pipeline.pipeline.classes import Experiment,_img_list_src


def _apply_threshold(img_data: list)-> float:
    img_path,manual_threshold = img_data
    img = imread(img_path)
    savedir = img_path.replace("Images","Masks_SimpleThreshold").replace('_Registered','').replace('_Blured','')
    # Set the threshold's value. Either as input or automatically if thres==None
    if manual_threshold:
        used_threshold = manual_threshold
    else:
        used_threshold,_ = cv2.threshold(img.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Apply the threshold
    _,mask = cv2.threshold(img.astype(np.uint16),used_threshold,255,cv2.THRESH_BINARY)
    
    # Clean the mask of small holes and small objects
    mask = remove_small_holes(mask.astype(bool),50)
    mask = remove_small_objects(mask,1000).astype(bool)
    imsave(savedir,mask.astype(np.uint16))
    return manual_threshold

def simple_threshold(exp_set_list: list[Experiment], simple_thresold_overwrite: bool=False, manual_threshold: int=None, img_fold_src: str=None)-> list[Experiment]:
    for exp_set in exp_set_list:
        # Check if exist
        if exp_set.process.simple_threshold and not simple_thresold_overwrite:
                # Log
            print(f"--> Simple threshold images already exist with {exp_set.process.simple_threshold}")
            continue
        # Generate list of image source
        img_list_src = _img_list_src(exp_set, img_fold_src)
        img_data = [(img_path,manual_threshold) for img_path in img_list_src]
        
        # log
        if manual_threshold:
            print(f"--> Thresholding images with a MANUAL threshold of {manual_threshold}")
        else:
            print(f"--> Thresholding images with a AUTOMATIC threshold")

        # Create blur dir
        if not isdir(join(sep,exp_set.exp_path+sep,'Masks_SimpleThreshold')):
            mkdir(join(sep,exp_set.exp_path+sep,'Masks_SimpleThreshold'))
        
        threshold_value_list = []
        with ThreadPoolExecutor() as executor:
            results = executor.map(_apply_threshold,img_data)
            for result in results:
                threshold_value_list.append(result)
        # Log
        threshold_value = round(np.mean(threshold_value_list),ndigits=2)
        print(f"\t---> Threshold value used: {threshold_value}")
        
        # Save settings
        exp_set.process.simple_threshold = [f"Manual threshold={manual_threshold}"]
        exp_set.save_as_json()    
    return exp_set_list  

