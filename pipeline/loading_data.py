from __future__ import annotations
from os.path import join, exists
from os import sep, mkdir,getcwd
import sys
parent_dir = getcwd()
sys.path.append(parent_dir)

from ImageAnalysis_pipeline.pipeline.classes import Experiment
from typing import Iterable
import numpy as np
from tifffile import imread

def load_stack(img_list: list[str], channel_list: Iterable[str], frame_range: Iterable[int])-> np.ndarray:
    # Load/Reload stack. Expected shape of images tzxyc
    exp_list = []
    for chan in channel_list:
        chan_list = []
        for frame in frame_range:
            f_lst = []
            for img in img_list:
                # To be able to load either _f3digit.tif or _f4digit.tif
                ndigit = len(img.split(sep)[-1].split('_')[1][1:])
                if img.__contains__(f'{chan}_f%0{ndigit}d'%(frame+1)):
                    f_lst.append(imread(img))
            chan_list.append(f_lst)
        exp_list.append(chan_list)
    if len(channel_list)==1:
        stack = np.squeeze(np.stack(exp_list))
    else:
        stack = np.moveaxis(np.squeeze(np.stack(exp_list)), [0], [-1])
    return stack

def _img_list_src(exp_set: Experiment, img_fold_src: str)-> list[str]:
    """If not manually specified, return the latest processed images list"""
    
    if img_fold_src and img_fold_src == 'Images':
        return exp_set.processed_images_list
    
    if img_fold_src and img_fold_src == 'Images_Registered':
        return exp_set.register_images_list
    
    if img_fold_src and img_fold_src == 'Images_Blured':
        return exp_set.blur_images_list
    
    # If not manually specified, return the latest processed images list
    if exp_set.process.img_blured:
        return exp_set.blur_images_list
    elif exp_set.process.img_registered:
        return exp_set.register_images_list
    else:
        return exp_set.processed_images_list