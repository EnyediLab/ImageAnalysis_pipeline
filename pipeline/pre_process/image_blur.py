from __future__ import annotations
from os import sep,mkdir,getcwd
from os.path import join, isdir
import sys
parent_dir = getcwd() 
# Add the parent to sys.pah
sys.path.append(parent_dir)

from ImageAnalysis_pipeline.pipeline.classes import Experiment,_img_list_src
from typing import Iterable
from tifffile import imwrite, imread
from cv2 import GaussianBlur
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def _apply_blur(img_data: Iterable)-> None:
    
    img_path,blur_kernel,blur_sigma = img_data
    img = imread(img_path)
    savedir = img_path.replace("Images","Images_Blured").replace('_Registered','')
    # Blur image and save
    imwrite(savedir,GaussianBlur(img,blur_kernel,blur_sigma).astype(np.uint16))

# # # # # # # # main functions # # # # # # # # # 
def blur_img(exp_set_list: list[Experiment], blur_kernel: list[int], blur_sigma: int, img_fold_src: str=None, blur_overwrite: bool = False)-> None:
    # Check if kernel contains 2 odd intengers >= to 3
    if not all(i%2!=0 for i in blur_kernel) and not all(i>=3 for i in blur_kernel):
        print("The input 'blur_kernel' must contain 2 odd intengers greater or equal to 3")

    # Get the exp_path and load exp_para
    for exp_set in exp_set_list:
        # Check if exist
        if exp_set.process.img_blured and not blur_overwrite:
                # Log
            print(f"--> Images are already blured with {exp_set.process.img_blur}")
            continue
        
        # Log
        print(f"--> Bluring images using a kernel of {blur_kernel} and sigma of {blur_sigma}")
        img_list_src = _img_list_src(exp_set, img_fold_src)
        img_data = [(img_path,blur_kernel,blur_sigma) for img_path in img_list_src]
        # Create blur dir
        if not isdir(join(sep,exp_set.exp_path+sep,'Images_Blured')):
            mkdir(join(sep,exp_set.exp_path+sep,'Images_Blured'))
        
        with ThreadPoolExecutor() as executor:
            executor.map(_apply_blur,img_data)
            
        # Save settings
        exp_set.process.img_blur = [f"blur_kernel={blur_kernel}",f"blur_sigma={blur_sigma}"]
        exp_set.save_as_json()
    return exp_set_list