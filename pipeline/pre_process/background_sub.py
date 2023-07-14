from concurrent.futures import ProcessPoolExecutor
from tifffile import imread, imwrite
from smo import SMO
import numpy as np

def _apply_bg_sub(processed_image: list)-> None:
    # Initiate SMO
    proc_img_path,smo = processed_image
    img = imread(proc_img_path)
    bg_img = smo.bg_corrected(img)
    # Reset neg val to 0
    bg_img[bg_img<0] = 0
    imwrite(proc_img_path,bg_img.astype(np.uint16))

# # # # # # # # main function # # # # # # # #
def background_sub(exp_set_list: list, sigma: float=0.0, size: int=7, bg_sub_overwrite: bool=False)-> list:
    """For each experiment, apply a background substraction on the images and return a list of Settings objects"""
    for exp_set in exp_set_list:
        if exp_set.process.background_sub and not bg_sub_overwrite:
            print(f"--> Background substraction was already apply on {exp_set.exp_path}")
            continue
        print(f"--> Applying background substraction on {exp_set.exp_path}, with sigma={sigma} and size={size}")
        
        # Add smo_object to img_path
        processed_image_list = exp_set.processed_image_list.copy()
        smo = SMO(shape=(exp_set.img_data.img_width,exp_set.img_data.img_length),sigma=sigma,size=size)
        processed_image_list = [(img_path,smo) for img_path in processed_image_list]
        
        with ProcessPoolExecutor() as executor:
            executor.map(_apply_bg_sub,processed_image_list)
            
        exp_set.process.background_sub = (f"sigma={sigma}",f"size={size}")
        exp_set.save_as_json()
    return exp_set_list

