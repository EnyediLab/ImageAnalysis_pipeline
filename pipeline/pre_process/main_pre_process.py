from __future__ import annotations
from dataclasses import dataclass, field
from os.path import join
from os import sep, getcwd
import sys
parent_dir = getcwd()
sys.path.append(parent_dir)

from ImageAnalysis_pipeline.pipeline.Experiment_Classes import Experiment
from os import sep, walk
import re
from ImageAnalysis_pipeline.pipeline.pre_process.image_sequence import img_seq_all
from ImageAnalysis_pipeline.pipeline.pre_process.image_blur import blur_img
from ImageAnalysis_pipeline.pipeline.pre_process.background_sub import background_sub
from ImageAnalysis_pipeline.pipeline.pre_process.image_registration import register_img, channel_shift_register

EXTENTION = ('.nd2','.tif','.tiff')

def get_img_path(folder: str)-> list[str]:
    # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
    imgS_path = []
    for root , _, files in walk(folder):
        for f in files:
            # Look for all files with selected extension and that are not already processed 
            if not re.search(r'_f\d\d\d',f) and f.endswith(EXTENTION):
                imgS_path.append(join(sep,root+sep,f))
    return sorted(imgS_path)


def gather_all_images(input_folder: str | list[str])-> list[str]:
    # look through the folder and collect all image files
    print(f"\nSearching for {EXTENTION} files in {input_folder}")
    # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
    if isinstance(input_folder,str):
        return get_img_path(input_folder)
    
    if isinstance(input_folder,list):
        img_path_list = []
        for folder in input_folder:
            img_path_list.extend(get_img_path(folder))
        return img_path_list

# # # # # # # main function # # # # # # # 
def pre_process_all(parent_folder: str, active_channel_list: list[str], full_channel_list: list[str]=None, file_type: str=None, 
                    img_seq_overwrite: bool=False, 
                    bg_sub: bool=True, sigma: float=0.0, size: int=7, bg_sub_overwrite: bool=False,
                    chan_shift: bool=False, reg_channel: str=None, reg_mtd: str='rigid_body', chan_shift_overwrite: bool=False,
                    register_images: bool=True, reg_ref: str='previous', reg_overwrite: bool=False,
                    blur: bool=False, blur_kernel: tuple(int)=(15,15), blur_sigma: int=5, img_fold_src: str=None, blur_overwrite: bool = False,
                    )-> list[Experiment]:
    
    img_path_list = gather_all_images(parent_folder)
    
    exp_set_list = img_seq_all(img_path_list,active_channel_list,full_channel_list,img_seq_overwrite)
    
    if bg_sub:
        if img_seq_overwrite==True: bg_sub_overwrite=True
        exp_set_list = background_sub(exp_set_list,sigma,size,bg_sub_overwrite)
    
    if chan_shift:
        if bg_sub_overwrite==True: chan_shift_overwrite=True
        exp_set_list = channel_shift_register(exp_set_list,reg_mtd,reg_channel,chan_shift_overwrite)
    
    if register_images:
        if bg_sub_overwrite==True or chan_shift_overwrite==True: reg_overwrite=True
        exp_set_list = register_img(exp_set_list,reg_channel,reg_mtd,reg_ref,reg_overwrite)
    
    if blur:
        if bg_sub_overwrite==True or reg_overwrite==True or chan_shift_overwrite==True: blur_overwrite=True
        exp_set_list = blur_img(exp_set_list,blur_kernel,blur_sigma,img_fold_src,blur_overwrite)
    
    return exp_set_list
    

@dataclass
class Settings:
    settings: dict
    bg_sub: dict = field(default_factory=dict)
    chan_shift: dict = field(default_factory=dict)
    register: dict = field(default_factory=dict)
    blur: dict = field(default_factory=dict)
    
    def __post_init__(self)-> None:
        if self.settings['run_bg_sub']:
            self.bg_sub = self.settings['bg_sub']


@dataclass
class PreProcess:
    input_folder: str | list[str]
    settings: dict
    img_path_list: list[str] = field(default_factory=list)
    experiment_list: list[Experiment] = field(default_factory=list)

    def __post_init__(self)-> None:
        self.img_path_list = gather_all_images(self.input_folder)
    
    def process_from_settings(self,settings: dict)-> list[Experiment]:
        pass
    
    def bg_sub(self)-> None:
        self.experiment_list = background_sub(self.experiment_list)







if __name__ == "__main__":
    from time import time
    
    # Test
    active_channel_list = ['GFP','RFP']

    parent_folder = '/media/ben/Analysis/Python/Test_images/Run2'
    
    t1 = time()
    exp_set_list = pre_process_all(
                        parent_folder=parent_folder,active_channel_list=active_channel_list,
                        file_type='.nd2',img_seq_overwrite=False,
                        bg_sub=True,bg_sub_overwrite=False,
                        chan_shift=False,reg_channel='RFP',chan_shift_overwrite=False,
                        register_images=False,
                        blur=True,blur_overwrite=True,
                        )
    t2 = time()
    if t2-t1<60: print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")
    else: print(f"Time to process: {round((t2-t1)/60,ndigits=1)} min\n")