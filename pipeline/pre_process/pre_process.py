from os import sep,walk
from os.path import isdir, join
import re
from time import time
from img_seq import img_seq_all
from background_sub import background_sub
from image_registration import register_img, channel_shift_register

def gather_all_images(parent_folder: str, file_type: str=None)-> list:
    # look through the folder and collect all image files
    if not isdir(parent_folder):
        raise ValueError(f"{parent_folder} is not a correct path. Try a full path")
    
    if file_type: extension = (file_type,)
    else: extension = ('.nd2','.tif','.tiff')
    print(f"\nSearching for {extension} files in {parent_folder}\n")
    # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
    imgS_path = []
    for root , _, files in walk(parent_folder):
        for f in files:
            # Look for all files with selected extension and that are not already processed 
            if not re.search(r'_f\d\d\d',f) and f.endswith(extension):
                imgS_path.append(join(sep,root+sep,f))
    return sorted(imgS_path)

# # # # # # # main function # # # # # # # 
def pre_process_all(parent_folder: str, active_channel_list: list, reg_channel: str, full_channel_list: list=None, 
         file_type: str=None, img_seq_overwrite: bool=False, bg_sub: bool=True, 
         sigma: float=0.0, size: int=7, register_images: bool=True, reg_mtd: str='rigid_body',
         chan_shift: bool=False, reg_ref: str='previous',
         bg_sub_overwrite: bool=False, chan_shift_overwrite: bool=False, reg_overwrite: bool=False )-> list:
    img_path_list = gather_all_images(parent_folder=parent_folder,file_type=file_type)
    
    exp_set_list = img_seq_all(img_path_list,active_channel_list,full_channel_list,img_seq_overwrite)
    
    if bg_sub:
        if img_seq_overwrite==True: bg_sub_overwrite=True
        background_sub(exp_set_list,sigma,size,bg_sub_overwrite)
    
    if chan_shift:
        if bg_sub_overwrite==True: chan_shift_overwrite=True
        channel_shift_register(exp_set_list,reg_mtd,reg_channel,chan_shift_overwrite)
    
    if register_images:
        register_img(exp_set_list,reg_channel,reg_mtd,reg_ref,reg_overwrite)
        
    return exp_set_list
    

if __name__ == "__main__":
    

    # Test
    active_channel_list = ['GFP','RFP','BFP']

    parent_folder = '/Users/benhome/BioTool/GitHub/cp_dev/Test_images/Run3'
    
    t1 = time()
    exp_set_list = pre_process_all(parent_folder,active_channel_list,'RFP',bg_sub=True,
                         chan_shift=True,register_images=False,file_type='.nd2',
                         img_seq_overwrite=True,bg_sub_overwrite=False,chan_shift_overwrite=False)
    t2 = time()
    if t2-t1<60: print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")
    else: print(f"Time to process: {round((t2-t1)/60,ndigits=1)} min\n")