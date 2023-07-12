from os import sep,scandir,walk,listdir,mkdir
import re
from time import time
from nd2reader import ND2Reader
from tifffile import imwrite,imread
from os.path import join,isdir,exists
import numpy as np
from metadata import get_metadata
from settings import Settings
from smo import SMO
from pystackreg import StackReg
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor


# # # # # # # Utility # # # # # # # 

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

def _name_img_list(meta: dict)-> list:
    # Create a name for each image
    img_name_list = []
    for serie in range(meta['n_series']):
        for t in range(meta['n_frames']):
            for z in range(meta['n_slices']):
                for chan in meta['active_channel_list']:
                    img_name_list.append([meta,chan+'_s%02d'%(serie+1)+'_f%04d'%(t+1)+'_z%04d'%(z+1)])
    return img_name_list

def _is_active(exp_path: str)-> bool:
    if exists(join(sep,exp_path+sep,'REMOVED_EXP.txt')):
        return False
    return True

def _is_img_processed(img_folder: str)-> list:
    if any(scandir(img_folder)):
        return True
    return False

def load_stack(img_list: list, channel_list: str or list, frame_range: int or range or list)-> np.ndarray:
    # Check for channel
    if isinstance(channel_list,str):
        channel_list = [channel_list]

    if isinstance(frame_range,int):
        frame_range = [frame_range]
    elif isinstance(frame_range,range):
        frame_range = list(frame_range)
    
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


# # # # # # # Image sequence # # # # # # # 
def _write_ND2(img_data: tuple)-> None:
    # Unpack img_data
    meta,img_name = img_data
    img_obj = ND2Reader(meta['img_path'])
    serie,frame,z_slice = [int(i[1:])-1 for i in img_name.split('_')[1:]]
    chan = meta['full_channel_list'].index(img_name.split('_')[0])
    
    # Get the image       
    if meta['n_slices']>1: 
        img = img_obj.get_frame_2D(c=chan,t=frame,z=z_slice,x=meta['img_width'],y=meta['img_length'],v=serie)
    else: img = img_obj.get_frame_2D(c=chan,t=frame,x=meta['img_width'],y=meta['img_length'],v=serie)
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

def _write_tif(img_data: tuple)-> None:
    # Unpack img_data
    meta,img_name,img = img_data
    # img = _expand_dim_tif(meta['img_path'],meta['axes'])
    _,frame,z_slice = [int(i[1:])-1 for i in img_name.split('_')[1:]]             
    chan = meta['full_channel_list'].index(img_name.split('_')[0])
    
    im_folder = join(sep,meta['exp_path_list'][0]+sep,'Images')
    imwrite(join(sep,im_folder+sep,img_name)+".tif",img[frame,z_slice,chan,...].astype(np.uint16))
    
def write_img(meta: dict)-> None:
    # Create all the names for the images+metadata
    img_name_list = _name_img_list(meta)
    
    # for img_name in img_name_list:
    #     if meta['file_type'] == '.nd2':  
    #         _write_ND2(img_name)
    #     elif meta['file_type'] == '.tif':
    #         _write_tif(img_name)
    # if meta['file_type'] == '.nd2':
    #     with Pool() as pool:
    #         results = pool.imap_unordered(_write_ND2,img_name_list)
    #         for result in results:
    #             result.join()
    # elif meta['file_type'] == '.tif':
    #     with Pool() as pool:
    #         results = pool.imap_unordered(_write_tif,img_name_list)
    #         for result in results:
    #             result.join()
            
    if meta['file_type'] == '.nd2':
        with ThreadPoolExecutor() as executor:
            executor.map(_write_ND2,img_name_list)
    elif meta['file_type'] == '.tif':
        img = _expand_dim_tif(meta['img_path'],meta['axes'])
        img_name_list = [x+[img] for x in img_name_list]
        with ThreadPoolExecutor() as executor:
            executor.map(_write_tif,img_name_list)

def _load_settings(exp_path: str, meta: dict)-> dict:
    if exists(join(sep,exp_path+sep,'exp_settings.json')):
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
    for serie in range(meta['n_series']):
        exp_path = meta['exp_path_list'][serie]
        meta['exp_path'] = exp_path
        
        img_folder = join(sep,exp_path+sep,'Images')
        if not exists(img_folder):
            mkdir(img_folder)
        
        if not _is_active(exp_path):
            print(f"-> Exp.: {exp_path} has been removed\n")
            continue
        
        if _is_img_processed(img_folder) and not img_seq_overwrite:
            print(f"-> Exp.: {exp_path} has already been processed\n")
            settings_list.append(_load_settings(exp_path,meta))
            continue
        
        # If images are not processed
        print(f"-> Exp.: {exp_path} is being processed\n")
        write_img(meta)
        
        settings = Settings.from_metadata(Settings,meta)
        settings.save_as_json()
        settings_list.append(settings)
    return settings_list
    
def process_all_imgs_file(imgS_path: list, active_channel_list: list, full_channel_list: list=None, img_seq_overwrite: bool=False)-> list:
    settings_list = []
    for img_path in imgS_path:
        settings_list.extend(create_img_seq(img_path,active_channel_list,full_channel_list,img_seq_overwrite))
    return settings_list


# # # # # # # # Image bg_substraction # # # # # # # #
def background_sub(settings_list: list, sigma: float=0.0, size: int=7, bg_sub_overwrite: bool=False)-> list:
    for settings in settings_list:
        if settings.background_sub and not bg_sub_overwrite:
            print(f"--> Background substraction was already apply on {settings.img_path}")
            continue
        print(f"--> Applying background substraction on {settings.img_path}, with sigma={sigma} and size={size}")
        _apply_bg_sub((settings.img_width,settings.img_length),settings.processed_image_list,sigma,size)
        settings.background_sub = (f"sigma={sigma}",f"size={size}")
        settings.save_as_json()
    return settings_list

def _apply_bg_sub(shape: tuple, processed_image_list: list, sigma: float=0.0, size: int=7)-> None:
    # Initiate SMO
    smo = SMO(shape=shape,sigma=sigma,size=size)
    
    for proc_img_path in processed_image_list:
        img = imread(proc_img_path)
        bg_img = smo.bg_corrected(img)
        # Reset neg val to 0
        bg_img[bg_img<0] = 0
        imwrite(proc_img_path,bg_img.astype(np.uint16))


# # # # # # # # Image Registration # # # # # # # # # 
def register_channel_shift(settings_list: list, reg_mtd: str, reg_channel: str, chan_shift_overwrite: bool=False)-> None:
    for settings in settings_list:
        if settings.channel_shift_corrected and not chan_shift_overwrite:
            print(f"--> Channel shift was already apply on {settings.img_path}")
            continue
        stackreg = _reg_mtd(reg_mtd)
        _correct_chan_shift(stackreg,settings,reg_channel)
        settings.channel_shift_corrected = [f"reg_channel={reg_channel}",f"reg_mtd={reg_mtd}"]
        settings.save_as_json()

def register_img(settings_list: list, reg_channel: str, reg_mtd: str, reg_ref: int, reg_overwrite: bool=False)-> None:
    for settings in settings_list:
        img_folder = join(sep,settings.exp_path+sep,'Images_Registered')
        if not exists(img_folder):
            mkdir(img_folder)
        
        if settings.img_registered and not reg_overwrite:
            print(f"--> Registration was already apply on {settings.exp_path}")
            continue
        
        stackreg = _reg_mtd(reg_mtd)
        if reg_ref=='first':
            print(f"--> Registering {settings.exp_path} with first image and {reg_mtd} method")
            _register_with_first(stackreg,settings,reg_channel,img_folder)
        elif reg_ref=='previous':
            print(f"--> Registering {settings.exp_path} with previous image and {reg_mtd} method")
            _register_with_previous(stackreg,settings,reg_channel,img_folder)
        elif reg_ref=='mean':
            print(f"--> Registering {settings.exp_path} with mean image and {reg_mtd} method")
            _register_with_mean(stackreg,settings,reg_channel,img_folder)
        settings.img_registered = [f"reg_channel={reg_channel}",f"reg_mtd={reg_mtd}",f"reg_ref={reg_ref}"]
        settings.save_as_json()
            
def _reg_mtd(reg_mtd: str)-> StackReg:
    mtd_list = ['translation','rigid_body','scaled_rotation','affine','bilinear']
    if reg_mtd not in mtd_list:
        raise ValueError(f"{reg_mtd} is not valid. Please only put {mtd_list}")
        
    if reg_mtd=='translation':       stackreg = StackReg(StackReg.TRANSLATION)
    elif reg_mtd=='rigid_body':      stackreg = StackReg(StackReg.RIGID_BODY)
    elif reg_mtd=='scaled_rotation': stackreg = StackReg(StackReg.SCALED_ROTATION)
    elif reg_mtd=='affine':          stackreg = StackReg(StackReg.AFFINE)
    elif reg_mtd=='bilinear':        stackreg = StackReg(StackReg.BILINEAR)
    return stackreg
        
def _correct_chan_shift(stackreg: StackReg, settings: Settings, reg_channel: str)-> None:
    if settings.channel_shift_corrected:
        print(f"--> Channel shift correction was already apply on {settings.exp_path}")
        return settings
    
    # If not already corrected
    print(f"--> Applying channel shift correction on {settings.exp_path}")
    reg_img_path = join(sep,settings.exp_path+sep,'Images')
    chan_list = settings.active_channel_list.copy()
    chan_list.remove(reg_channel)
    
    for f in range(settings.n_frames):
        print(f"---> Frame {f+1}/{settings.n_frames}")
        # Load ref
        ref = load_stack(img_list=settings.processed_image_list,channel_list=reg_channel,frame_range=f)
        # Load im
        frame_name = '_f%04d'%(f+1)
        for chan in chan_list:
            im = load_stack(img_list=settings.processed_image_list,channel_list=chan,frame_range=f)
            for z in range(settings.n_slices):
                # Build z name
                z_name = '_z%04d.tif'%(z+1)
                # Apply transfo
                if settings.n_slices>1: reg_im = stackreg.register_transform(ref[z,...],im[z,...])
                else: reg_im = stackreg.register_transform(ref,im)
                # Save
                reg_im[reg_im<0] = 0
                imwrite(join(sep,reg_img_path+sep,chan+frame_name+z_name),reg_im.astype(np.uint16))

def _register_with_first(stackreg: StackReg, settings: Settings, reg_channel: str, img_folder: str)-> None:
    # Load ref image
    img_ref = load_stack(img_list=settings.processed_image_list,channel_list=reg_channel,frame_range=0)
    if settings.n_slices>1: img_ref = np.amax(img_ref,axis=0)
    
    for f in range(settings.n_frames):
        # Load image to register
        img = load_stack(img_list=settings.processed_image_list,channel_list=reg_channel,frame_range=f)
        if settings.n_slices>1: img = np.amax(img,axis=0)
        # Get the transfo matrix
        tmats = stackreg.register(ref=img_ref,mov=img)
        
        for chan in settings.active_channel_list:
            # Load image to transform
            img = load_stack(img_list=settings.processed_image_list,channel_list=chan,frame_range=f)
            for z in range(settings.n_slices):
                # Apply transfo
                if settings.n_slices==1: reg_img = stackreg.transform(mov=img,tmat=tmats)
                else: reg_img = stackreg.transform(mov=img[z,...],tmat=tmats)
                # Save
                reg_img[reg_img<0] = 0
                imwrite(join(sep,img_folder+sep,chan+'_f%04d'%(f+1)+'_z%04d.tif'%(z+1)),reg_img.astype(np.uint16))

def _register_with_mean(stackreg: StackReg, settings: Settings, reg_channel: str, img_folder: str)-> None:
    # Load ref image
    if settings.n_slices==1: img_ref = np.mean(load_stack(img_list=settings.processed_image_list,channel_list=reg_channel,frame_range=range(settings.n_frames)),axis=0)
    else: img_ref = np.mean(np.amax(load_stack(img_list=settings.processed_image_list,channel_list=reg_channel,frame_range=range(settings.n_frames)),axis=1),axis=0)

    for f in range(settings.n_frames):
        # Load image to register
        img = load_stack(img_list=settings.processed_image_list,channel_list=reg_channel,frame_range=f)
        if settings.n_slices>1: img = np.amax(img,axis=0)
        # Get the transfo matrix
        tmats = stackreg.register(ref=img_ref,mov=img)
        
        for chan in settings.active_channel_list:
            # Load image to transform
            img = load_stack(img_list=settings.processed_image_list,channel_list=chan,frame_range=f)
            for z in range(settings.n_slices):
                # Apply transfo
                if settings.n_slices==1: reg_img = stackreg.transform(mov=img,tmat=tmats)
                else: reg_img = stackreg.transform(mov=img[z,...],tmat=tmats)
                # Save
                reg_img[reg_img<0] = 0
                imwrite(join(sep,img_folder+sep,chan+'_f%04d'%(f+1)+'_z%04d.tif'%(z+1)),reg_img.astype(np.uint16))

def _register_with_previous(stackreg: StackReg, settings: Settings, reg_channel: str, img_folder: str)-> None:
    for f in range(1,settings.n_frames):
        # Load ref image
        if f==1:
            img_ref = load_stack(img_list=settings.processed_image_list,channel_list=reg_channel,frame_range=f-1)
            if settings.n_slices>1: img_ref = np.amax(img_ref,axis=0)
        else:
            img_ref = load_stack(img_list=settings.register_images_list,channel_list=reg_channel,frame_range=f-1)
            if settings.n_slices>1: img_ref = np.amax(img_ref,axis=0)
        # Load image to register
        img = load_stack(img_list=settings.processed_image_list,channel_list=reg_channel,frame_range=f)
        if settings.n_slices>1: img = np.amax(img,axis=0)
        # Get the transfo matrix
        tmats = stackreg.register(ref=img_ref,mov=img)
        print(settings.active_channel_list)
        for chan in settings.active_channel_list:
            # Load image to transform
            img = load_stack(img_list=settings.processed_image_list,channel_list=chan,frame_range=f)
            fst_img = load_stack(img_list=settings.processed_image_list,channel_list=chan,frame_range=f-1)
            for z in range(settings.n_slices):
                # Copy the first image to the reg_folder
                if f==1:
                    if settings.n_slices==1: imwrite(join(sep,img_folder+sep,chan+'_f%04d'%(1)+'_z%04d.tif'%(z+1)),fst_img.astype(np.uint16))
                    else: imwrite(join(sep,img_folder+sep,chan+'_f%04d'%(1)+'_z%04d.tif'%(z+1)),fst_img[z,...].astype(np.uint16))
                # Apply transfo
                if settings.n_slices==1: reg_img = stackreg.transform(mov=img,tmat=tmats)
                else: reg_img = stackreg.transform(mov=img[z,...],tmat=tmats)
                # Save
                reg_img[reg_img<0] = 0
                imwrite(join(sep,img_folder+sep,chan+'_f%04d'%(f+1)+'_z%04d.tif'%(z+1)),reg_img.astype(np.uint16))




















def main(parent_folder: str, active_channel_list: list, reg_channel: str, full_channel_list: list=None, 
         file_type: str=None, img_seq_overwrite: bool=False, bg_sub: bool=True, 
         sigma: float=0.0, size: int=7, register_images: bool=True, reg_mtd: str='rigid_body',
         chan_shift: bool=False, reg_ref: str='previous',
         bg_sub_overwrite: bool=False, chan_shift_overwrite: bool=False, reg_overwrite: bool=False )-> list:
    imgS_path = gather_all_images(parent_folder=parent_folder,file_type=file_type)
    
    settings_list = process_all_imgs_file(imgS_path,active_channel_list,full_channel_list,img_seq_overwrite)
    
    if bg_sub:
        if img_seq_overwrite==True: bg_sub_overwrite=True
        background_sub(settings_list,sigma,size,bg_sub_overwrite)
    
    if chan_shift:
        if bg_sub_overwrite==True: chan_shift_overwrite=True
        register_channel_shift(settings_list,reg_mtd,reg_channel,chan_shift_overwrite)
    
    if register_images:
        register_img(settings_list,reg_channel,reg_mtd,reg_ref,reg_overwrite)
        
    return settings_list
    

if __name__ == "__main__":
    

    # Test
    active_channel_list = ['GFP','RFP']

    parent_folder = '/Users/benhome/BioTool/GitHub/cp_dev/Test_images/Run2'
    
    t1 = time()
    settings_list = main(parent_folder,active_channel_list,'RFP',bg_sub=False,
                         chan_shift=False,register_images=False,img_seq_overwrite=True)
    t2 = time()
    if t2-t1<60: print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")
    else: print(f"Time to process: {round((t2-t1)/60,ndigits=1)} min\n")