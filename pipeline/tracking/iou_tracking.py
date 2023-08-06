from __future__ import annotations
from os import getcwd, sep, mkdir, listdir
import sys

parent_dir = getcwd()
sys.path.append(parent_dir)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from os.path import join, isdir
from ImageAnalysis_pipeline.pipeline.classes import Experiment
from ImageAnalysis_pipeline.pipeline.loading_data import is_processed, mask_list_src, load_stack
from ImageAnalysis_pipeline.pipeline.mask_transformation.mask_morph import morph_missing_mask, morph_missing_mask_para
from cellpose.utils import stitch3D
from cellpose.metrics import _intersection_over_union
from scipy.stats import mode
from tifffile import imsave
import numpy as np



def modif_stitch3D(masks,stitch_threshold):
    print('  ---> Tracking cells...')
    # Invert stitch_threshold
    stitch_threshold = 1 - stitch_threshold
    # basic stitching from Cellpose
    masks = stitch3D(masks, stitch_threshold=stitch_threshold)
    
    # create mastermask to have all possible cells on one mask. Doing this by doing 'mode' operation 
    # to get the value present in most t-frames per pixel. Ignoring backgound by setting zero to nan. Threfore conversion to float is needed.
    rawmasks_ignorezero = masks.copy().astype(float)
    rawmasks_ignorezero[rawmasks_ignorezero == 0] = np.nan
    master_mask = mode(rawmasks_ignorezero, axis=0, keepdims=False, nan_policy='omit')[0]
    master_mask = master_mask.astype(int)

    # second stitch round by using mastermask to compair with every frame
    #slighly changed code from Cellpose 'stitch3D'
    """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
    mmax = masks[0].max()
    empty = 0

    for i in range(len(masks)):
        iou = _intersection_over_union(masks[i], master_mask)[1:,1:]
        if not iou.size and empty == 0:
            mmax = masks[i].max()
        elif not iou.size and not empty == 0:
            icount = masks[i].max()
            istitch = np.arange(mmax+1, mmax + icount+1, 1, int)
            mmax += icount
            istitch = np.append(np.array(0), istitch)
            masks[i] = istitch[masks[i]]
        else:
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1)==0.0)[0]
            istitch[ino] = 0
            istitch = np.append(np.array(0), istitch)
            masks[i] = istitch[masks[i]]
            empty = 1
    return masks

def check_mask_size(mask_stack: np.ndarray, shape_thres_percent: float)-> np.ndarray:
    print('  ---> Checking mask size')
    new_mask = np.zeros((mask_stack.shape))
    for obj in list(np.unique(mask_stack))[1:]:
        temp = mask_stack.copy()
        temp[temp!=obj] = 0
        t,_,_=np.where(temp!=0)
        f_lst,size_lst=np.unique(t,return_counts=True)
        mean_size = np.mean(size_lst)
        up = mean_size+mean_size*shape_thres_percent # shape_threshold is the max % of up or down allowed
        down = mean_size-mean_size*shape_thres_percent
        temp[f_lst[np.where((size_lst<down)|(size_lst>up))]] = 0
        new_mask += temp
    return new_mask

def reassign_mask_val(mask_stack: np.ndarray)-> np.ndarray:
    print('  ---> Reassigning masks value')
    for n, val in enumerate(list(np.unique(mask_stack))):
        mask_stack[mask_stack == val] = n
    return mask_stack

def trim_mask(mask_stack: np.ndarray, numb_frames: int)-> np.ndarray:
    print('  ---> Trimming masks')
    for obj in list(np.unique(mask_stack))[1:]:
        if len(set(np.where(mask_stack==obj)[0])) != numb_frames:
            mask_stack[mask_stack==obj] = 0
    return mask_stack


# # # # # # # # main functions # # # # # # # # # 
def iou_tracking(exp_set_list: list[Experiment], channel_seg: str, mask_fold_src: str,
                 stitch_thres_percent: float=0.75, shape_thres_percent: float=0.2,
                 iou_track_overwrite: bool=False, n_mask: int=5)-> list[Experiment]:
    
    for exp_set in exp_set_list:
        if is_processed(exp_set.masks.iou_tracking,channel_seg,iou_track_overwrite):
            print(f" --> Cells have already been tracked for the '{channel_seg}' channel")
            continue
        
        # Track images
        print(f" --> Tracking cells for the '{channel_seg}' channel")
        
        # Create blur dir and apply blur
        if not isdir(join(sep,exp_set.exp_path+sep,'Masks_IoU_Track')):
            mkdir(join(sep,exp_set.exp_path+sep,'Masks_IoU_Track'))
        
        # Load masks
        mask_src_list = mask_list_src(exp_set,mask_fold_src)
        mask_stack = load_stack(mask_src_list,[channel_seg],range(exp_set.img_properties.n_frames))
        
        # Track masks
        mask_stack = modif_stitch3D(mask_stack,stitch_thres_percent)
        
        # Check shape size for detecting merged cells
        mask_stack = check_mask_size(mask_stack,shape_thres_percent)
        
        # Re-assign the new value to the masks and obj
        mask_stack = reassign_mask_val(mask_stack)
        
        # Morph missing masks
        mask_stack = morph_missing_mask(mask_stack,n_mask)
        # mask_stack = morph_missing_mask_para(mask_stack,n_mask)

        # Trim masks
        mask_stack = trim_mask(mask_stack,exp_set.img_properties.n_frames)
        
        # Save masks
        for i,path in enumerate(mask_src_list):
            mask_path = path.replace('Masks','Masks_IoU_Track').replace('_Cellpose','').replace('_Threshold','')
            imsave(mask_path,mask_stack[i,...].astype('uint16'))
        
        # Save settings
        if exp_set.masks.iou_tracking:
            exp_set.masks.iou_tracking.update({channel_seg:{'mask_fold_src':mask_fold_src,'stitch_thres_percent':stitch_thres_percent,
                                        'shape_thres_percent':shape_thres_percent,'n_mask':n_mask}})
        else: 
            exp_set.masks.iou_tracking = {channel_seg:{'mask_fold_src':mask_fold_src,'stitch_thres_percent':stitch_thres_percent,
                                        'shape_thres_percent':shape_thres_percent,'n_mask':n_mask}}
        exp_set.save_as_json()
    return exp_set_list



if __name__ == "__main__":
    folder = '/Users/benhome/BioTool/GitHub/cp_dev/Test_images/Run4/c4z1t91v1_s1/Masks_Cellpose'
    mask_folder_src = [join(sep,folder+sep,file) for file in sorted(listdir(folder)) if file.endswith('.tif')]
    mask_stack = load_stack(mask_folder_src,['RFP'],range(91))
    print(type(mask_stack))