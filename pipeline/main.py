from __future__ import annotations
from os import getcwd
import sys
parent_dir = getcwd()
sys.path.append(parent_dir)
from time import time
# TODO: fix module path 
from ImageAnalysis_pipeline.pipeline.pre_process.main_pre_process import pre_process_all
from ImageAnalysis_pipeline.pipeline.segmentation.segmentation import simple_threshold
from ImageAnalysis_pipeline.pipeline.segmentation.cp_segmentation import cellpose_segmentation
if __name__ == "__main__":

    t1 = time()
    preprocess_parameters = {'parent_folder': '/Users/benhome/BioTool/GitHub/cp_dev/Test_images/Run2',
                             'active_channel_list': ['GFP','RFP'],
                             'full_channel_list': None, 
                             'file_type': '.nd2',
                             'img_seq_overwrite': False,
                             'bg_sub': True,'sigma': 0.0,'size': 7,'bg_sub_overwrite': False,
                             'chan_shift': False, 'reg_channel': 'RFP', 'reg_mtd': 'rigid_body', 'chan_shift_overwrite': False,
                             'register_images': True, 'reg_ref': 'previous', 'reg_overwrite': False,
                             'blur': True, 'blur_kernel': (15,15), 'blur_sigma': 5,'img_fold_src': None, 'blur_overwrite': False,}
                    
    segmentation_parameters = {'manual_threshold': None, 'simple_thresold_overwrite': True, 'img_fold_src': None}
    
    cp_segmentation_parameters = {'channel_seg':'RFP','model_type':'cyto2','nuclear_marker':None,
                                  'cellpose_overwrite':True,'stitch':None,'img_fold_src':'Images_Registered',
                                  'diameter':60.,'do_3D':True,'flow_threshold':0.4,'cellprob_threshold':0.0}
    
    
    exp_set_list = pre_process_all(**preprocess_parameters)
    # exp_set_list = simple_threshold(exp_set_list,**segmentation_parameters)
    exp_set_list = cellpose_segmentation(exp_set_list,**cp_segmentation_parameters)
    
    t2 = time()
    if t2-t1<60: print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")
    else: print(f"Time to process: {round((t2-t1)/60,ndigits=1)} min\n")