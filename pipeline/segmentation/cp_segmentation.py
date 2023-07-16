from __future__ import annotations
from os import getcwd,sep,mkdir
import sys

parent_dir = getcwd()
sys.path.append(parent_dir)

from cellpose import models, core
from cellpose.io import logger_setup
from os.path import join, isdir
import cv2
from skimage.morphology import remove_small_objects,remove_small_holes
import numpy as np
from tifffile import imsave
from IPython.display import clear_output
from concurrent.futures import ThreadPoolExecutor
from ImageAnalysis_pipeline.pipeline.classes import Experiment,ImageProperties
from ImageAnalysis_pipeline.pipeline.loading_data import load_stack,_img_list_src

def _apply_cellpose_segmentation(img_data: list)-> None:
    img_list,stack_channel_list,frame,model,cellpose_eval = img_data
    img = load_stack(img_list,stack_channel_list,[frame])
    img_path = img_list[0].replace("Images","Masks_Cellpose").replace('_Registered','').replace('_Blured','')
    
    # Run Cellpose. Returns 4 variables
    masks_cp, __, __, = model.eval(img,**cellpose_eval)
    
    # Save mask
    if masks_cp.ndim==3:
        for z_silce in masks_cp.shape[0]:
            savedir = img_path.replace("_z0001",f"_z{z_silce+1:04d}")
            imsave(savedir,masks_cp[z_silce,...].astype('uint16'))
    else:
        imsave(img_path,masks_cp.astype('uint16'))
    
    
def cellpose_segmentation(exp_set_list: list[Experiment], channel_seg: str, model_type: str='cyto2', nuclear_marker: str=None,
                          cellpose_overwrite: bool=False, stitch: float=None, do_log: bool=True, img_fold_src: str=None, **kwargs)-> None:
    """Function to run cellpose segmentation. See https://github.com/MouseLand/cellpose for more details."""
    for exp_set in exp_set_list:
        # Check if exist
        if exp_set.process.cellpose_segmentation and not cellpose_overwrite:
                # Log
            print(f"--> Images have already been segmented with cellpose with {exp_set.process.cellpose_segmentation}")
            continue
        
        # Else run cellpose
        model_settings = _setup_cellpose_model(model_type,**kwargs)
        cellpose_eval = _setup_cellpose_eval(exp_set.img_properties,nuclear_marker,stitch,**kwargs)
        stack_channel_list = [channel_seg]
        if nuclear_marker: stack_channel_list.append(nuclear_marker)
        
        # Set log on
        if do_log: 
            logger_setup();
        
        print(f"--> Segmenting images for the '{channel_seg}' channel of exp. '{exp_set.exp_path}'")
        
        model = models.CellposeModel(**model_settings)
        img_list_src = _img_list_src(exp_set,img_fold_src)
        img_data = []
        for frame in range(exp_set.img_properties.frames):
            img_list = [file for file in img_list_src if f'_f{frame:04d}_' in file]
            img_data.append([img_list,stack_channel_list,frame,model,cellpose_eval])
        with ThreadPoolExecutor() as executor:
            executor.map(_apply_cellpose_segmentation,img_data)
        
        # Save settings
        exp_set.process.cellpose_segmentation = {'model_settings':model_settings,'cellpose_eval':cellpose_eval}
        exp_set.save_as_json()
            
def _setup_cellpose_model(model_type: str='cyto2', **kwargs)-> dict:
    # Default settings for cellpose model
    model_settings = {'gpu':False,'net_avg':False,'device':None,'diam_mean':30.,'residual_on':True,
                              'style_on':True,'concatenation':False,'nchan':2}
    
    build_in_models = ['cyto','nuclei','tissuenet','livecell','cyto2',
                     'CP','CPx','TN1','TN2','TN3','LC1','LC2','LC3','LC4']
    
    # To be able to raise error for kwargs, need to know eval keys as well
    eval_settings_keys = ['batch_size', 'channels', 'channel_axis', 'z_axis', 'invert', 'normalize',
                          'diameter', 'do_3D', 'anisotropy', 'net_avg', 'augment', 'tile', 'tile_overlap', 
                          'resample', 'interp', 'flow_threshold', 'cellprob_threshold', 'min_size', 
                          'stitch_threshold', 'rescale', 'progress', 'model_loaded']
    if kwargs:
        for k,v in kwargs.items():
            if k in model_settings:
                model_settings[k] = v
            elif k in eval_settings_keys:
                continue
            else:
                raise ValueError(f"Model setting '{k}' not recognized. Please choose one of the following: {model_settings.keys()}")
    
    
    if model_type not in build_in_models:
        if isdir(model_type):
            model_settings['pretrained_model'] = model_type
            model_settings['model_type'] = False
        else:
            raise ValueError(" ".join(f"Model type '{model_type}' not recognized.",
                                      f"Please choose one of the following: {build_in_models}",
                                        "or provide a path to a pretrained model."))
    else:
        model_settings['model_type'] = model_type
        model_settings['pretrained_model'] = False
    
    return model_settings

def _setup_cellpose_eval(img_properties: ImageProperties, nuclear_marker: str=None, stich: float=None, **kwargs)-> dict:
    # Default kwargs for cellpose eval
    cellpose_eval = {'batch_size':8,'channels':[0,0],'channel_axis':None,'z_axis':None,
            'invert':False,'normalize':True,'diameter':60.,'do_3D':False,'anisotropy':None,
            'net_avg':False,'augment':False,'tile':True,'tile_overlap':0.1,'resample':True,
            'interp':True,'flow_threshold':0.4,'cellprob_threshold':0.0,'min_size':500,
            'stitch_threshold':0.0,'rescale':None,'progress':None,'model_loaded':False}
    
    if nuclear_marker:
        cellpose_eval['channels'] = [1,2]
    
    if img_properties.n_slices>1:
        cellpose_eval['z_axis'] = 0
        cellpose_eval['do_3D'] = True
        cellpose_eval['anisotropy'] = 2.0
        if stich:
            cellpose_eval['stitch_threshold'] = stich
            cellpose_eval['anisotropy'] = None
            cellpose_eval['do_3D'] = False
    
    # To be able to raise error for kwargs, need to know model keys as well
    model_keys = ['gpu','net_avg','device','diam_mean','residual_on',
                    'style_on','concatenation','nchan']
    if kwargs:
        for k,v in kwargs.items():
            if k in cellpose_eval:
                cellpose_eval[k] = v
            elif k in model_keys:
                continue
            else:
                raise ValueError(" ".join(f"Cellpose run setting '{k}' not recognized.",
                                  f"Please choose one of the following: {cellpose_eval.keys()}"))
    return cellpose_eval
        
