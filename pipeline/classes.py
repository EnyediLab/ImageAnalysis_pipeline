from __future__ import annotations
from dataclasses import dataclass,fields,field
import json
from os import sep,listdir,getcwd
from os.path import join
from functools import cached_property

@dataclass
class LoadClass:
    def from_dict(self, input_dict: dict)->dict:
        fieldSet = {f.name for f in fields(self) if f.init}
        filteredArgDict = {k : v for k, v in input_dict.items() if k in fieldSet}
        return self(**filteredArgDict)

@dataclass
class Process(LoadClass):
    background_sub: list = field(default_factory=list)
    channel_shift_corrected: list = field(default_factory=list)
    img_registered: list = field(default_factory=list)
    img_blured: list = field(default_factory=list)
    simple_threshold: list = field(default_factory=list)
    cellpose_segmentation: dict = field(default_factory=dict)

@dataclass
class ImageProperties(LoadClass):
    """Get metadata from nd2 or tif file, using ND2Reader or TiffFile and ImageJ"""
    img_width: int
    img_length: int
    n_frames: int
    full_n_channels: int
    n_slices: int
    n_series: int
    img_path: str
    
@dataclass
class Analysis(LoadClass):
    pixel_microns: float
    interval_sec: float
    file_type: str
    level_0_tag: str
    level_1_tag: str

@dataclass
class Experiment(LoadClass):
    exp_path: str
    active_channel_list: list
    full_channel_list: list
    img_properties: ImageProperties = field(default_factory=ImageProperties)
    analysis: Analysis = field(default_factory=Analysis)
    process: Process = field(default_factory=Process)

    @cached_property
    def processed_images_list(self)-> list:
        im_folder = join(sep,self.exp_path+sep,'Images')
        return [join(sep,im_folder+sep,f) for f in sorted(listdir(im_folder)) if f.endswith('.tif')]
    
    @property
    def register_images_list(self)-> list:
        im_folder = join(sep,self.exp_path+sep,'Images_Registered')
        return [join(sep,im_folder+sep,f) for f in sorted(listdir(im_folder)) if f.endswith('.tif')]
    
    @property
    def blur_images_list(self)-> list:
        im_folder = join(sep,self.exp_path+sep,'Images_Blured')
        return [join(sep,im_folder+sep,f) for f in sorted(listdir(im_folder)) if f.endswith('.tif')]

    def save_as_json(self)->None:
        main_dict = self.__dict__.copy()
        main_dict['img_data'] = self.img_properties.__dict__
        main_dict['analysis'] = self.analysis.__dict__
        main_dict['process'] = self.process.__dict__
        
        with open(join(sep,self.exp_path+sep,'exp_settings.json'),'w') as fp:
            json.dump(main_dict,fp,indent=4)
    

def init_from_json(json_path: str)-> Experiment:
    with open(json_path) as fp:
        meta = json.load(fp)
    meta['img_data'] = ImageProperties.from_dict(ImageProperties,meta['img_data'])
    meta['analysis'] = Analysis.from_dict(Analysis,meta['analysis'])
    meta['process'] = Process.from_dict(Process,meta['process'])
    return Experiment.from_dict(Experiment,meta)
    
def init_from_dict(input_dict: dict)-> Experiment:
    input_dict['img_data'] = ImageProperties.from_dict(ImageProperties,input_dict)
    input_dict['analysis'] = Analysis.from_dict(Analysis,input_dict)
    input_dict['process'] = Process.from_dict(Process,input_dict)
    return Experiment.from_dict(Experiment,input_dict)

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


if __name__ == '__main__':
    json_path = '/Users/benhome/BioTool/GitHub/cp_dev/Test_images/Run2/c2z25t23v1_nd2_s1/exp_settings.json'
    
    # settings = Settings.from_json(Settings,json_path=json_path)
    # print(settings)
    # exp = init_from_json(json_path)
    # print(exp.process.background_sub)
    # d = {'a':1,'b':2,'c':3,'d':4,'e':5}
    # proc = Process.from_dict(Process,d)
    # print(type(proc))
    print(getcwd())
    