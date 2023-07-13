from dataclasses import dataclass,fields,field
import json
from os import sep,listdir
from os.path import join
from functools import cached_property


@dataclass
class Settings():
    """Get metadata from nd2 or tif file, using ND2Reader or TiffFile and ImageJ"""
    exp_path: str
    img_width: int
    img_length: int
    n_frames: int
    full_n_channels: int
    n_slices: int
    n_series: int
    active_channel_list: list
    full_channel_list: list
    img_path: str
    
    pixel_microns: float
    interval_sec: float
    file_type: str
    level_0_tag: str
    level_1_tag: str
    background_sub: list = field(default_factory=list)
    channel_shift_corrected: list = field(default_factory=list)
    img_registered: list = field(default_factory=list)
    
    @cached_property
    def processed_image_list(self)-> list:
        im_folder = join(sep,self.exp_path+sep,'Images')
        return [join(sep,im_folder+sep,f) for f in sorted(listdir(im_folder)) if f.endswith('.tif')]
    
    @property
    def register_images_list(self)-> list:
        im_folder = join(sep,self.exp_path+sep,'Images_Registered')
        return [join(sep,im_folder+sep,f) for f in sorted(listdir(im_folder)) if f.endswith('.tif')]
    
    def from_json(self, json_path: str)-> dict:
        with open(json_path) as fp:
            file = json.load(fp)
        
        fieldSet = {f.name for f in fields(self) if f.init}
        filteredArgDict = {k : v for k, v in file.items() if k in fieldSet}
        return self(**filteredArgDict)
    
    def from_metadata(self, meta: dict)->dict:
        fieldSet = {f.name for f in fields(self) if f.init}
        filteredArgDict = {k : v for k, v in meta.items() if k in fieldSet}
        return self(**filteredArgDict)
    
    def save_as_json(self)->None:
        with open(join(sep,self.exp_path+sep,'exp_settings.json'),'w') as fp:
            json.dump(self.__dict__,fp,indent=4)


if __name__ == '__main__':
    json_path = '/Users/benhome/BioTool/GitHub/cp_dev/Test_images/Run2/c2z25t23v1_nd2_s1/exp_settings.json'
    
    settings = Settings.from_json(Settings,json_path=json_path)
    print(settings)
    
    