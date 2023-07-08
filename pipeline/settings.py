from dataclasses import dataclass,fields,field
import json
from os import sep
from os.path import join


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
    pixel_microns: float
    interval_sec: float
    file_type: str
    img_path: str
    level_0_tag: str
    level_1_tag: str
    active_channel_list: list
    full_channel_list: list
        
    def from_json(self, json_path: str):
        with open(json_path) as fp:
            file = json.load(fp)
        
        fieldSet = {f.name for f in fields(self) if f.init}
        filteredArgDict = {k : v for k, v in file.items() if k in fieldSet}
        return self(**filteredArgDict)
    
    def from_metadata(self,meta: dict):
        fieldSet = {f.name for f in fields(self) if f.init}
        filteredArgDict = {k : v for k, v in meta.items() if k in fieldSet}
        return self(**filteredArgDict)
    
    def save_as_json(self):
        with open(join(sep,self.exp_path+sep,'exp_settings.json'),'w') as fp:
            json.dump(self.__dict__,fp,indent=4)







if __name__ == '__main__':
    json_path = '/Users/benhome/BioTool/GitHub/cp_dev/exp_settings.json'
    
    settings = Settings.from_json(Settings,json_path=json_path)
    print(settings)