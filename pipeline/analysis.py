from experiments import Exp_Indiv
from os import sep
from os.path import join
import numpy as np
import pandas as pd


class Analysis(Exp_Indiv):
    def __init__(self, exp_path,channel_seg):
        super().__init__(exp_path,channel_seg)

        self.tag = self.exp_prop['metadata']['tag']
        self.pixSize = self.exp_prop['metadata']['pixel_microns']

    def extract_channelData(self,imgFold,maskFold,channel_seg=None,df_ow=False):
        """
        Function that will extract the mean values of each masks on all the channels at every frames.
        Output is a dataframe.
        """
        if 'extract_channelData' not in self.exp_prop['fct_inputs'] or self.exp_prop['fct_inputs']['extract_channelData']['maskFold']!=maskFold or df_ow:
            # Log
            print(f"---> Extracting channel data with {maskFold}")

            # Load stack
            img_stack = Analysis.load_stack(imgFold_path=join(sep,self.exp_path+sep,imgFold))
            
            # Load mask(s)
            if channel_seg:
                chan_seg = channel_seg
            else:
                chan_seg = self.channel_seg

            if maskFold == 'Masks_Compartment':
                # Load mask stack
                masks = []
                masks.append(Analysis.load_mask(maskFold_path=join(sep,self.exp_path+sep,maskFold),channel_seg=chan_seg,mask_shape='mb'))
                masks.append(Analysis.load_mask(maskFold_path=join(sep,self.exp_path+sep,maskFold),channel_seg=chan_seg,mask_shape='cyto'))
                masks.append(Analysis.load_mask(maskFold_path=join(sep,self.exp_path+sep,maskFold),channel_seg=chan_seg,mask_shape='full'))
                mask_name = ['mb','cyto','full']
            elif maskFold == 'Masks_Class':
                # Load mask stack
                maskFold_path = join(sep,self.exp_path+sep,self.exp_prop['masks_process']['classification']['folder'])
                masks = [Analysis.load_mask(maskFold_path=maskFold_path,channel_seg=self.exp_prop['channel_seg']['Masks_Class'][0])]
                mask_name = ['']
                pos_lst = self.exp_prop['masks_process']['classification']['pos']
                neg_lst = self.exp_prop['masks_process']['classification']['neg']
            else:
                # Load mask stack
                masks = [Analysis.load_mask(maskFold_path=join(sep,self.exp_path+sep,maskFold),channel_seg=chan_seg)]
                mask_name = ['']
            
            # Create df to store analyses of the cell
            if maskFold == 'Masks_Class':
                keys = ['Cell','Frames','tag','exp']+[f"{chan}_{mn}" for mn in mask_name for chan in self.channel_list]+['cell_class','pos_cell']
            else:
                keys = ['Cell','Frames','tag','exp']+[f"{chan}_{mn}" for mn in mask_name for chan in self.channel_list]
            dict_analysis = {k:[] for k in keys}
            
            # Create tag and exp name
            split_path = self.exp_path.split(sep)[-1].split('_')
            exp_name = '_'.join([self.tag,split_path[0],split_path[-1]])

            # Get channel data
            for obj in list(np.unique(masks[0]))[1:]:
                if self.frames==1:
                    dict_analysis['Cell'].append(f"{exp_name}_cell{obj}")
                    dict_analysis['tag'].append(self.tag)
                    dict_analysis['exp'].append(exp_name)
                    dict_analysis['Frames'].append(1)
                    if maskFold == 'Masks_Class':
                        dict_analysis['cell_class'].append(f"if cell is {self.exp_prop['channel_seg']['Masks_Class'][1]}")
                        if obj in pos_lst:
                            dict_analysis['pos_cell'].append(1)
                        elif obj in neg_lst:
                            dict_analysis['pos_cell'].append(0)
                    for i,m in enumerate(masks):
                        for c,chan in enumerate(self.channel_list):
                            if self.chan_numb==1: dict_analysis[f"{chan}_{mask_name[i]}"].append(np.nanmean(a=img_stack,where=m==obj))
                            else: dict_analysis[f"{chan}_{mask_name[i]}"].append(np.nanmean(a=img_stack[...,c],where=m==obj))
                else:
                    for f in range(self.frames):
                        dict_analysis['Frames'].append(f+1)
                        dict_analysis['Cell'].append(f"{exp_name}_cell{obj}")
                        dict_analysis['tag'].append(self.tag)
                        dict_analysis['exp'].append(exp_name)
                        if maskFold == 'Masks_Class':
                            dict_analysis['cell_class'].append(f"if cell is {self.exp_prop['channel_seg']['Masks_Class'][1]}")
                            if obj in pos_lst:
                                dict_analysis['pos_cell'].append(1)
                            elif obj in neg_lst:
                                dict_analysis['pos_cell'].append(0)
                        for i,m in enumerate(masks):
                            for c,chan in enumerate(self.channel_list):
                                if self.chan_numb==1: dict_analysis[f"{chan}_{mask_name[i]}"].append(np.nanmean(a=img_stack[f,...],where=m[f,...]==obj))
                                else: dict_analysis[f"{chan}_{mask_name[i]}"].append(np.nanmean(a=img_stack[f,...,c],where=m[f,...]==obj))
            
            # Convert to df
            if hasattr(self,'df_analysis'): # It should only overwrite common columns
                self.df_analysis.update(pd.DataFrame.from_dict(dict_analysis))
            else:
                self.df_analysis = pd.DataFrame.from_dict(dict_analysis)
            
            # Update self.exp_prop and save df
            self.exp_prop['df_analysis'] = self.df_analysis
            self.exp_prop['fct_inputs']['extract_channelData'] = {'imgFold':imgFold,'maskFold':maskFold,'channel_seg':channel_seg}
            Analysis.save_exp_prop(exp_path=self.exp_path,exp_prop=self.exp_prop)
            self.df_analysis.to_csv(join(sep,self.exp_path+sep,'df_analysis.csv'),index=False)
        else:
            # Log
            print(f"---> Channel data are already extracted with {maskFold}")
            # Load df
            self.df_analysis = pd.read_csv(join(sep,self.exp_path+sep,'df_analysis.csv'))
        
    def extract_centroids(self,maskFold,channel_seg=None,df_ow=False):
        """
        This function extract the centroids of all masks at every frames.
        The output is a dataframe.
        """
        if 'extract_centroids' not in self.exp_prop['fct_inputs'] or self.exp_prop['fct_inputs']['extract_centroids']['maskFold']!=maskFold or df_ow:
            # Log
            print(f"---> Extracting centroids for {maskFold}")
            
            # Load mask(s)
            if channel_seg:
                chan_seg = channel_seg
            else:
                chan_seg = self.channel_seg

            if maskFold == 'Masks_Compartment':
                # Load mask stack
                mask_stack = Analysis.load_mask(maskFold_path=join(sep,self.exp_path+sep,maskFold),channel_seg=chan_seg,mask_shape='cyto')
            elif maskFold == 'Masks_Class':
                # Load mask stack
                maskFold_path = join(sep,self.exp_path+sep,self.exp_prop['masks_process']['classification']['folder'])
                mask_stack = Analysis.load_mask(maskFold_path=maskFold_path,channel_seg=self.exp_prop['channel_seg']['Masks_Class'][0])
            else:
                mask_stack = Analysis.load_mask(maskFold_path=join(sep,self.exp_path+sep,maskFold),channel_seg=chan_seg)
            
            # Create tag and exp name
            split_path = self.exp_path.split(sep)[-1].split('_')
            exp_name = '_'.join([self.tag,split_path[0],split_path[-1]])

            # Get centroids
            df = Analysis.centroids(mask_stack=mask_stack,frames=self.frames,z_slice=self.z_size,exp_name=exp_name)
            # Create attr if it doesn't exist
            if not hasattr(self, 'df_analysis'):
                self.df_analysis = df
            elif hasattr(self, 'df_analysis') and 'Cent.X' in self.df_analysis: # If just want to overwrite columns
                self.df_analysis.update(df)
            else:
                self.df_analysis = pd.merge(self.df_analysis,df,on=['Cell','Frames'])
        
            # Update self.exp_prop and save df
            self.exp_prop['df_analysis'] = self.df_analysis
            self.exp_prop['fct_inputs']['extract_centroids'] = {'maskFold':maskFold,'channel_seg':channel_seg}
            Analysis.save_exp_prop(exp_path=self.exp_path,exp_prop=self.exp_prop)
            self.df_analysis.to_csv(join(sep,self.exp_path+sep,'df_analysis.csv'),index=False)
        else:
            # Log
            print(f"---> Centroids are already extracted for {maskFold}")
            # Load df
            self.df_analysis = pd.read_csv(join(sep,self.exp_path+sep,'df_analysis.csv'))

    def cell_distance(self,imgFold,maskLabel='wound',df_ow=False):
        # Check maskLabel and centroids
        if type(maskLabel)==str:
            maskLabel = list(maskLabel)
        if type(maskLabel)!=list:
            raise TypeError(f"Maskname cannot be of type {type(maskLabel)}. Only string or list of strings are accepted")
        if not hasattr(self, 'df_analysis') or 'Cent.X' not in self.df_analysis:
            raise AttributeError(f"Centroids are missing for {self.exp_path}. Please run 'extract_centroids()' first")
        
        # Run analysis
        if 'cell_distance' not in self.exp_prop['fct_inputs'] or self.exp_prop['fct_inputs']['cell_distance']['maskLabel']!=maskLabel or df_ow:
            # Log
            print(f"---> Creating ref mask for {maskLabel}")
        
            # Convert mask to dmap
            for maskname in maskLabel:
                # Load dmap mask
                mask_ref = Analysis.apply_dmap(mask_stack=Analysis.ref_mask(imgFold_path=join(sep,self.exp_path+sep,imgFold),maskLabel=maskname,ref_mask_ow=df_ow),
                                                frames=self.frames)
                
                # Extract dmap
                self.df_analysis[maskname] = 0
                for ind in range(self.df_analysis.shape[0]):
                    if self.frames == 1:
                        dt_im = mask_ref.copy()
                    else:
                        fr = self.df_analysis['Frames'].iloc[ind]
                        dt_im = mask_ref[fr,...].copy()
                    cX = self.df_analysis['Cent.X'].iloc[ind]
                    cY = self.df_analysis['Cent.Y'].iloc[ind]
                    self.df_analysis.loc[ind,maskname] = np.ceil(dt_im[cY,cX]*self.pixSize)
                
                # TODO: add all the migration-related calculations here
                # Migration calculation

            # Update self.exp_prop and save df
            self.exp_prop['df_analysis'] = self.df_analysis
            self.exp_prop['fct_inputs']['cell_distance'] = {'imgFold':imgFold,'maskLabel':maskLabel,'df_ow':df_ow}
            Analysis.save_exp_prop(exp_path=self.exp_path,exp_prop=self.exp_prop)
            self.df_analysis.to_csv(join(sep,self.exp_path+sep,'df_analysis.csv'),index=False)
        else:
            # Log
            print(f"---> Cell distance was already created for {maskLabel}")
            # Load df
            self.df_analysis = pd.read_csv(join(sep,self.exp_path+sep,'df_analysis.csv'))

    def pixel_distance(self,imgFold,maskFold,channel_seg=None,maskLabel='wound',pix_ana_ow=False,ref_mask_ow=False): 
        
        if 'pixel_distance' not in self.exp_prop['fct_inputs'] or pix_ana_ow:
            # Load dmap ref_mask and mask
            mask_ref = Analysis.apply_dmap(mask_stack=Analysis.ref_mask(imgFold_path=join(sep,self.exp_path+sep,imgFold),maskLabel=maskLabel,ref_mask_ow=ref_mask_ow),
                                                    frames=self.frames)      
            # Load mask
            if channel_seg:
                chan_seg = channel_seg
            else:
                chan_seg = self.channel_seg
            mask_stack = Analysis.load_mask(maskFold_path=join(sep,self.exp_path+sep,maskFold),channel_seg=chan_seg,z_slice=1)
            if self.frames==1: mask_stack = [mask_stack] # just to be able to run it in the for loop

            # Convert frames to minutes
            interval = self.exp_prop['metadata']['interval_sec']
            ts = [0.0]+list(np.round(a=np.linspace(interval,interval*(self.frames-1),self.frames-1)/60,decimals=2))
            
            # Extract data
            self.df_pixel = pd.DataFrame()
            for f,t in enumerate(ts):
                d = {}
                d['dmap'] = np.round(mask_ref[f,...][mask_stack[f].astype(bool)]*self.pixSize).astype(int)
                for chan in self.channel_list:
                    img_stack = Analysis.load_stack(imgFold_path=join(sep,self.exp_path+sep,imgFold),channel_list=chan,input_range=[f])
                    if self.z_size>1: img_stack = np.amax(a=img_stack,axis=0)
                    d[chan] = img_stack[mask_stack[f].astype(bool)].astype(float)
                df = pd.DataFrame.from_dict(d)
                df['time'] = t
                self.df_pixel = pd.concat([self.df_pixel,df])
            
            # Add tag and exp_name
            self.df_pixel['tag'] = self.tag
            split_path = self.exp_path.split(sep)[-1].split('_')
            self.df_pixel['exp'] = '_'.join([self.tag,split_path[0],split_path[-1]])

            # Update self.exp_prop and save df
            self.exp_prop['df_pixel'] = self.df_pixel
            self.exp_prop['fct_inputs']['pixel_distance'] = {'imgFold':imgFold,'maskFold':maskFold,'channel_seg':channel_seg,'maskLabel':maskLabel}
            Analysis.save_exp_prop(exp_path=self.exp_path,exp_prop=self.exp_prop)
            self.df_pixel.to_csv(join(sep,self.exp_path+sep,'df_pixel.csv'),index=False)
        else:
            # Log
            print(f"---> {maskLabel} mask was already created for pixel distance and value already extracted")
            # Load df
            self.df_pixel = pd.read_csv(join(sep,self.exp_path+sep,'df_pixel.csv'))
    


    
    
