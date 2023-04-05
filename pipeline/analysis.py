from experiments import Exp_Indiv
from os import sep
from os.path import join,exists
import numpy as np
import pandas as pd
from math import sqrt


class Analysis(Exp_Indiv):
    def __init__(self, exp_path,channel_seg,interval=None,tag=None, tag_ow=False):
        super().__init__(exp_path,channel_seg)
        if tag: 
            if not isinstance(tag,dict): raise TypeError("tag needs to be a dictionary with 'exp_path' as key and 'tag' as value")
            self.tag = tag[exp_path]
        elif tag_ow:
            print('Tags were overwritten')
            self.tag = exp_path.split(sep)[-2]
        else: self.tag = self.exp_prop['metadata']['tag']
        self.pixSize = self.exp_prop['metadata']['pixel_microns']
        
        # Convert frames to minutes
        if interval: self.interval = interval
        else: self.interval = self.exp_prop['metadata']['interval_sec']
        self.ts = [0.0]+list(np.round(a=np.linspace(self.interval,self.interval*(self.frames-1),self.frames-1)/60,decimals=2))

    def extract_channelData(self,imgFold,maskFold,stim_time=None,start_baseline=0,posCont_time=None,channel_seg=None,df_ow=False): #TODO: batch it
        """
        Function that will extract the mean values of each masks on all the channels at every frames.
        Output is a dataframe.
        """
        if 'extract_channelData' not in self.exp_prop['fct_inputs'] or self.exp_prop['fct_inputs']['extract_channelData']['maskFold']!=maskFold or df_ow:
            # Log
            print(f"---> Extracting channel data with {maskFold}")

            # Setup mask loading
            if maskFold == 'Masks_Compartment':
                # Load mask variable
                chan_keys = self.exp_prop['channel_seg']['Masks_Compartment']
                masks = [{chan:[{'maskFold_path':join(sep,self.exp_path+sep,maskFold),'channel_seg':chan,'mask_shape':shape} for shape in ['_mb','_cyto','_full']]} for chan in chan_keys]
                # Create all possible keys for masks
                mask_keys = [f'{chan}{shape}' for chan in self.channel_list for shape in ['_mb','_cyto','_full']]
            elif maskFold == 'Masks_Class':
                # Load mask variable
                chan_keys = self.exp_prop['channel_seg']['Masks_Class']
                maskFold_path = join(sep,self.exp_path+sep,self.exp_prop['masks_process']['classification']['folder'])
                masks = [{chan_keys[0]:[{'maskFold_path':maskFold_path,'channel_seg':chan_keys[0]}]}]
                pos_lst = self.exp_prop['masks_process']['classification']['pos']
                neg_lst = self.exp_prop['masks_process']['classification']['neg']
                # Create all possible keys for masks
                mask_keys = [chan for chan in self.channel_list]
               
            else:
                # Load mask variable
                chan_keys = self.exp_prop['channel_seg'][maskFold]
                masks = [{chan:[{'maskFold_path':join(sep,self.exp_path+sep,maskFold),'channel_seg':chan}]} for chan in chan_keys]
                # Create all possible keys for masks
                mask_keys = [chan for chan in self.channel_list]
                
            # Create df to store analyses of the cell
            keys = ['cell','frames','time','mask_chan','tag','exp']+mask_keys
            if maskFold == 'Masks_Class': keys +=['cell_class','pos_cell']
            
            # Create tag and exp name
            split_path = self.exp_path.split(sep)[-1].split('_')
            exp_name = '_'.join([self.tag,split_path[0],split_path[-1]])

            # Get channel data
            df = pd.DataFrame()
            for d_chan in masks:
                masks_lst = list(*d_chan.values())
                dict_analysis = {k:[] for k in keys}
                for f,t in enumerate(self.ts):
                    
                    for m in masks_lst:
                        # Load mask
                        mask = Analysis.load_mask(input_range=[f],do_log=False,**m)
                        
                        for obj in list(np.unique(mask))[1:]:
                            dict_analysis['cell'].append(f"{exp_name}_{m['channel_seg']}_cell{obj}")
                            dict_analysis['mask_chan'].append(m['channel_seg'])
                            dict_analysis['tag'].append(self.tag)
                            dict_analysis['exp'].append(exp_name)
                            dict_analysis['frames'].append(f+1)
                            dict_analysis['time'].append(t)
                            if maskFold == 'Masks_Class':
                                dict_analysis['cell_class'].append(f"if cell is {chan_keys[1]}")
                                if obj in pos_lst:
                                    dict_analysis['pos_cell'].append(1)
                                elif obj in neg_lst:
                                    dict_analysis['pos_cell'].append(0)
                            for chan in self.channel_list:
                                # Load img
                                img = Analysis.load_stack(imgFold_path=join(sep,self.exp_path+sep,imgFold),channel_list=chan,input_range=[f])
                                for name in mask_keys:
                                    if chan in name:
                                        dict_analysis[name].append(np.nanmean(a=img,where=mask==obj))
                df = pd.concat([df,pd.DataFrame.from_dict(dict_analysis)])

            # Transform df
            self.df_analysis = Analysis.transfo_df(df_input=df,channel_list=mask_keys,
                                                   stim_time=stim_time,start_baseline=start_baseline,posCont_time=posCont_time)
            
            # Update self.exp_prop and save df
            self.exp_prop['df_analysis'] = self.df_analysis
            self.exp_prop['fct_inputs']['extract_channelData'] = {'imgFold':imgFold,'maskFold':maskFold,'channel_seg':channel_seg}
            Analysis.save_exp_prop(exp_path=self.exp_path,exp_prop=self.exp_prop)
            df.to_csv(join(sep,self.exp_path+sep,'df_analysis.csv'),index=False)
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
            
            # Setup mask loading
            if maskFold == 'Masks_Compartment':
                # Load mask variable
                chan_keys = self.exp_prop['channel_seg']['Masks_Compartment']
                masks_dict = [{chan:[{'maskFold_path':join(sep,self.exp_path+sep,maskFold),'do_log':False,'mask_shape':'cyto','channel_seg':chan}]} for chan in chan_keys]
                
            elif maskFold == 'Masks_Class':
                # Load mask variable
                chan_keys = self.exp_prop['channel_seg']['Masks_Class']
                maskFold_path = join(sep,self.exp_path+sep,self.exp_prop['masks_process']['classification']['folder'])
                masks_dict = [{chan_keys[0]:[{'maskFold_path':maskFold_path,'do_log':False,'channel_seg':chan_keys[0]}]}]
            else:
                # Load mask variable
                chan_keys = self.exp_prop['channel_seg'][maskFold]
                masks_dict = [{chan:[{'maskFold_path':join(sep,self.exp_path+sep,maskFold),'do_log':False,'channel_seg':chan}]} for chan in chan_keys]
            
            # Create df to store analyses of the cell
            if self.z_size==1: axes_keys = ['y','x']
            else: axes_keys = ['y','x','z']
            keys = ['cell','frames','time','mask_chan','mask_ID']+axes_keys
            
            # Create tag and exp name
            split_path = self.exp_path.split(sep)[-1].split('_')
            exp_name = '_'.join([self.tag,split_path[0],split_path[-1]])

            # Get centroids                                         
            df = pd.DataFrame()
            for d_chan in masks_dict:
                masks_lst = list(*d_chan.values())
                dict_analysis = {k:[] for k in keys}
                for f,t in enumerate(self.ts):
                    for m in masks_lst:
                        # Load mask
                        mask = Analysis.load_mask(input_range=[f],**m)
                        for obj in list(np.unique(mask))[1:]:
                            dict_analysis['cell'].append(f"{exp_name}_{m['channel_seg']}_cell{obj}")
                            dict_analysis['mask_chan'].append(m['channel_seg'])
                            dict_analysis['frames'].append(f+1)
                            dict_analysis['time'].append(t)
                            dict_analysis['mask_ID'].append(obj)
                            if self.z_size==1: 
                                y,x = np.where(mask==obj)
                                if y.size > 0:
                                    dict_analysis["y"].append(round(np.nanmean(y)))
                                    dict_analysis["x"].append(round(np.nanmean(x)))
                            else:
                                z,y,x = np.where(mask==obj)
                                if y.size > 0:
                                    dict_analysis["y"].append(round(np.nanmean(y)))
                                    dict_analysis["x"].append(round(np.nanmean(x)))
                                    dict_analysis["z"].append(round(np.nanmean(z)))
                df = pd.concat([df,pd.DataFrame.from_dict(dict_analysis)])
            
            # Create attr if it doesn't exist
            if not hasattr(self, 'df_analysis'):
                self.df_analysis = pd.DataFrame.from_dict(df)
            else:
                self.df_analysis = pd.merge(self.df_analysis,pd.DataFrame.from_dict(df),on=['cell','frames','time','mask_chan'])
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

    def cell_distance(self,imgFold,maskLabel='wound',df_ow=False,ref_mask_ow=False): # TODO: Check if df_analysis exists
        # Check maskLabel and centroids
        if type(maskLabel)==str:
            maskLabel = [maskLabel]
        if type(maskLabel)!=list:
            raise TypeError(f"Maskname cannot be of type {type(maskLabel)}. Only string or list of strings are accepted")
        if not hasattr(self, 'df_analysis') or 'x' not in self.df_analysis:
            raise AttributeError(f"Centroids are missing for {self.exp_path}. Please run 'extract_centroids()' first")
        
        if ref_mask_ow: df_ow = True

        # Run analysis
        if 'cell_distance' not in self.exp_prop['fct_inputs'] or self.exp_prop['fct_inputs']['cell_distance']['maskLabel']!=maskLabel or df_ow:
            # Log
            print(f"---> Creating ref mask for {maskLabel}")
        
            # Convert mask to dmap
            for maskname in maskLabel:
                # Load dmap mask
                mask_ref = Analysis.apply_dmap(mask_stack=Analysis.ref_mask(imgFold_path=join(sep,self.exp_path+sep,imgFold),maskLabel=maskname,ref_mask_ow=ref_mask_ow),
                                                frames=self.frames)
                # Extract dmap
                self.df_analysis[maskname] = 0
                for ind in range(self.df_analysis.shape[0]):
                    f = self.df_analysis['frames'].iloc[ind]-1
                    # Load dmap mask
                    dt_im = mask_ref[f,...].copy()
                    cX = self.df_analysis['x'].iloc[ind]
                    cY = self.df_analysis['y'].iloc[ind]
                    self.df_analysis.loc[ind,maskname] = np.ceil(dt_im[cY,cX]*self.pixSize)
                
                # Add Migration column
                mig_keys = ['dOW','dEW','dOE','dist','tot_length','Dp','Dw','speed','tot_speed']
                keys = [f"{maskname}_{mk}" for mk in mig_keys]
                for col in keys: self.df_analysis[col] = 0

                # Migration calculation
                for cell in self.df_analysis.cell.unique():
                    df = self.df_analysis.loc[self.df_analysis['cell']==cell].copy()
                    ff = df['frames'].min()
                    df[f"{maskname}_dOW"] = int(df.loc[df['frames']==ff,maskname])
                    lf = df['frames'].max()
                    df[f"{maskname}_dEW"] = int(df.loc[df['frames']==lf,maskname])
                    yff,xff = df.loc[df['frames']==ff,['y','x']].values[0]
                    ylf,xlf = df.loc[df['frames']==lf,['y','x']].values[0]
                    df[f"{maskname}_dOE"] = np.round(sqrt((xff-xlf)**2+(yff-ylf)**2),decimals=2)
                    df.loc[df['frames']==ff,f"{maskname}_dist"] = 0
                    for f in sorted(list(df.frames.unique()))[1:]:
                        y1,x1 = df.loc[df['frames']==f-1,['y','x']].values[0]
                        y2,x2 = df.loc[df['frames']==f,['y','x']].values[0]
                        df.loc[df['frames']==f,f"{maskname}_dist"] = np.round(sqrt((x1-x2)**2+(y1-y2)**2))
                    df.loc[df['frames']==ff,f"{maskname}_speed"] = 0   
                    df[f"{maskname}_speed"] = np.round(df[f"{maskname}_dist"]/self.interval,decimals=2)
                    df[f"{maskname}_tot_length"] = df[f"{maskname}_dist"].sum()
                    df[f"{maskname}_Dp"] = np.round(df[f"{maskname}_dOE"]/df[f"{maskname}_tot_length"],decimals=2)
                    df[f"{maskname}_Dw"] = np.round((df[f"{maskname}_dOW"]-df[f"{maskname}_dEW"])/df[f"{maskname}_tot_length"],decimals=2)
                    df[f"{maskname}_tot_speed"] = np.round(df.loc[df['frames']>ff,f"{maskname}_speed"].mean(),decimals=2)
                    self.df_analysis.loc[self.df_analysis['cell']==cell] = df

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
        
        if 'pixel_distance' not in self.exp_prop['fct_inputs'] or pix_ana_ow or not exists(join(sep,self.exp_path+sep,'df_pixel.csv')):
            # Load dmap ref_mask and mask
            mask_ref = Analysis.apply_dmap(mask_stack=Analysis.ref_mask(imgFold_path=join(sep,self.exp_path+sep,imgFold),maskLabel=maskLabel,ref_mask_ow=ref_mask_ow),
                                                    frames=self.frames)      
            # Load mask
            if channel_seg:
                chan_seg = channel_seg
            else:
                chan_seg = self.channel_seg
            mask_stack = Analysis.load_mask(maskFold_path=join(sep,self.exp_path+sep,maskFold),channel_seg=chan_seg)
            if self.frames==1: mask_stack = [mask_stack] # just to be able to run it in the for loop

            # Extract data
            self.df_pixel = pd.DataFrame()
            for f,t in enumerate(self.ts):
                d = {}
                d['dmap'] = np.round(mask_ref[f,...][mask_stack[f].astype(bool)]*self.pixSize).astype(int)
                for chan in self.channel_list:
                    img_stack = Analysis.load_stack(imgFold_path=join(sep,self.exp_path+sep,imgFold),channel_list=chan,input_range=[f])
                    if self.z_size>1: img_stack = np.amax(a=img_stack,axis=0)
                    d[chan] = img_stack[mask_stack[f].astype(bool)].astype(float)
                df = pd.DataFrame.from_dict(d)
                df['time'] = t
                df['frame'] = f+1
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


            
            
            
            
            
            # # Average baseline and stimulation
            # df.loc[df['time']<=stim_time,'baseline_avg'] = df.loc[df['time']<=stim_time,col_name].mean()
            # df.loc[df['time']>stim_time,'stim_avg'] = df.loc[df['time']>stim_time,col_name].mean()
            




    
    
