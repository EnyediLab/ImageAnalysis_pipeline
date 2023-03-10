import BaxTrack as BTP # FIXME: remove MATLAB from here!!!
from utility import Utility
from experiments import Exp_Indiv
import pandas as pd
from analysis import Analysis
from os.path import join,exists
from os.path import isdir
from os import sep,walk
import matplotlib.pyplot as plt

# TODO: implement Docker
# TODO: write all the docstrings
# TODO: make all the notebook template
# TODO: implement parallel processing
class Experiments(Utility):
    """Class that carries all the necessary variable to run all the pipelines.
    It also convert all selected 2D files into image sequences. Background substraction 
    and image registration can also be applied to the images."""

    def __init__(self,parent_folder,channel_list,channel_seg,file_type='.nd2'): # TODO: enable tif file
        """
        Class that saves all the necessary variables to run the pipeline.

        Args:
            - parent_folder (str): Path of a folder which contain all the image files to be analysied. It can contain many level of subfolders. It will save the path of all the files of the chosen file_type.
            - channel_list ([str]): Name of the different channel (of your choice) that will be use to label the image of each channel. The order of the channels depends on the order of channel used during acquisition. For instance, if acquisition was made green channel first then red channel, then channel_list = ['green','red'].
            - channel_seg (str): Name of the channel to be segmented
            - exclude_channel (str or [str]): Tag(s) of channel to be excluded from analysis
            - file_type (str): Extention of the files to analyse.
        """
        # Check if path is valid
        if not isdir(parent_folder):
            raise ValueError(f"{parent_folder} is not a correct path. Try a full path")
      
        self.parent_folder = parent_folder
        self.file_type = file_type
       
        # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
        self.exp_dict = {}
        self.imgS_path = []
        for root , subs, files in walk(self.parent_folder):
            for f in files:
                if f.endswith(self.file_type):
                    self.imgS_path.append(join(sep,root+sep,f))
                if f.__contains__('exp_properties.pickle'):
                    exp_prop = Utility.open_exp_prop(exp_path=root)
                    self.exp_dict[root] = exp_prop
        self.imgS_path.sort()
        self.channel_list = channel_list
        
        # Channel check
        if channel_seg not in self.channel_list:
            raise ValueError(f"'{channel_seg}' is not in your channel list: {self.channel_list}")
        else:
            self.channel_seg = channel_seg

    def pre_process_all(self,bg_sub='Auto',imseq_ow=False,true_channel_list=None,reg=False,**kwargs):
        # Unpack all kwargs
        defSMO = {'sigma':0.0,'size':7} # Default kwargs for Auto bgsub methods
        defReg = {'reg_mtd':'translation','reg_ref':'mean','reg_ow':False,'reg_channel':self.channel_seg} # Default kwargs for image registration
        if kwargs: # Replace default val with input args
            for k,v in kwargs.items():
                if k in defReg:
                    defReg[k] = v
                elif k in defSMO:
                    defSMO[k] = v
                else:
                    raise AttributeError(f"kwargs '{k}' is not valid. Only accepted entries are:\n\t- for background substraction: {list(defSMO.keys())}\n\t- for image registration: {list(defReg.keys())}")
        if imseq_ow:
            defReg['reg_ow'] = True
        
        # Log
        print(f"Opening all '{self.file_type}' files:\n")
        
        # Create image seqs
        for img_path in self.imgS_path:
            self.exp_dict.update(self.create_imseq(img_path=img_path,imseq_ow=imseq_ow,channel_list=self.channel_list,file_type=self.file_type,true_channel_list=true_channel_list))
        
        # Add class para to exp_para and Get all the path for ACTIVE experiments
        self.exp_folder_path = []
        for k in self.exp_dict.keys():
            if self.exp_dict[k]['status']=='active':
                self.exp_folder_path.append(k)

        # Apply reg or bg_sub
        for exp_path in self.exp_folder_path:
            # Apply bg_sub
            if bg_sub:
                if bg_sub == 'Auto':
                    if self.exp_dict[exp_path]['img_preProcess']['bg_sub']=='Auto':
                        print(f"--> 'Auto' background substraction already applied on {exp_path} with: sigma={self.exp_dict[exp_path]['fct_inputs']['smo_bg_sub']['smo_sigma']} and size={self.exp_dict[exp_path]['fct_inputs']['smo_bg_sub']['smo_size']}")
                    else:
                        self.exp_dict[exp_path] = Experiments.smo_bg_sub(imgFold_path=join(sep,exp_path+sep,'Images'),**defSMO)
                
                elif bg_sub == 'Manual':
                    if self.exp_dict[exp_path]['bg_sub']=='Manual':
                        print(f"--> 'Manual' background substraction already applied on {exp_path}")
                    else:
                        self.exp_dict[exp_path] = Experiments.man_bg_sub(imgFold_path=join(sep,exp_path+sep,'Images'))
            # Apply reg
            if reg:
                self.exp_dict[exp_path] = Experiments.im_reg(imgFold_path=join(sep,exp_path+sep,'Images'),**defReg)
    
    def remove_exp(self,exp_path):
        open(join(sep,exp_path+sep,'REMOVED_EXP.txt'),'w')
        self.exp_dict[exp_path]['status'] = 'REMOVED'
    
    def exp_get_chanNpath(self,channel_seg=None,exp_path=None):
        # Get channel
        if channel_seg:
            chan_seg = channel_seg
        else:
            chan_seg = self.channel_seg

        # Setup paths
        if exp_path:
            # Convert to list
            if type(exp_path)==str:
                exp_path = [exp_path]
            if type(exp_path)!=list:
                raise TypeError(f"Only string or list of strings is allowed")

            # change var
            exp_folder_path = exp_path
        else:
            exp_folder_path = self.exp_folder_path
        return chan_seg, exp_folder_path

    def exp_cp_seg(self,imgFold='Images',exp_path=None,channel_seg=None,seg_ow=False,nucMarker=None,stitch=None,do_log=True,**kwargs):
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)
            
        # run segmentation
        if not hasattr(self,'exps'):
            self.exps = []

            # Run seg  
            for path in exp_folder_path:
                # Seg
                exp = Exp_Indiv(exp_path=path,channel_seg=self.channel_seg)
                exp.cellpose_segment(imgFold=imgFold,channel_seg=chan_seg,seg_ow=seg_ow,nucMarker=nucMarker,stitch=stitch,do_log=do_log,**kwargs)
                
                # Update list and exp_dict
                self.exps.append(exp)
                self.exp_dict[path] = exp.exp_prop
        else:
            exp_lst = [exp.exp_path for exp in self.exps]

            # Run seg  
            for path in exp_folder_path:
                if path in exp_lst:
                    # Seg
                    self.exps[exp_lst.index(path)].cellpose_segment(imgFold=imgFold,channel_seg=chan_seg,seg_ow=seg_ow,nucMarker=nucMarker,stitch=stitch,**kwargs)
                    
                    # Update exp_dict
                    self.exp_dict[path] = self.exps[exp_lst.index(path)].exp_prop
                else:
                    # Create obj and Seg
                    exp = Exp_Indiv(exp_path=path,channel_seg=self.channel_seg)
                    exp.cellpose_segment(imgFold=imgFold,channel_seg=chan_seg,seg_ow=seg_ow,nucMarker=nucMarker,stitch=stitch,**kwargs)

                    # Update list and exp_dict
                    self.exps.append(exp)
                    self.exp_dict[path] = exp.exp_prop
        
    def exp_bax_track(self,imgFold=None,maskFold='Masks_CP',exp_path=None,channel_seg=None,trim=False,track_ow=False,**kwargs):
        # Initiates MATLAB runtime
        if not hasattr(self,'track'):
            self.track = BTP.initialize()
        
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)
        
        # List of paths
        exp_lst = [exp.exp_path for exp in self.exps]

        # Run track  
        for path in exp_folder_path:
            # track
            self.exps[exp_lst.index(path)].baxter_tracking(objtrack=self.track,imgFold=imgFold,maskFold=maskFold,channel_seg=chan_seg,trim=trim,track_ow=track_ow,**kwargs)

            # Update exp_dict
            self.exp_dict[path] = self.exps[exp_lst.index(path)].exp_prop
        
    def exp_track_cells(self,maskFold='Masks_CP',exp_path=None,channel_seg=None,stitch_threshold=0.25,shape_threshold=0.2,stitch_ow=False):
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)
        
        # List of paths
        exp_lst = [exp.exp_path for exp in self.exps]
        
        # Run stitch
        for path in exp_folder_path:
            # Stitch
            self.exps[exp_lst.index(path)].stitch_masks(stitch_threshold=stitch_threshold,channel_seg=chan_seg,stitch_ow=stitch_ow,maskFold=maskFold,shape_threshold=shape_threshold)

            # Update exp_dict
            self.exp_dict[path] = self.exps[exp_lst.index(path)].exp_prop
    
    def exp_compart(self,maskFold='Masks_Trimmed',channel_seg=None,exp_path=None,rad_ero=10,rad_dil=None,compart_ow=False,dil_iteration=1,ero_iteration=1):
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)
        
        # List of paths
        exp_lst = [exp.exp_path for exp in self.exps]

        # Run track  
        for path in exp_folder_path:
            self.exps[exp_lst.index(path)].cell_compart(maskFold=maskFold,channel_seg=chan_seg,rad_ero=rad_ero,rad_dil=rad_dil,compart_ow=compart_ow,dil_iteration=dil_iteration,ero_iteration=ero_iteration)
            
            # Update exp_dict
            self.exp_dict[path] = self.exps[exp_lst.index(path)].exp_prop
    
    def exp_class(self,primary_channel,secondary_channel,exp_path=None,maskFold='Masks_Trimmed',rad_ero=10,class_ow=False,**kwargs):
        # Get channel and path
        __,exp_folder_path = self.exp_get_chanNpath(exp_path=exp_path)
        
        # List of paths
        exp_lst = [exp.exp_path for exp in self.exps]

        # Run track  
        for path in exp_folder_path:
            if secondary_channel not in self.exps[exp_lst.index(path)].exp_prop['channel_seg'][maskFold]:
                raise AttributeError(f"The secondary channel {secondary_channel} also need to be process. Please run segmentation+tracking")
                
            self.exps[exp_lst.index(path)].cell_class(maskFold=maskFold,primary_channel=primary_channel,secondary_channel=secondary_channel,rad_ero=rad_ero,class_ow=class_ow,**kwargs)
            # Update exp_dict
            self.exp_dict[path] = self.exps[exp_lst.index(path)].exp_prop

    def exp_analysis(self,imgFold,maskFold,channel_seg=None,exp_path=None,df_ow=False,do_cell_dist=False,maskLabel='wound'):
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)
        
        # Run analysis
        if not hasattr(self,'exps_analysis'):
            self.exps_analysis = []

            # Extract data
            self.masterdf_analysis = pd.DataFrame()
            for path in exp_folder_path:
                anal_exp = Analysis(exp_path=path,channel_seg=self.channel_seg)
                anal_exp.extract_channelData(imgFold=imgFold,maskFold=maskFold,channel_seg=chan_seg,df_ow=df_ow)
                anal_exp.extract_centroids(maskFold=maskFold,channel_seg=chan_seg,df_ow=df_ow)
                if do_cell_dist:
                    # Run ref_mask
                    anal_exp.cell_distance(imgFold=imgFold,df_ow=df_ow,maskLabel=maskLabel)
                # Update exp_prop and concat masterdf
                self.exp_dict[path] = anal_exp.exp_prop
                self.masterdf_analysis = pd.concat([self.masterdf_analysis,anal_exp.df_analysis])
                self.exps_analysis.append(anal_exp)
        else:
            exp_lst = [exp.exp_path for exp in self.exps_analysis]

            # Extract data
            self.masterdf_analysis = pd.DataFrame()
            for path in exp_folder_path:
                if path in exp_lst:
                    self.exps_analysis[exp_lst.index(path)].extract_channelData(imgFold=imgFold,maskFold=maskFold,channel_seg=chan_seg,df_ow=df_ow)
                    self.exps_analysis[exp_lst.index(path)].extract_centroids(maskFold=maskFold,channel_seg=chan_seg,df_ow=df_ow)
                    if do_cell_dist:
                        self.exps_analysis[exp_lst.index(path)].cell_distance(imgFold=imgFold,df_ow=df_ow,maskLabel=maskLabel)
                    # Update exp_prop and concat masterdf
                    self.exp_dict[path] = self.exps_analysis[exp_lst.index(path)].exp_prop
                    self.masterdf_analysis = pd.concat([self.masterdf_analysis,anal_exp.df_analysis])
                else:
                    anal_exp = Analysis(exp_path=path,channel_seg=self.channel_seg)
                    anal_exp.extract_channelData(imgFold=imgFold,maskFold=maskFold,channel_seg=chan_seg,df_ow=df_ow)
                    anal_exp.extract_centroids(maskFold=maskFold,channel_seg=chan_seg,df_ow=df_ow)
                    if do_cell_dist:
                        # Run ref_mask
                        anal_exp.cell_distance(imgFold=imgFold,df_ow=df_ow,maskLabel=maskLabel)
                    # Update exp_prop and concat masterdf
                    self.exp_dict[path] = anal_exp.exp_prop
                    self.masterdf_analysis = pd.concat([self.masterdf_analysis,anal_exp.df_analysis])
                    self.exps_analysis.append(anal_exp)
        
        # Save df as csv to parent folder
        self.masterdf_analysis.to_csv(join(sep,self.parent_folder+sep,'masterdf_analysis.csv'),index=False)

    def exp_man_mask(self,exp_path=None,channel_seg=None,csv_name='ManualTracking',radius=10,morph=True,n_mask=2,mantrack_ow=False):
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)
        
        # Run man_mask
        if not hasattr(self,'exps'):
            self.exps = []
            for path in exp_folder_path:
                # Seg
                exp = Exp_Indiv(exp_path=path,channel_seg=self.channel_seg)
                exp.man_mask(channel_seg=chan_seg, csv_name=csv_name, radius=radius,morph=morph, n_mask=n_mask ,mantrack_ow=mantrack_ow)
                # Update exp_dict
                self.exps.append(exp)
                self.exp_dict[path] = exp.exp_prop
        else:
            # List of paths
            exp_lst = [exp.exp_path for exp in self.exps]
        
            # Run stitch
            for path in exp_folder_path:
                if path in exp_lst:
                    # Stitch
                    self.exps[exp_lst.index(path)].man_mask(channel_seg=chan_seg, csv_name=csv_name, radius=radius, morph=morph, n_mask=n_mask, mantrack_ow=mantrack_ow)

                    # Update exp_dict
                    self.exp_dict[path] = self.exps[exp_lst.index(path)].exp_prop
                else:
                    # Seg
                    exp = Exp_Indiv(exp_path=path,channel_seg=self.channel_seg)
                    exp.man_mask(channel_seg=chan_seg, csv_name=csv_name, radius=radius, mantrack_ow=mantrack_ow)
                    # Update exp_dict
                    self.exps.append(exp)
                    self.exp_dict[path] = exp.exp_prop

    def exp_threshold_seg(self,imgFold='Images',do_blur=True,man_threshold=None,channel_seg=None,exp_path=None,thres_ow=False,**kwargs):
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)

        # run segmentation
        if not hasattr(self,'exps'):
            self.exps = []

            # Run seg  
            for path in exp_folder_path:
                # Seg
                exp = Exp_Indiv(exp_path=path,channel_seg=self.channel_seg)
                exp.threshold_seg(imgFold=imgFold,
                                thres=man_threshold,
                                blur=do_blur,
                                channel_seg=chan_seg,
                                thres_ow=thres_ow,
                                **kwargs)
                # Update list and exp_dict
                self.exps.append(exp)
                self.exp_dict[path] = exp.exp_prop
        else:
            exp_lst = [exp.exp_path for exp in self.exps]

            # Run seg  
            for path in exp_folder_path:
                if path in exp_lst:
                    # Seg
                    self.exps[exp_lst.index(path)].threshold_seg(imgFold=imgFold,
                                                                thres=man_threshold,
                                                                blur=do_blur,
                                                                channel_seg=chan_seg,
                                                                thres_ow=thres_ow,
                                                                **kwargs)
                    # Update exp_dict
                    self.exp_dict[path] = self.exps[exp_lst.index(path)].exp_prop
                else:
                    # Create obj and Seg
                    exp = Exp_Indiv(exp_path=path,channel_seg=self.channel_seg)
                    exp.threshold_seg(imgFold=imgFold,
                                    thres=man_threshold,
                                    blur=do_blur,
                                    channel_seg=chan_seg,
                                    thres_ow=thres_ow,
                                    **kwargs)
                    # Update list and exp_dict
                    self.exps.append(exp)
                    self.exp_dict[path] = exp.exp_prop

    def exp_pixel_distance(self,imgFold,maskFold,exp_path=None,channel_seg=None,maskLabel='wound',do_cond_df=False,pix_ana_ow=False): # TODO: modify function to load masterdf if any
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)

        # run pixel_distance
        if exists(join(sep,self.parent_folder+sep,'masterdf_pixel.csv')) and not pix_ana_ow:
            # Load masterdf
            self.masterdf_pixel = pd.read_csv(join(sep,self.parent_folder+sep,'masterdf_pixel.csv'))
            # Recall all experiment
            self.exps_analysis = []
            for path in exp_folder_path:
                # Seg
                exp = Analysis(exp_path=path,channel_seg=self.channel_seg)
                exp.pixel_distance(imgFold=imgFold,
                                channel_seg=chan_seg,
                                maskFold=maskFold,
                                maskLabel=maskLabel,
                                pix_ana_ow=pix_ana_ow,
                                )
                # Update list and exp_dict
                self.exps_analysis.append(exp)
            # Log
            print('Analysis were already processed for all experiments')        
        else:
            # Extract data
            self.masterdf_pixel = pd.DataFrame()
            self.exps_analysis = []
            for path in exp_folder_path:
                # Seg
                exp = Analysis(exp_path=path,channel_seg=self.channel_seg)
                exp.pixel_distance(imgFold=imgFold,
                                channel_seg=chan_seg,
                                maskFold=maskFold,
                                maskLabel=maskLabel,
                                pix_ana_ow=pix_ana_ow,
                                )
                # Update list and exp_dict
                self.exps_analysis.append(exp)
                self.exp_dict[path] = exp.exp_prop
                self.masterdf_pixel = pd.concat([self.masterdf_pixel,exp.df_pixel])
            # Save masterdf
            self.masterdf_pixel.to_csv(join(sep,self.parent_folder+sep,'masterdf_pixel.csv'),index=False)

        # Create conddf
        if do_cond_df:
            for tag in self.masterdf_pixel['tag'].unique():
                # Extract df
                conddf_pixel = self.masterdf_pixel.loc[self.masterdf_pixel['tag']==tag,:].copy()
                # Save conddf
                conddf_pixel.to_csv(join(sep,self.parent_folder+sep,tag+sep,'conddf_pixel.csv'),index=False)

    def exp_plot_HM(self,col_name,deltaF,maxdt=None,intBin=5,col_lim=[0,2],exp_path=None,savedir=None,row_col=None,figsize=None,cbar_label=r'$\Delta$F/F$_{min}$',**kwargs):
        # Get channel and path
        __, exp_folder_path = self.exp_get_chanNpath(exp_path=exp_path)

        # Create the subplot
        if row_col: nrow,ncol = row_col
        else:
            nrow = round(len(exp_folder_path)/4)
            if len(exp_folder_path)<4: ncol = len(exp_folder_path)
            else: ncol = 4
        fig,ax = plt.subplots(nrow,ncol,sharey=True,figsize=figsize)

        # upper limit?
        if maxdt:
            if isinstance(maxdt,float) or type(maxdt)==int: maxdmap = maxdt
            elif isinstance(maxdt,bool): 
                max_lst = [pd.read_csv(join(sep,path+sep,'df_pixel.csv')).dmap.max() for path in exp_folder_path]
                maxdmap = min(max_lst)
        
        # Recall all experiment
        self.df_pixel_lst = []
        r = 0; c = 0 # Initialise the axes
        for path in exp_folder_path:
            # Create tag and exp name
            split_path = path.split(sep)[-1].split('_')
            exp_name = '_'.join([path.split(sep)[-2],split_path[0],split_path[-1]])
            # Load df
            df = pd.read_csv(join(sep,path+sep,'df_pixel.csv'))
            self.df_pixel_lst.append(df)
            # Bin it and plot it
            if maxdt: bin_df = Experiments.pixel_bin(df_pixel=df.loc[df['dmap']<=maxdmap,:].copy(),intBin=intBin,col_name=col_name,deltaF=deltaF)
            else: bin_df = Experiments.pixel_bin(df_pixel=df,intBin=intBin,col_name=col_name,deltaF=deltaF)
            Experiments.plot_HM(bin_df,title=exp_name,axes=ax[r,c],savedir=savedir,cbar_label=cbar_label,col_lim=col_lim,**kwargs)
            # Adjust the axes
            c += 1
            if c==ncol: c = 0; r+=1
    
    # TODO: def add_exp_para_var()
    
    # TODO: def read_exp_para()


    # def add_exp(self,exp_path,exp_para=None,bg_sub=None,reg=False,**kwargs):
    #     # Unpack all kwargs
    #     defSMO = {'sigma':0.0,'size':7} # Default kwargs for Auto bgsub methods
    #     defReg = {'reg_mtd':'translation','reg_ref':'mean','reg_ow':False,'reg_channel':self.channel_seg} # Default kwargs for image registration
    #     if kwargs: # Replace default val with input args
    #         for k,v in kwargs.items():
    #             if k in defReg:
    #                 defReg[k] = v
    #             elif k in defSMO:
    #                 defSMO[k] = v
    #             else:
    #                 raise AttributeError(f"kwargs '{k}' is not valid. Only accepted entries are:\n\t- for background substraction: {list(defSMO.keys())}\n\t- for image registration: {list(defReg.keys())}")
        
    #     # Prepare settings for image seq
    #     if not hasattr(self,'exp_dict'):
    #         self.exp_dict = {}
        
    #     # Load exp_para
    #     print(f"Adding {exp_path} experiment:\n")
    #     if exists(join(sep,exp_path+'exp_settings.pickle')):
    #         with open(join(sep,exp_path+'exp_settings.pickle'),'rb') as pickfile:
    #                 exp_para = pickle.load(pickfile)
    #                 self.exp_dict[exp_path] = exp_para
    #     else:

        
    #     # Create image seqs
        
        
    #     # Add class para to exp_para and Get all the path for ACTIVE experiments
    #     self.exp_folder_path = []
    #     for k in self.exp_dict.keys():
    #         self.exp_dict[k]['channel_list'] = self.channel_list
    #         self.exp_dict[k]['channel_seg'] = self.channel_seg
    #         if self.exp_dict[k]['status']=='active':
    #             self.exp_folder_path.append(k)

    #     # Apply reg or bg_sub
    #     for exp_path in self.exp_folder_path:
    #         # Apply bg_sub
    #         if bg_sub:
    #             if bg_sub == 'Auto':
    #                 if self.exp_dict[exp_path]['bg_sub']=='Auto':
    #                     print(f"--> 'Auto' background substraction already applied on {exp_path} with: sigma={self.exp_dict[exp_path]['smo_sigma']} and size={self.exp_dict[exp_path]['smo_size']}")
    #                 else:
    #                     self.smo_bg_sub(imgFold_path=join(sep,exp_path+sep,'Images'),exp_para=self.exp_dict[exp_path],channel_list=self.channel_list,**defSMO)
                
    #             elif bg_sub == 'Manual':
    #                 if self.exp_dict[exp_path]['bg_sub']=='Manual':
    #                     print(f"--> 'Manual' background substraction already applied on {exp_path}")
    #                 else:
    #                     self.man_bg_sub(imgFold_path=join(sep,exp_path+sep,'Images'),exp_para=self.exp_dict[exp_path],channel_list=self.channel_list)
    #         # Apply reg
    #         if reg:
    #             self.exp_dict[exp_path]['reg'] = self.im_reg(imgFold_path=join(sep,exp_path+sep,'Images'),exp_para=self.exp_dict[exp_path],channel_list=self.channel_list,**defReg)
   
################################################################



# def initiate_config(parent_folder,channel_list,channel_seg,file_type=".nd2",
#                     bg_method='Auto',exclude_channel=None,imseq_ow=False,do_reg=False,**kwargs):
#     """Function that initiate the Config class. It will contain all necessary variables to run the selected pipeline.
#     It will convert all 2D files (atm) of the selected file_type in the parent_folder and subsequent folders into
#     an image sequence. It can also apply a background substraction and register the images.
    
#     If there is more than one conditions, it is recommended to store each condition in subsequent folders, as it will extract 
#     the name of those folders and store it as a tag for each experiment.
    
#     If selected, background substraction will be apply, using the method:
#         - Manual: manually select an area on the image that will then be substracted to all valid channels.
#         - Auto: perform automatic bg sub, using the SMO algorithm (https://github.com/maurosilber/SMO). It requires additional args 'sigma' and 'size', see kwargs.
    
#     If selected, image registration will be apply, using pystackreg (https://pypi.org/project/pystackreg/) with additional args 'reg_mtd', 'reg_ref' and 'reg_ow', see kwargs.
#         - Methods:
#             - translation
#             - rigid body (translation + rotation)
#             - scaled rotation (translation + rotation + scaling)
#             - affine (translation + rotation + scaling + shearing)
#             - bilinear (non-linear transformation; does not preserve straight lines)
#         - Reference image:
#             - previous
#             - first
#             - mean

#     Args:
#         - parent_folder (str): Path of a folder which contain all the image files to be analysied. It can contain many level of subfolders. It will save the path of all the files of the chosen file_type.
#         - channel_list ([str]): Name of the different channel (of your choice) that will be use to label the image of each channel. The order of the channels depends on the order of channel used during acquisition. For instance, if acquisition was made green channel first then red channel, then channel_list = ['green','red'].
#         - channel_seg (str): Name of the channel to be segmented
#         - file_type (str, optional): Extention of the files to analyse. Defaults to ".nd2"
#         - bg_method (str, optional): (str): Define bg sub method, from 'Auto', 'Manual' or None. Defaults to 'Auto'.
#         - exclude_channel (str or [str], optional): Name of the channel to exclude from bg sub (e.g. bright field). Defaults to None.
#         - ow (bool, optional): Overwrite existing images. Defaults to False.
#         - do_reg (bool, optional): Apply image registration. Defaults to False.
#         - kwargs (any, optional):
#             - sigma (float, optional): Standard deviation for the Gaussian kernel. Defaults to 0.
#             - size (int, optional): Size of the averaging kernel. Should be smaller than foreground. Defaults to 7.
#             - reg_mtd (str, optional): Registration method to apply, from 'translation','rigid_body','scaled_rotation','affine' or 'bilinear'. Defaults to 'translation'.
#             - reg_ref (str, optional): Image reference for the transformation, from 'previous', 'first' or 'mean'. Defaults to 'mean'.
#             - ow_reg (bool, optional): Overwrite existing registered images. Defaults to False.

#     Returns:
#         - config (obj): class Config object.

#     Note:
#         3D images will be soon implemented"""

#     config = Config(parent_folder=parent_folder,channel_list=channel_list,channel_seg=channel_seg,exclude_channel=exclude_channel,file_type=file_type)
#     config.create_imseq(bg_method=bg_method,imseq_ow=imseq_ow,reg=do_reg,**kwargs)
#     return config
