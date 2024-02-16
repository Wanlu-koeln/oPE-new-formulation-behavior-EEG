# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
# %% Cell 1: import libraries
import matplotlib
matplotlib.use('Qt5Agg')  # You can also try 'Qt4Agg' if 'Qt5Agg' doesn't work

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.stats import pearsonr
import random as rand
from os import listdir
import mne
mne.set_log_level('Critical')
import statsmodels.sandbox.stats.multicomp as stats
from mne.stats.regression import linear_regression
import copy
import pandas
import numpy as np
import mne
from scipy.spatial.distance import squareform, pdist
from scipy.sparse import csc_matrix
from mne.stats import spatio_temporal_cluster_1samp_test as stct
from mne.stats import linear_regression
from scipy.stats import mstats
import numpy as np
import matplotlib.pyplot as plt


# %%from mne.preprocessing import find_outliers, ICA, read_ica, create_ecg_epochs, create_eog_epochs

sourceloc = '/Users/wanlufu/Downloads/eeg_dat/'
files = [x for x in listdir(sourceloc) if "_postICA-epo.fif" in x]

for file in files:
    print((sourceloc + file)) 

len(files)
# %% average epochs
def do_grand(f):
    
    epochs = [mne.read_epochs(sourceloc + fi) for fi in f]
    
    evokeds = [epoch_parameters(epoch) for epoch in epochs]
    grand_average = mne.grand_average(evokeds)
    fig = grand_average.plot_joint()
    fig.suptitle('Grand average');#times = [.1,.17,.22,.32,.47]
    
# %% define epoch parameters   
def epoch_parameters(epoch):
    epoch.set_eeg_reference()#(ref_channels = ['Cz'])
    epoch.crop(-.2,.8)
    evoked = epoch.average()
    return evoked

# %%
mpl.rcParams['savefig.dpi'] = 100 
do_grand(files)

# %% import presictors
predictors_source = sourceloc + 'human_eeg_indi_characteristics_behavior.csv'

# %% add predicotr variables
predictors = pandas.read_csv(predictors_source) #predictors['stimcat'] = 0.25 #predictors['stimcat'][predictors.stimcode == 'w_hild'] = 0.75 #predictors['stimcat'][predictors.stimcode == 'cs'] = 0 #predictors['stimcat'][predictors.stimcode == 'pw'] = 1 predictors['log_vpe'] = np.round(np.log(predictors['vis_pred_err']),1) predictors['log_lum'] = np.round(np.log(predictors['lumi']),2) predictors['stand_vpe'] = np.round((predictors['vis_pred_err']-np.mean(predictors['vis_pred_err']))/np.std(predictors['vis_pred_err']),2) predictors['stand_lum'] = np.round((predictors['lumi']-np.mean(predictors['lumi']))/np.std(predictors['lumi']),2) predictors['iscs'] = 0 predictors['iscs'][predictors.stimcode == 'cs'] = 1 predictors['ispw'] = 0 predictors['ispw'][predictors.stimcode == 'pw'] = 1 predictors['isw'] = 1 predictors['isw'][predictors.stimcode == 'w_hild'] = 0 predictors['isw'][predictors.stimcode == 'w_lowld'] = 0 predictors['intercept'] = 1 predictors

predictors['log_vpe'] = np.round(np.log(predictors['vis_pred_err']),1)
predictors['log_lum'] = np.round(np.log(predictors['lumi']),2)

predictors['stand_vpe'] = np.round((predictors['vis_pred_err']-np.mean(predictors['vis_pred_err']))/np.std(predictors['vis_pred_err']),2)
predictors['stand_lum'] = np.round((predictors['lumi']-np.mean(predictors['lumi']))/np.std(predictors['lumi']),2)


predictors['iscs'] = 0
predictors['iscs'][predictors.stimcode == 'cs'] = 1

predictors['ispw'] = 0
predictors['ispw'][predictors.stimcode == 'pw'] = 1

predictors['isw'] = 1
predictors['isw'][predictors.stimcode == 'w_hild'] = 0
predictors['isw'][predictors.stimcode == 'w_lowld'] = 0

predictors['intercept'] = 1

predictors

# %% define predictor variables

predictors['lex_1'] = (predictors['lex']-.5)*2
predictors['concr_woNA'] = predictors['concr']
predictors['concr_woNA'][ pandas.isnull(predictors['concr_woNA'])] = 0#max(predictors['concr'])
predictors['lumXvpe'] = predictors['stand_vpe'] * predictors['stand_lum']
predictors['lexXvpe'] = predictors['stand_vpe'] * (predictors['lex_1'])
predictors['freqXvpe'] = predictors['stand_vpe'] * (predictors['lgfreqCount'])
predictors['freqXope'] = predictors['ope_norm'] * (predictors['lgfreqCount'])
predictors['lexXold'] = predictors['old'] * (predictors['lex_1'])
predictors['lexXlum'] = predictors['stand_lum'] * (predictors['lex_1'])
predictors['lexXfreq'] = predictors['stand_vpe'] * zscore(predictors['lgfreqCount'])
predictors['lexXope'] = predictors['ope_norm'] * (predictors['lex_1'])
predictors['lexXape'] = predictors['ape_norm'] * (predictors['lex_1'])
predictors['lexXspe'] = predictors['spe_norm'] * (predictors['lex_1'])



predictors

# %% design evoked
def make_design(epochs, variables, db_):
   
    predictors_ = db_ #if "wy2201_postICA-epo.fif" not in epochs.info["subject_info"] else db_[2:].copy()

    #print(len(predictors_))
    predictors = predictors_#.query("lex == 1")#.query("rand_select == 1")#.query("lex == 1")
    epochs_ = epochs[predictors.index] #if "wy2201_postICA-epo.fif" not in epochs.info._attributes["subject_info"] else epochs[predictors.index-2]
    n_trials = len(epochs_.events)
    
    design = np.zeros((n_trials, len(variables) + 1))
    
    design[:, 0] = 1
    
    for ii, key in enumerate(variables):
        design[:, ii + 1] = predictors[key]
    
    intercept = np.ones((len(epochs_),), dtype=np.float64)
    
    evoked = linear_regression(epochs_, design, ['intercept'] + variables)
    
    return evoked

# %% define evoked
def make_lm(f, factors):
  
    epochs = mne.read_epochs(sourceloc + f)
    epochs.drop_bad()
    epochs.set_eeg_reference(ref_channels =['Cz'])
    epochs.crop(-.2,.8)
    
    #predictors_ = predictors if "wy2201_postICA-epo.fif" not in epochs.info["subject_info"] else predictors[2:].copy()
    
    db_ = predictors#.query("lex == 1")
    epochs_ = epochs[db_.index] 
    epochs_.info._attributes["subject_info"] = f
    print(epochs_.info._attributes["subject_info"])
    
    evoked = make_design(epochs_, factors, db_)
 
    return evoked

# %% add predictors
variables = ['lex_1','ope_norm','lumi','lexXope']#'lgfreqCount','ope_norm','lumi','freqXope'
lm_tmp = make_lm(files[29], variables)
#lm_tmp["vis_pred_err"].beta.plot_joint(times = [.05,.1,.12,.14,.15,.2,.25,.3,.35,.4,.45])#times = [.39,.49]);
#lm_tmp["vis_pred_err"].t_val.plot_joint(times = [.05,.1,.12,.14,.15,.17,.2,.25,.3,.35,.4,.45])#times = [.1,.12,.14,.2,.22,.24]);

mpl.rcParams['savefig.dpi'] = 150

# %% add all participants
#,'lex_1','lexXvpe','freqXvpe','lgfreqCount''stand_vpe','stand_lum','simBOLDentro','cvp','bfreq','old','lgfreqCount','lex'
lms = [make_lm(f, variables) for f in files] 
i = 0
for i in range(len(lms)):
    print(files[i])
    print(i)
    #lms[i]["vis_pred_err"].beta.plot_joint(times = [.35,.4,.45,.5,.55,.6,.65]);#.05,.1,.15,.2,.25,.3,


mpl.rcParams['savefig.dpi'] = 100

#%% extract data for each participant at time point 0
# Assuming lms is a list of Evoked objects
participants_beta_at_time_0 = {}

# Iterate over participants
for i, evoked in enumerate(lms):
    participant = files[i]  # Assuming files contains participant information
    
    # Initialize a dictionary for each participant
    participants_beta_at_time_0[participant] = {}

    # Access the beta coefficients from the linear regression for each variable
    for variable in variables:
        # Access the beta coefficients at time point 0
        beta_at_time_0 = evoked[variable].beta

        # Store the beta coefficients in the participants_beta_at_time_0 dictionary
        participants_beta_at_time_0[participant][variable] = beta_at_time_0
        
#%%
import pandas as pd

# Assuming participants_beta_at_time_0 is a dictionary with participant IDs as keys
# and values as dictionaries containing the four variables,
# and inside each variable, there is an 'EvokedArray' object

# Initialize an empty dictionary to store the extracted data frames
extracted_data_frames = {}

# Iterate over participants in the dictionary
for participant_id, participant_data in participants_beta_at_time_0.items():
    
    # Extract the data for each variable at time point 0
    extracted_data = {}
    for variable_name, evoked_array in participant_data.items():
        # Get the data and ch_names from the EvokedArray at time point 0
        data_at_time_0 = evoked_array.data[:, 0]  # Assuming 0 corresponds to time point 0
        variable_data_per_channel = {
            'TimePoint0': data_at_time_0,
            'Channel': evoked_array.ch_names
        }
        extracted_data[variable_name] = variable_data_per_channel

    # Convert the extracted data into a data frame for each variable
    data_frames_per_variable = {
        variable_name: pd.DataFrame(extracted_data[variable_name]) for variable_name in extracted_data
    }

    # Store the data frames in the dictionary with the participant ID as the key
    extracted_data_frames[participant_id] = data_frames_per_variable

# Now, extracted_data_frames contains a dictionary for each participant ID
# Each dictionary contains data frames for each variable
# Each data frame has data at time point 0 and channel names as columns

#%%
import pandas as pd

# Initialize an empty dictionary to store the modified data frames
modified_data_frames = {}

# Iterate over participants in the dictionary
for participant_id, data_frames_per_variable in extracted_data_frames.items():
    # Initialize a dictionary to store modified data frames for each variable
    modified_data_frames_per_variable = {}
    
    # Iterate over variables
    for variable_name, data_frame in data_frames_per_variable.items():
        # Rename the 'TimePoint0' column based on participant ID
        data_frame = data_frame.rename(columns={'TimePoint0': participant_id[:6]})
        modified_data_frames_per_variable[variable_name] = data_frame
    
    # Store the modified data frames in the dictionary with the participant ID as the key
    modified_data_frames[participant_id] = modified_data_frames_per_variable

# Now, modified_data_frames contains a dictionary for each participant ID
# Each dictionary contains modified data frames for each variable
# Each modified data frame has channel data as rows, and the first 6 letters of the participant ID as columns
#%%
# Initialize an empty dictionary to store the merged data frames
merged_data_frames = {}

# Iterate over each variable
for variable_name in modified_data_frames['AD2809_postICA-epo.fif']:
    
    # Merge the data frames for each participant under each variable
    merged_df = pd.concat(
        [df[variable_name].drop(columns=['Participant', 'Channel']) for df in modified_data_frames.values()],
        axis=1
    )

    # Add the 'channel' column from the first participant's data frame
    merged_df['Channel'] = modified_data_frames['AD2809_postICA-epo.fif'][variable_name]['Channel']
    
    # Store the merged data frame in the dictionary with the variable name as the key
    merged_data_frames[variable_name] = merged_df

# Now, merged_data_frames contains a dictionary for each variable
# Each dictionary contains a merged data frame with only one 'channel' column
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'lex_1' is one of the variables in merged_data_frames
variable_name = 'ope_norm'

# Extract the data frame for the specific variable
variable_df = merged_data_frames[variable_name]

# Exclude the 'Channel' column
channels = variable_df['Channel'].unique()
# Increase the figure size for better readability
figsize = (10, 5)
for i, channel in enumerate(channels):
    # Create a new figure for each channel
    plt.figure(figsize=figsize)
    
    # Filter data for the specific channel
    channel_data = variable_df[variable_df['Channel'] == channel].drop(columns=['Channel'])
    
    # Plot box plot and strip plot
    sns.boxplot(data=channel_data, showfliers=False)
    sns.stripplot(data=channel_data, color='black', jitter=True, size=2)
    
    plt.title(f'Box Plot for {variable_name} - Channel: {channel}')
    plt.xlabel('Participant')
    plt.ylabel('Values')
    
    # Display participant information next to each data point
    for i, participant in enumerate(channel_data.index):
        # Display participant ID and corresponding column name
        for j, value in enumerate(channel_data.columns):
            plt.text(i + j * 0.2, channel_data.loc[participant, value], f'{participant}\n{value}', ha='center', va='bottom', fontsize=8, color='black')
        
    # Display x-axis labels next to each data point
    plt.xticks(np.arange(len(channel_data.index)), channel_data.index, rotation=45, ha='right', fontsize=8)
    
    # Save the figure with a unique name for each channel
    plt.savefig(f'box_plot_channel_{channel}.png')
    plt.show()
# %%
grands = {variable:mne.grand_average([lm_[variable].beta for lm_ in lms])
          for variable in variables}
# %%
def adapt_data_for_plj(grands,key,times,title):
    mpl.rcParams.update({'font.size': 13})
    #grands[key].data = grands[key].data /1e6
    grands[key].plot_joint(
        times = times[key], 
        title=title[key])#title=key)#
    #grands[key].data = grands[key].data *1e6


# %%

mpl.rcParams['savefig.dpi'] = 100

#topomap_args = {'vmin': -1.5e6,'vmax': 1.5e6}
#ts_args={"scalings":{"eeg":1}}
#topomap_args={"scalings":{"eeg":1}}
#ts_args = {'ylim':{"eeg":[-2e6, 2e6]}}

#for betas
ts_args = {'ylim':{"eeg":[-.6, .6]}}
topomap_args = {'vmin': -.5,'vmax': .5}
times = {
           'ope_norm': [.23], 
#           #'lgfreqCount':[.23,.34], 
#           'ape_norm': [.23], 
#           'spe_norm': [.23], 
           'lex_1': [.43], 
           'lumi': [.23,.34], 
#           #'freqXope':[.43]
           'lexXope': [.43]
#           'lexXape': [.43], 
#           'lexXspe': [.43]
         }
title = {'ope_norm': 'Effect size [$\mu$V] for Pixel-based prediction error covariate',
   #      'ape_norm': 'Effect size [$\mu$V] for Letter-based prediction error covariate',
  #       'spe_norm': 'Effect size [$\mu$V] for Sequence-based prediction error covariate',
         'lex_1': 'Effect size [$\mu$V] for lexicality covariate',
        #'lgfreqCount':'Effect size [$\mu$V] for word_frequency covariate',
         'lumi':'Effect size [$\mu$V] for pixel number covariate',
        # 'freqXope': 'Effect size [$\mu$V] for word_frequency by Pixel-based prediction error interaction'
         'lexXope': 'Effect size [$\mu$V] for lexicality by Pixel-based prediction error interaction'
 #        'lexXape': 'Effect size [$\mu$V] for lexicality by Letter-based prediction error interaction',
#         'lexXspe': 'Effect size [$\mu$V] for lexicality by Sequence-based prediction error interaction'
         }

for key in grands:
    adapt_data_for_plj(grands, key,times, title)
    #grands[key].plot_joint(title=key,times = times)#,times = times,ts_args=ts_args,topomap_args=topomap_args);#,ts_args=ts_args,topomap_args=topomap_args
    #plot_topomap(times = times,vmin=-.2,vmax=.2)
    #plot_topomap(times = times,vmin=-1.3,vmax=1.3)
    #plot_joint(title=key,ts_args=ts_args,times = times);#,topomap_args=topomap_args);

# %% 

def indi_evok_per_cond(lm):
    #in_grands = {}
    in_lm = {}
    for key in variables:
        in_lm[key] = lm[key].beta
        #in_grands[cond] = mne.grand_average(all_evokeds[cond])

    return in_lm

# %%

#rerps (bei dir wäre ein anderer name besser ...): eine liste von dictionaries, 
#bei denen die keys conditions sind und die values erps; ein dictionary pro VP


rerps = [indi_evok_per_cond(lm_) for lm_ in lms]

# %% 
erp = grands[variables[0]].copy()
from mne.channels.layout import _find_topomap_coords
xy = np.asarray(_find_topomap_coords(erp.info, range(len(erp.ch_names))))

dsts = squareform(pdist(xy))

def check_connectivities(name):
    idx = np.where(dsts[erp.ch_names.index(name), :] < 0.575)
    for ch, ii in zip(erp.ch_names, sum(dsts < 0.575)):
        print(ch + ": " + str(ii))

connectivity = dsts < 0.5  # Adjust the threshold as needed
connectivity = csc_matrix(connectivity)

# Define the function to create the connectivity matrix
# def create_connectivity_matrix(erp):
#     xy = np.asarray(mne.channels.layout._find_topomap_coords(erp.info, range(len(erp.ch_names))))
#     dsts = squareform(pdist(xy))
#     connectivity = dsts < 0.5  # Adjust the threshold as needed
#     return csc_matrix(connectivity)

# Update the function to perform the cluster test
def make_clustest(mmn_d, alpha=.05):
    T_obs, clusters, cluster_pv, H0 = stct(mmn_d, n_jobs=12, out_type='mask', t_power=4)
    inds = (np.mean([c for c, cp in zip(clusters, cluster_pv) if cp < alpha], axis=0) > 0)
    return inds

# Inside the loop where you use the functions
#erp = grands[variables[0]].copy()
#connectivity = create_connectivity_matrix(erp)

data_dict = {cond:np.asarray([r_[cond].data.T for r_ in rerps]) for cond in grands}
ind_dict = {cond: make_clustest(erp) for cond, erp in data_dict.items()}

erp_dict = {}
for c in ind_dict:
    erp_dict[c] = grands[c].copy()
    erp_dict[c].data = data_dict[c].mean(0).T

alphabet = "abcdefghijklmnopqrstuvwxyz"
def check_conds(n):
    try:
        i_ = int(n[-1].lower())
        if i_ != 0 and i_ < 3:
            return "center"
        elif bool(i_ % 2):
            return "right"
        else:
            return "left"
    except ValueError:
        return "center"
   
# %% 
def plot_masked_image(erp, inds=False):
    d = erp.copy().data.T
    if inds is not False:
        d *= inds
        d[d == 0.0] = np.nan
        d[d == -0.0] = np.nan
    from mne.channels.layout import _find_topomap_coords
    xy = np.asarray(_find_topomap_coords(erp.info,
                                        range(len(erp.ch_names))))    

    f, axes = plt.subplots(3, 1)
    positions = ("left", "center", "right")

    max_ = 6e-07#np.nanmax(np.abs(d)) *1.5e-07
    print(max_)
    #if max_ > 2:
     #    max_ = max_ *1e-07
    #if max_ > .1:
     #    max_ = max_ *1e-06
    
    
    #print(max_)

    for ax, where in zip(axes, positions):
        ch_is = [ii for ii, ch in enumerate(erp.ch_names)
                 if check_conds(ch) == where]
        x = xy[:,1].copy()
        x += (xy[:,0] / 100)
        x[ch_is] += 100
        ch_is = x.argsort()[::-1][:len(ch_is)][::-1]
#         print(d.shape)
#         print(d[:, ch_is].shape)
        im = ax.imshow(d[:, ch_is].T, origin="lower", cmap="RdBu_r",
                  extent=(erp.times[0], erp.times[-1], 0, len(ch_is)),
                  aspect='auto', vmin=-max_, vmax=max_,
                  interpolation='nearest')
        s = "{}\n\n{}".format(("Channels\n< back <   > front >"
                               if where == "center" else ""),
                               where)
        ax.set_ylabel(s)
        ax.set_yticks(())
        
        times_xax = ['-200','0','200','400','600','800']
        
        if where != "right":
            ax.set_xticks(())
            
        else:
            ax.set_xlabel('Time [ms]')#(r"Event related potentials [$\mu$V]")#
            ax.set_xticklabels(labels=times_xax)
            #plt.colorbar(im)
#         print([erp.ch_names[ii] for ii in ch_is])

    s = ",\nmasked for significance" if inds is not False else ""
    plt.suptitle("Channel-wise evoked activity{}".format(s))
    
    # if inds is not False:
    #     significant_times = erp.times[inds.any(axis=0)]
    #     print("Time points with significant clusters:", significant_times)

    # plt.show()  # Added to display the plot
    
# %% 

mpl.rcParams['savefig.dpi'] = 100

for cond in erp_dict:
   f = plt.figure()
   plot_masked_image(erp_dict[cond], 
                     ind_dict[cond]
                    )
   plt.suptitle(cond)
   
    # Print time points where significant clusters are present for each condition
   # significant_times = erp_dict[cond].times[ind_dict[cond].any(axis=0)]
   # print("Time points with significant clusters for condition '{}':".format(cond), significant_times)

   # # Optionally, save the figure
   # plt.savefig("figure_{}.png".format(cond))  # Adjust the filename as needed

   # plt.show()  # Added to display the plot
   
# %%
alphabet = "abcdefghijklmnopqrstuvwxyz"
def check_conds(n):
    try:
        i_ = int(n[-1].lower())
        if i_ != 0 and i_ < 3:
            return "center"
        elif bool(i_ % 2):
            return "right"
        else:
            return "left"
    except ValueError:
        return "center"

def plot_masked_image_fdr(erp, inds=False):
    d = erp.copy().data.T
    if inds is not False:
        d *= inds
        d[d == 0.0] = np.nan
        d[d == -0.0] = np.nan
    from mne.channels.layout import _find_topomap_coords
    xy = np.asarray(_find_topomap_coords(erp.info,
                                        range(len(erp.ch_names))))    

    f, axes = plt.subplots(3, 1)
    positions = ("left", "center", "right")

    max_ = np.nanmax(np.abs(d)) #*1e-07
    print(max_)
    #if max_ > 2:
     #    max_ = max_ *1e-07
    #if max_ > .1:
     #    max_ = max_ *1e-06
    
    
    #print(max_)

    for ax, where in zip(axes, positions):
        ch_is = [ii for ii, ch in enumerate(erp.ch_names)
                 if check_conds(ch) == where]
        x = xy[:,1].copy()
        x += (xy[:,0] / 100)
        x[ch_is] += 100
        ch_is = x.argsort()[::-1][:len(ch_is)][::-1]
#         print(d.shape)
#         print(d[:, ch_is].shape)
        print(d.shape, erp.times.shape)
        im = ax.imshow(d[:, ch_is].T, origin="lower", cmap="RdBu_r",
                  extent=(erp.times[0], erp.times[-1], 0, len(ch_is)),
                  aspect='auto', vmin=-max_, vmax=max_,
                  interpolation='nearest')
        s = "{}\n\n{}".format(("Channels\n< back <   > front >"
                               if where == "center" else ""),
                               where)
        ax.set_ylabel(s)
        ax.set_yticks(())
        
        times_xax = np.linspace(erp.times[0], erp.times[-1], 5)
        
        if where != "right":
            ax.set_xticks(())
            
        else:
            ax.set_xlabel('Time [ms]')#(r"Event related potentials [$\mu$V]")#
            #ax.set_xticklabels(labels=times_xax)
            #plt.colorbar(im)
#         print([erp.ch_names[ii] for ii in ch_is])

    s = ",\nmasked for significance" if inds is not False else ""
    plt.suptitle("Channel-wise evoked activity{}".format(s))
# %%
def get_fdr_significance_indices(linreg_obj):
    data = linreg_obj.data
    ps = data.reshape(-1)
    reject, *_ = stats.multipletests(ps, alpha=0.05, method='fdr_bh', returnsorted=False)
    return reject.reshape(data.shape)

erp = grands['lexXope']
inds = get_fdr_significance_indices(erp)

#inds = erp.p_val.data < .01
plot_masked_image_fdr(erp, inds.T)


