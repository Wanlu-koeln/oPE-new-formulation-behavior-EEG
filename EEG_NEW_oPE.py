# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
# %% Cell 1: import libraries
import matplotlib

matplotlib.use ( 'agg' )  # You can also try 'Qt4Agg' if 'Qt5Agg' doesn't work
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import zscore
import os
from os import listdir
import mne
import pandas
mne.set_log_level ( 'Critical' )
import mne
from mne.channels import find_ch_adjacency
from scipy.spatial.distance import squareform , pdist
from mne.stats import spatio_temporal_cluster_1samp_test as stct
from mne.stats import linear_regression
from scipy.sparse import csc_matrix



# %%from mne.preprocessing import find_outliers, ICA, read_ica, create_ecg_epochs, create_eog_epochs

sourceloc = '/Users/wanlufu/Downloads/eeg_dat/'
files = [ x for x in listdir ( sourceloc ) if "_postICA-epo.fif" in x ]

for file in files :
    print ( (sourceloc + file) )

len ( files )



# %% import presictors
predictors_source = sourceloc + 'human_eeg_indi_characteristics_behavior.csv'

# %% add predicotr variables
predictors = pandas.read_csv (
    predictors_source )  # predictors['stimcat'] = 0.25 #predictors['stimcat'][predictors.stimcode == 'w_hild'] = 0.75 #predictors['stimcat'][predictors.stimcode == 'cs'] = 0 #predictors['stimcat'][predictors.stimcode == 'pw'] = 1 predictors['log_vpe'] = np.round(np.log(predictors['vis_pred_err']),1) predictors['log_lum'] = np.round(np.log(predictors['lumi']),2) predictors['stand_vpe'] = np.round((predictors['vis_pred_err']-np.mean(predictors['vis_pred_err']))/np.std(predictors['vis_pred_err']),2) predictors['stand_lum'] = np.round((predictors['lumi']-np.mean(predictors['lumi']))/np.std(predictors['lumi']),2) predictors['iscs'] = 0 predictors['iscs'][predictors.stimcode == 'cs'] = 1 predictors['ispw'] = 0 predictors['ispw'][predictors.stimcode == 'pw'] = 1 predictors['isw'] = 1 predictors['isw'][predictors.stimcode == 'w_hild'] = 0 predictors['isw'][predictors.stimcode == 'w_lowld'] = 0 predictors['intercept'] = 1 predictors

predictors[ 'log_vpe' ] = np.round ( np.log ( predictors[ 'vis_pred_err' ] ) , 1 )
predictors[ 'log_lum' ] = np.round ( np.log ( predictors[ 'lumi' ] ) , 2 )

predictors[ 'stand_vpe' ] = np.round (
    (predictors[ 'vis_pred_err' ] - np.mean ( predictors[ 'vis_pred_err' ] )) / np.std (
        predictors[ 'vis_pred_err' ] ) , 2 )
predictors[ 'stand_lum' ] = np.round (
    (predictors[ 'lumi' ] - np.mean ( predictors[ 'lumi' ] )) / np.std ( predictors[ 'lumi' ] ) , 2 )

predictors[ 'iscs' ] = 0
predictors[ 'iscs' ][ predictors.stimcode == 'cs' ] = 1

predictors[ 'ispw' ] = 0
predictors[ 'ispw' ][ predictors.stimcode == 'pw' ] = 1

predictors[ 'isw' ] = 1
predictors[ 'isw' ][ predictors.stimcode == 'w_hild' ] = 0
predictors[ 'isw' ][ predictors.stimcode == 'w_lowld' ] = 0

predictors[ 'intercept' ] = 1

predictors

# %% define predictor variables

predictors[ 'lex_1' ] = (predictors[ 'lex' ] - .5) * 2
predictors[ 'concr_woNA' ] = predictors[ 'concr' ]
predictors[ 'concr_woNA' ][ pandas.isnull ( predictors[ 'concr_woNA' ] ) ] = 0  # max(predictors['concr'])
predictors[ 'lumXvpe' ] = predictors[ 'stand_vpe' ] * predictors[ 'stand_lum' ]
predictors[ 'lexXvpe' ] = predictors[ 'stand_vpe' ] * (predictors[ 'lex_1' ])
predictors[ 'freqXvpe' ] = predictors[ 'stand_vpe' ] * (predictors[ 'lgfreqCount' ])
predictors[ 'lexXold' ] = predictors[ 'old' ] * (predictors[ 'lex_1' ])
predictors[ 'lexXlum' ] = predictors[ 'stand_lum' ] * (predictors[ 'lex_1' ])
predictors[ 'lexXfreq' ] = predictors[ 'stand_vpe' ] * zscore ( predictors[ 'lgfreqCount' ] )
predictors[ 'lexXope' ] = predictors[ 'ope_norm' ] * (predictors[ 'lex_1' ])
predictors[ 'lexXape' ] = predictors[ 'ape_norm' ] * (predictors[ 'lex_1' ])
predictors[ 'lexXspe' ] = predictors[ 'spe_norm' ] * (predictors[ 'lex_1' ])

predictors


# %% design evoked
def make_design ( epochs , variables , db_ ) :
    predictors_ = db_  # if "wy2201_postICA-epo.fif" not in epochs.info["subject_info"] else db_[2:].copy()

    # print(len(predictors_))
    predictors = predictors_  # .query("lex == 1")#.query("rand_select == 1")#.query("lex == 1")
    epochs_ = epochs[
        predictors.index ]  # if "wy2201_postICA-epo.fif" not in epochs.info._attributes["subject_info"] else epochs[predictors.index-2]
    n_trials = len ( epochs_.events )

    design = np.zeros ( (n_trials , len ( variables ) + 1) )

    design[ : , 0 ] = 1

    for ii , key in enumerate ( variables ) :
        design[ : , ii + 1 ] = predictors[ key ]

    intercept = np.ones ( (len ( epochs_ ) ,) , dtype = np.float64 )

    evoked = linear_regression ( epochs_ , design , [ 'intercept' ] + variables )

    return evoked
#%%
def make_lm ( f , factors ) :
    epochs = mne.read_epochs ( sourceloc + f )
    epochs.drop_bad ()
    epochs.set_eeg_reference ( ref_channels = [ 'Cz' ] )
    epochs.crop ( -.2 , .8 )

    # predictors_ = predictors if "wy2201_postICA-epo.fif" not in epochs.info["subject_info"] else predictors[2:].copy()

    db_ = predictors  # .query("lex == 1")
    epochs_ = epochs[ db_.index ]
    epochs_.info._attributes[ "subject_info" ] = f
    print ( epochs_.info._attributes[ "subject_info" ] )

    evoked = make_design ( epochs_ , factors , db_ )

    return evoked
#%%
variables = ['lex_1','ope_norm','lexXope','lumi']
lm_tmp = make_lm(files[29], variables)
#lm_tmp["vis_pred_err"].beta.plot_joint(times = [.05,.1,.12,.14,.15,.2,.25,.3,.35,.4,.45])#times = [.39,.49]);
#lm_tmp["vis_pred_err"].t_val.plot_joint(times = [.05,.1,.12,.14,.15,.17,.2,.25,.3,.35,.4,.45])#times = [.1,.12,.14,.2,.22,.24]);

mpl.rcParams['savefig.dpi'] = 150
#%%
lms = [make_lm(f, variables) for f in files]
i = 0
for i in range(len(lms)):
    print(files[i])
    print(i)
    #lms[i]["vis_pred_err"].beta.plot_joint(times = [.35,.4,.45,.5,.55,.6,.65]);#.05,.1,.15,.2,.25,.3,


mpl.rcParams['savefig.dpi'] = 100
#%%
grands = {variable:mne.grand_average([lm_[variable].beta for lm_ in lms])
          for variable in variables}
#%%
def adapt_data_for_plj(grands,key,title):
    mpl.rcParams.update({'font.size': 13})
    #grands[key].data = grands[key].data /1e6
    grands[key].plot_joint(
        #times = times[key],
        title=title[key])#title=key)#
    #grands[key].data = grands[key].data *1e6
#%%
output_folder = '/Users/wanlufu/Downloads/eeg_dat/'
os.makedirs(output_folder, exist_ok=True)
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
#           'ape_norm': [.23],
#           'spe_norm': [.23],
           'lex_1': [.23,.43],
           'lexXope':[.43],
           'lumi':[.43]
#           'lexXape': [.43],
#           'lexXspe': [.43]
           }
title = {'ope_norm': 'Effect size [$\mu$V] for 50% oPE covariate',
         'lex_1': 'Effect size [$\mu$V] for lexicality covariate',
         'lexXope': 'Effect size [$\mu$V] for lexicality by 50% oPE interaction',
         'lumi':'Effect size [$\mu$V] for pixel number covariate'
         }

for key in grands:
    adapt_data_for_plj(grands,key,title)
    #grands[key].plot_joint(title=key,times = times)#,times = times,ts_args=ts_args,topomap_args=topomap_args);#,ts_args=ts_args,topomap_args=topomap_args
    #plot_topomap(times = times,vmin=-.2,vmax=.2)
    #plot_topomap(times = times,vmin=-1.3,vmax=1.3)
    #plot_joint(title=key,ts_args=ts_args,times = times);#,topomap_args=topomap_args);


    # Save the figure with a unique name based on the key
    fig = plt.gcf ()
    fig.savefig ( os.path.join ( output_folder , f'{key}_plot.png' ) )
    plt.close ()  # Close the figure to free up resources

# %%

def indi_evok_per_cond ( lm ) :
    # in_grands = {}
    in_lm = {}
    for key in variables :
        in_lm[ key ] = lm[ key ].beta
        # in_grands[cond] = mne.grand_average(all_evokeds[cond])

    return in_lm


# %%

# rerps (bei dir wäre ein anderer name besser ...): eine liste von dictionaries,
# bei denen die keys conditions sind und die values erps; ein dictionary pro VP


rerps = [ indi_evok_per_cond ( lm_ ) for lm_ in lms ]

# %% 
erp = grands[ variables[ 0 ] ].copy ()
from mne.channels.layout import _find_topomap_coords

xy = np.asarray ( _find_topomap_coords ( erp.info , range ( len ( erp.ch_names ) ) ) )

dsts = squareform ( pdist ( xy ) )
#%%
def run_cluster_permutation_test ( data , epochs_list , n_permutations = 1000 ) :
    # Use the info from the first epoch for connectivity computation
    connectivity , ch_names = mne.channels.find_ch_adjacency ( epochs_list[ 0 ].info , ch_type = 'eeg' )

    # Extract the data from EvokedArray objects
    data = [ evoked.data for evoked in data ]
    # Ensure all data arrays have the same size along the concatenation axis
    max_size = max ( arr.shape[ -1 ] for arr in data )
    data_padded = [ np.pad ( arr , ((0 , 0) , (0 , max_size - arr.shape[ -1 ])) , mode = 'constant' ) for arr in data ]

    # Concatenate the padded data along the last axis
    data_concat = np.concatenate ( data_padded , axis = -1 )

    # Define the clustering parameters
    threshold = 2.0  # Set an appropriate threshold

    # Run the spatio-temporal clustering
    clusters , p_values , _, _ = stct ( data_concat , threshold = threshold , tail = 0 , n_permutations = n_permutations )

    return clusters , p_values


#%%
output_dir = '/Users/wanlufu/Downloads/eeg_dat/'  # Specify the directory where you want to save the plots

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

epochs_list = [mne.read_epochs(f) for f in files]
n_permutations = 1000  # Set the number of permutations

for variable in variables:
    data_all = [lm[variable].beta.data for lm in lms]
    data_stack = np.stack(data_all, axis=0)

    # Run cluster-based permutation test
    clusters, p_values = run_cluster_permutation_test(data_stack, epochs_list, n_permutations)

    # Plot the results
    plt.figure()
    plt.plot(data_stack.mean(axis=0).T)
    plt.title(f"Cluster-based permutation test for {variable}")
    plt.xlabel('Time points')
    plt.ylabel('Mean beta value')

    # Save the plot (show on the right panel in .pages)
    plot_filename = os.path.join(output_dir, f"cluster_permutation_{variable}.png")
    plt.savefig(plot_filename)
    plt.close()

    times = epochs_list[0].times
#%% want to print the significant time range of cluster-based permutation results (got errors)
def print_significant_time_ranges(clusters, p_values, times, variables, threshold=0.05):
    # Find indices of time points that are part of significant clusters
    significant_clusters = [i for i, p_value in enumerate(p_values) ]#if p_value < threshold]

    # Extract start and stop times of each significant cluster for each variable
    significant_time_ranges = {variable: [] for variable in variables}

    for variable_index, variable in enumerate(variables):
        for i in significant_clusters:
            cluster = clusters[i][variable_index]
            # Find the indices where the cluster is significant
            significant_indices = np.where(cluster)

            if len(significant_indices[0]) > 0:
                # Extract the start and stop times
                start, stop = times[significant_indices[0][0]], times[significant_indices[0][-1]]
                significant_time_ranges[variable].append((start, stop))

    # Print the significant time ranges for each variable
    for variable, time_ranges in significant_time_ranges.items():
        print(f"Significant time ranges for {variable}:")
        for start, stop in time_ranges:
            print(f"Start: {start:.3f}s, Stop: {stop:.3f}s")

#%%
# Run cluster-based permutation test
clusters, p_values = run_cluster_permutation_test(data_stack, epochs_list, n_permutations)

# Print significant time ranges
print_significant_time_ranges(clusters, p_values, times, variables)

#%%
#plot the topography (got errors)

stc_clusters = mne.SourceEstimate(np.ones(len(times)), vertices=[[], []], tmin=times.min(),
                                       tstep=times[1] - times[0])
stc_clusters.data[:, :] = p_values[:, np.newaxis] < 0.05

    # Create a template Info object to use for plotting (choose one epoch)
    info_template = epochs_list[0].info

    # Convert the SourceEstimate to an Evoked object for plotting
    evoked_clusters = mne.EvokedArray(stc_clusters.data.T, info=info_template, tmin=times.min())

    # Plot the significant clusters on the brain
    brain = evoked_clusters.plot_topo(hemi='both', title=f"Cluster permutation brain for {variable}",
                                 #     subjects_dir=mne.datasets.sample.data_path(), time_unit='s')

    # Save the brain plot
    brain_filename = os.path.join(output_dir, f"cluster_permutation_brain_{variable}.png")
    brain.savefig(fname=brain_filename)
    brain.close()










