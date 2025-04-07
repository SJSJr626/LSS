#!/usr/bin/env python

#BATCH --job-name=emu_lvl1_fc
#SBATCH --partition 16C_128G
#SBATCH --account iacc_madlab
#SBATCH --qos pq_madlab
#SBATCH -e err_lvl_lss_run_all
#SBATCH -o out_lvl_lss_run_all


"""
================================================================
EMU STUDY fMRI: FSL
================================================================

A firstlevel workflow for EMU STUDY session task data (SCAN 1).

This workflow makes use of:

- FSL

For example::

  python emu_study_lvl1_ATM.py -s 1001
                               -o /home/sjsuss626/data/madlab/data/mri/emu/frstlvl/study
                               -w /scratch/madlab/emu/frstlvl/study

"""

import os
import pandas as pd
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function, Merge
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.fsl.model import Level1Design, FEATModel, FILMGLS
from nipype.interfaces.fsl.utils import ImageMaths
from nipype.interfaces import afni
from nipype.interfaces import fsl
import nipype.interfaces.utility as util
from nipype.interfaces import freesurfer as fs

#from nipype import config, logging
#config.enable_debug_mode()
#logging.update_logging(config)

#Subject list
#subject_id = "sub-4040"

# Need to change the work and sink directories for each analysis
work_dir = '/home/sjsuss626/scratch/madlab/emur01/fsl/first_level/lvl1_negvneu_lss'
out_dir = '/home/sjsuss626/data/madlab/McMakin_EMUR01/derivatives/emu_ses-S1_NegvNeu/lvl1_negvneu_lss'

if not os.path.isdir(work_dir):
    os.mkdir(work_dir)

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# Functions
pop_lambda = lambda x : x[0]

def subjectinfo(subject_id):
    base_proj_dir = "/home/sjsuss626/data/madlab/McMakin_EMUR01/derivatives/emst_evs/fsl_ev_files"
    import os
    from os.path import join
    from nipype.interfaces.base import Bunch
    from copy import deepcopy
    import numpy as np
    from glob import glob
    
    output = []

    # For the current run, of which there are 2
    model_counter = 0 #MODEL COUNTER WILL UPDATE FOR EACH TRIAL OF A GIVEN CONDITION
    for curr_run in [1,2]:
        # READ IN THE 3-COLUMN FORMAT TEXT FILES FOR CURRENT RUN EVS
        # Variable to store all EVs for subsequent hit negative targets for current run
        data_neg = np.genfromtxt(base_proj_dir +
                                          '/{0}/ses-S1/{0}_ses-S1_task-study_run-{1}_neg_events.tsv'.format(subject_id, curr_run),
                                          delimiter='\t', dtype=str)

        # Variable to store all EVs for subsequent hit neutral targets for current run
        data_neu = np.genfromtxt(base_proj_dir +
                                          '/{0}/ses-S1/{0}_ses-S1_task-study_run-{1}_neu_events.tsv'.format(subject_id, curr_run),
                                          delimiter='\t', dtype=str)


        # CONSOLIDATE ALL POSITIVE, NEGATIVE AND NEUTRAL  
        orig_all_neg_neu_data = {'data_neg': data_neg,
                                      'data_neu': data_neu}

        # ITERATE OVER THE KEYS OF THE DICTIONARY TO ISOLATE THE CONDITIONS OF INTEREST
        for curr_key in orig_all_neg_neu_data.keys():
            # ESTABLISH TRIAL COUNTER FOR NAMING OF REGRESSORS
            trial_counter = 1
            # ISOLATE CURRENT CONDITION DATA USING POP FUNCTION
            # DICTIONARY WILL NO LONGER HAVE THAT KEY
            # I USE THAT FUNCTIONALITY TO ESTABLISH THE PENDING KEYS (NOT YET ITERATED OVER)
            copy_all_neg_neu_data = dict(orig_all_neg_neu_data)
            curr_condition_data = copy_all_neg_neu_data.pop(curr_key)

            if curr_condition_data.shape[0] > 0:
                if curr_condition_data.size == 3: # ONLY ONE EVENT OF THIS CONDITION DURING THIS RUN
                    names = [curr_key + '_run%d_trl%d' %(curr_run, trial_counter)]
                    onsets = [[float(curr_condition_data[0])]]
                    durations = [[float(curr_condition_data[1])]]
                    amplitudes = [[float(curr_condition_data[2])]]
                    # DEAL WITH THE REMAINING DATA THAT HASN'T BEEN ITERATED THROUGH YET (AKA PENDING)
                    for pending_key in copy_all_neg_neu_data.keys():
                        pending_data = copy_all_neg_neu_data[pending_key]
                        if pending_data.shape[0] > 0:
                            names.append(pending_key)
                            if pending_data.size == 3: #ONLY ONE EVENT OF THIS CONDITION
                                onsets.append([float(pending_data[0])])
                                durations.append([float(pending_data[1])])
                                amplitudes.append([float(pending_data[2])])
                            else:
                                onsets.append(list(map(float, pending_data[:, 0])))
                                durations.append(list(map(float, pending_data[:, 1])))
                                amplitudes.append(list(map(float, pending_data[:, 2])))

                    # UPDATE TRIAL COUNTER
                    trial_counter = trial_counter + 1

                    # Insert the contents of each run at the index of model_counter
                    output.insert(model_counter,
                                  Bunch(conditions = names,
                                        onsets = deepcopy(onsets),
                                        durations = deepcopy(durations),
                                        amplitudes = deepcopy(amplitudes),
                                        tmod = None,
                                        pmod = None,
                                        regressor_names = None,
                                        regressors = None))

                    # UPDATE MODEL COUNTER
                    model_counter = model_counter + 1
                else: # THERE IS MORE THAN ONE EVENT OF THIS CONDITION DURING THIS RUN
                    # ITERATE OVER THE NUMBER OF TRIALS WITHIN THAT CONDITION
                    for curr_cond_trl in range(len(curr_condition_data)):
                        # ESTABLISH THE LISTS FOR NAMES, ONSETS, DURATIONS, AND AMPLITUDES FOR ALL MODELS
                        # WE WILL HAVE AS MANY MODELS AS TRIALS ACROSS RUNS FOR THE DIFFERENT CONDITIONS
                        names = []
                        onsets = []
                        durations = []
                        amplitudes = []
                        curr_cond_trl_name = curr_key + '_run%d_trl%d' %(curr_run, trial_counter)
                        curr_cond_trl_onset = [float(curr_condition_data[curr_cond_trl][0])]
                        curr_cond_trl_dur = [float(curr_condition_data[curr_cond_trl][1])]
                        curr_cond_trl_amp = [float(curr_condition_data[curr_cond_trl][2])]
                    
                        names.append(curr_cond_trl_name)
                        onsets.append(curr_cond_trl_onset)
                        durations.append(curr_cond_trl_dur)
                        amplitudes.append(curr_cond_trl_amp)
                
                        # ISOLATE THE REMAINING TRIALS FOR THE CURRENT CONDITION USING THE NUMPY DELETE FUNCTION
                        # THIS FUNCTION WILL NOT MODIFY THE ORIGINAL VARIABLE LIKE POP DOES ABOVE
                        curr_cond_remaining_data = np.delete(curr_condition_data, curr_cond_trl, 0)
                        curr_cond_remaining_name = curr_key + '_allbut_run%d_trl%d' %(curr_run, trial_counter)
                        curr_cond_remaining_onsets = list(map(float, curr_cond_remaining_data[:, 0]))
                        curr_cond_remaining_durs = list(map(float, curr_cond_remaining_data[:, 1]))
                        curr_cond_remaining_amps = list(map(float, curr_cond_remaining_data[:, 2]))
                    
                        names.append(curr_cond_remaining_name)
                        onsets.append(curr_cond_remaining_onsets)
                        durations.append(curr_cond_remaining_durs)
                        amplitudes.append(curr_cond_remaining_amps)
                
                        # DEAL WITH THE PENDING DATA THAT HASN'T BEEN ITERATED THROUGH YET
                        # THIS IS WHERE THAT POP FUNCTION ABOVE CAME IN HANDY
                        for pending_key in copy_all_neg_neu_data.keys():
                            pending_data = copy_all_neg_neu_data[pending_key]
                            if pending_data.shape[0] > 0:
                                names.append(pending_key)
                                if pending_data.size == 3: #ONLY ONE EVENT OF THIS CONDITION
                                    onsets.append([float(pending_data[0])])
                                    durations.append([float(pending_data[1])])
                                    amplitudes.append([float(pending_data[2])])
                                else:
                                    onsets.append(list(map(float, pending_data[:, 0])))
                                    durations.append(list(map(float, pending_data[:, 1])))
                                    amplitudes.append(list(map(float, pending_data[:, 2])))

                        # UPDATE TRIAL COUNTER
                        trial_counter = trial_counter + 1

                        # Insert the contents of each run at the index of model_counter
                        output.insert(model_counter,
                                      Bunch(conditions = names,
                                            onsets = deepcopy(onsets),
                                            durations = deepcopy(durations),
                                            amplitudes = deepcopy(amplitudes),
                                            tmod = None,
                                            pmod = None,
                                            regressor_names = None,
                                            regressors = None))

                        # UPDATE MODEL COUNTER
                        model_counter = model_counter + 1

    return output

def get_contrasts(info):
    contrasts = []
    # For each bunch received from subjectinfo function in get_contrasts node
    for i, j in enumerate(info):
        curr_run_contrasts = []
        # For each EV list name received from the bunch
        for curr_cond in j.conditions:
            curr_cont = (curr_cond, 'T', [curr_cond], [1])
            curr_run_contrasts.append(curr_cont)
            
        contrasts.append(curr_run_contrasts)

    return contrasts

def get_subs(cons):
    subs = []
    for run_cons in cons:
        run_subs = []
        for i, con in enumerate(run_cons):
            run_subs.append(('cope%d.' % (i+1), 'cope%02d_%s.' % (i+1, con[0])))
            run_subs.append(('varcope%d.' % (i+1), 'varcope%02d_%s.' % (i+1, con[0])))
            run_subs.append(('zstat%d.' % (i+1), 'zstat%02d_%s.' % (i+1, con[0])))
            run_subs.append(('tstat%d.' % (i+1), 'tstat%02d_%s.' % (i+1, con[0])))
        subs.append(run_subs)
    return subs


def motion_noise(subjinfo, files):
    '''Grabs Motion Noise Files and Parameters Estimates'''
    import pandas as pd
    motion_noise_params = []
    motion_noi_par_names = []
    if not isinstance(files, list):
        files = [files]
    if not isinstance(subjinfo, list):
        subjinfo = [subjinfo]

    for i, filename in enumerate(files):
        nonvary_noi_par_names = [
            'trans_x', 'trans_x_derivative1',
            'trans_y', 'trans_y_derivative1',
            'trans_z', 'trans_z_derivative1',
            'rot_x', 'rot_x_derivative1',
            'rot_y', 'rot_y_derivative1',
            'rot_z', 'rot_z_derivative1'
        ]
        all_noise_data = pd.read_csv(filename, sep='\t')
        nonsteady_noise_covars = [x_ns for x_ns in all_noise_data.keys() if 'non_steady' in x_ns]
        motion_noise_covars = [x_m for x_m in all_noise_data.keys() if 'motion_outlier' in x_m]
        cosine_noise_coavars = [x_cos for x_cos in all_noise_data.keys() if 'cosine' in x_cos]

        if len(nonsteady_noise_covars) > 0 and len(motion_noise_covars) > 0:
            curr_mot_noi_par_names = cosine_noise_coavars + nonvary_noi_par_names + nonsteady_noise_covars + motion_noise_covars
        elif len(nonsteady_noise_covars) > 0 and len(motion_noise_covars) == 0:
            curr_mot_noi_par_names = cosine_noise_coavars + nonvary_noi_par_names + nonsteady_noise_covars
        elif len(motion_noise_covars) == 0 and len(motion_noise_covars) > 0:
            curr_mot_noi_par_names = cosine_noise_coavars + nonvary_noi_par_names + motion_noise_covars
        else:
            curr_mot_noi_par_names = cosine_noise_coavars + nonvary_noi_par_names

        noise_data = all_noise_data[curr_mot_noi_par_names]
        noise_data = noise_data.fillna(0)

        motion_noise_params.append([[]] * noise_data.shape[1])
        for i_x, curr_noise_name in enumerate(curr_mot_noi_par_names):
            motion_noise_params[i][i_x] = noise_data[curr_noise_name].tolist()
        motion_noi_par_names.append(curr_mot_noi_par_names)

    for j, i in enumerate(subjinfo):
        if i.regressor_names == None:
            i.regressor_names = []
        if i.regressors == None:
            i.regressors = []

        if 'run1' in i.conditions[0]:
            curr_run = 0
        elif 'run2' in i.conditions[0]:
            curr_run = 1

        for j3, i3 in enumerate(motion_noise_params[curr_run]):
            i.regressor_names.append(motion_noi_par_names[curr_run][j3])
            i.regressors.append(i3)

    return subjinfo

#    print(motion_noise_params)
#    print(len(subjinfo))
#        if info.regressor_names is None:
#            info.regressor_names = []
#        if info.regressors is None:
#            info.regressors = []
#        for j, param in enumerate(motion_noise_params[i]):
#            info.regressor_names.append(motion_noi_par_names[i][j])
#            info.regressors.append(param)
#            print(i,j)
#    return subjinfo

#Keep for now, investigate further later. 
def expand_files(subjinfo, in_files):
    if not isinstance(in_files, list):
        in_files = [in_files]
    files_expanded = []
    for j,i in enumerate(subjinfo):
        # Deal with multiple trialwise models for each run
        if 'run1' in i.conditions[0]:
            files_expanded.append(in_files[0])
        elif 'run2' in i.conditions[0]:
            files_expanded.append(in_files[1])

    return files_expanded

####################Copied from other script################
# Need to change the work and sink directories for each analysis
work_dir = '/home/sjsuss626/scratch/madlab/emur01/fsl/first_level/lvl1_negvneu_lss'
sink_dir = '/home/sjsuss626/data/madlab/McMakin_EMUR01/derivatives/emu_ses-S1_NegvNeu/lvl1_negvneu_lss'
base_mripreprocdata_dir = '/home/sjsuss626/data/madlab/McMakin_EMUR01/derivatives/fmriprep'

if not os.path.isdir(sink_dir):
    os.mkdir(sink_dir)

'''Defines the Firstlevel Workflow for the EMU task'''
frstlvl_wf = Workflow(name='frstlvl_wf')


#Pulls list of subject IDS
#sids_df = pd.read_csv('/home/sjsuss626/data/madlab/McMakin_EMUR01/code/steve/usable_IDs.csv', header=None)
#sids_df = pd.read_csv('/home/sjsuss626/data/madlab/McMakin_EMUR01/code/steve/pending_subjects_lvl2.csv', header=None)
#sids_str_list = sids_df[0].tolist()
#sids_str_list = [f"sub-{curr_sid}" for curr_sid in sids_int_list]

# the below comment out when finished development of script
sids = ['sub-4262']

print(sids)

subjID_infosource = Node(IdentityInterface(fields=['subject_id']), name='subjID_infosource')
subjID_infosource.iterables = ('subject_id', sids)

############################################################


frstlvl_lss_wf = Workflow(name='frstlvl_lss_wf')

info = dict(task_mri_files=[['subject_id']],
            motion_noise_files=[['subject_id']])

# Create a Function node to define stimulus onsets, etc... for each subject
subject_info = Node(Function(input_names=['subject_id'],
                                output_names=['output'],
                                function=subjectinfo),
                    name='subject_info')
#subject_info.plugin_args={'sbatch_args': ('--partition IB_40C_512G --nodelist=n089 --cpus-per-task=1 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
frstlvl_lss_wf.connect(subjID_infosource, 'subject_id', subject_info, 'subject_id')

# Create another Function node to define the contrasts for the experiment
getcontrasts = Node(Function(input_names=['subject_id','info'],
                                output_names=['contrasts'],
                                function=get_contrasts),
                    name='getcontrasts')
#getcontrasts.plugin_args={'sbatch_args': ('--partition IB_40C_512G --nodelist=n089 --cpus-per-task=1 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
#frstlvl_lss_wf.connect(subjID_infosource, 'subject_id', getcontrasts, 'subject_id')
frstlvl_lss_wf.connect(subject_info, 'output', getcontrasts, 'info')

# Create a Function node to substitute names of files created during pipeline
getsubs = Node(Function(input_names=['cons'],
                        output_names=['subs'],
                        function=get_subs),
                name='getsubs')
#getsubs.plugin_args={'sbatch_args': ('--partition IB_40C_512G --nodelist=n089 --cpus-per-task=1 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
frstlvl_lss_wf.connect(subjID_infosource, 'subject_id', getsubs, 'subject_id')
frstlvl_lss_wf.connect(subject_info, 'output', getsubs, 'info')
frstlvl_lss_wf.connect(getcontrasts, 'contrasts', getsubs, 'cons')

# Create a datasource node to get the task_mri and motion-noise files
datasource = Node(DataGrabber(infields=['subject_id'], outfields=list(info.keys())), name='datasource')
datasource.inputs.template = '*'
datasource.inputs.base_directory = os.path.abspath('/home/sjsuss626/data/madlab/McMakin_EMUR01/derivatives/fmriprep')
datasource.inputs.field_template = dict(task_mri_files='%s/ses-S1/func/*_ses-S1_task-study_run-*_space-MNIPediatricAsym_cohort-5_res-2_desc-preproc_bold.nii.gz',
                                        motion_noise_files='%s/ses-S1/func/*_ses-S1_task-study_run-*_desc-confounds_timeseries.tsv') #add subject specific masks from fmri prep (dialate by 1)
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True
datasource.inputs.raise_on_empty = True
#datasource.plugin_args={'sbatch_args': ('--partition IB_40C_512G --nodelist=n089 --cpus-per-task=1 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
frstlvl_lss_wf.connect(subjID_infosource, 'subject_id', datasource, 'subject_id')

# Create a Function node to modify the motion and noise files to be single regressors
motionnoise = Node(Function(input_names=['subjinfo', 'files'],
                            output_names=['subjinfo'],
                            function=motion_noise),
                    name='motionnoise')
#motionnoise.plugin_args={'sbatch_args': ('--partition IB_40C_512G --nodelist=n089 --cpus-per-task=1 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
frstlvl_lss_wf.connect(subject_info, 'output', motionnoise, 'subjinfo')
frstlvl_lss_wf.connect(datasource, 'motion_noise_files', motionnoise, 'files')


# Function node to expand task functional data
expand_epi_files = Node(Function(input_names = ['subjinfo', 'in_files'],
                                    output_names = ['files_expanded'],
                                    function = expand_files),
                        name = 'expand_epi_files')
# The bunch from subject_info function containing regressor names, onsets, durations, and amplitudes
#expand_epi_files.plugin_args={'sbatch_args': ('--partition IB_40C_512G --nodelist=n089 --cpus-per-task=1 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
frstlvl_lss_wf.connect(motionnoise, 'subjinfo', expand_epi_files, 'subjinfo')
frstlvl_lss_wf.connect(datasource, 'task_mri_files', expand_epi_files, 'in_files')

# Create a specify model node
specify_model = Node(SpecifyModel(), name='specify_model')
specify_model.inputs.high_pass_filter_cutoff = -1.0
specify_model.inputs.input_units = 'secs'
specify_model.inputs.time_repetition = 1.760
#specify_model.plugin_args={'sbatch_args': ('--partition IB_40C_512G --nodelist=n089 --cpus-per-task=1 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
frstlvl_lss_wf.connect(expand_epi_files, 'files_expanded', specify_model, 'functional_runs')
frstlvl_lss_wf.connect(motionnoise, 'subjinfo', specify_model, 'subject_info')

# Create an InputSpec node for the modelfit node
modelfit_inputspec = Node(IdentityInterface(fields=['session_info', 'interscan_interval', 'contrasts',
                                                    'film_threshold', 'functional_data', 'bases',
                                                    'model_serial_correlations'], mandatory_inputs=True),
                            name = 'modelfit_inputspec')
modelfit_inputspec.inputs.bases = {'dgamma':{'derivs': False}}
modelfit_inputspec.inputs.film_threshold = 0.0
modelfit_inputspec.inputs.interscan_interval = 1.760
modelfit_inputspec.inputs.model_serial_correlations = True
#modelfit_inputspec.plugin_args={'sbatch_args': ('--partition IB_40C_512G --nodelist=n089 --cpus-per-task=1 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
frstlvl_lss_wf.connect(expand_epi_files, 'files_expanded', modelfit_inputspec, 'functional_data')
frstlvl_lss_wf.connect(getcontrasts, 'contrasts', modelfit_inputspec, 'contrasts')
frstlvl_lss_wf.connect(specify_model, 'session_info', modelfit_inputspec, 'session_info')

# Create a level1 design node
level1_design = MapNode(Level1Design(),
                        iterfield = ['contrasts', 'session_info'],
                        name='level1_design')
#level1_design.plugin_args={'sbatch_args': ('--partition IB_40C_512G --nodelist=n089 --cpus-per-task=1 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
frstlvl_lss_wf.connect(modelfit_inputspec, 'interscan_interval', level1_design, 'interscan_interval')
frstlvl_lss_wf.connect(modelfit_inputspec, 'session_info', level1_design, 'session_info')
frstlvl_lss_wf.connect(modelfit_inputspec, 'contrasts', level1_design, 'contrasts')
frstlvl_lss_wf.connect(modelfit_inputspec, 'bases', level1_design, 'bases')
frstlvl_lss_wf.connect(modelfit_inputspec, 'model_serial_correlations', level1_design, 'model_serial_correlations')

# Creat a MapNode to generate a model for each run
generate_model = MapNode(FEATModel(),
                            iterfield=['fsf_file', 'ev_files'],
                            name='generate_model')
generate_model.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
generate_model.inputs.output_type = 'NIFTI_GZ'
#generate_model.plugin_args={'sbatch_args': ('--partition IB_40C_512G --nodelist=n089 --cpus-per-task=1 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
frstlvl_lss_wf.connect(level1_design, 'fsf_files', generate_model, 'fsf_file')
frstlvl_lss_wf.connect(level1_design, 'ev_files', generate_model, 'ev_files')

# Create a MapNode to estimate the model using FILMGLS
estimate_model = MapNode(FILMGLS(),
                            iterfield=['design_file', 'in_file', 'tcon_file'],
                            name='estimate_model')
estimate_model.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
estimate_model.inputs.mask_size = 5
estimate_model.inputs.output_type = 'NIFTI_GZ'
estimate_model.inputs.results_dir = 'results'
estimate_model.inputs.smooth_autocorr = True
#estimate_model.plugin_args={'sbatch_args': ('--partition IB_40C_1.5T --nodelist=n098,n104,n105 --cpus-per-task=1 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
#estimate_model.plugin_args={'sbatch_args': ('--partition IB_40C_512G --nodelist=n110,n111 --cpus-per-task=1 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
frstlvl_lss_wf.connect(modelfit_inputspec, 'film_threshold', estimate_model, 'threshold')
frstlvl_lss_wf.connect(modelfit_inputspec, 'functional_data', estimate_model, 'in_file')
frstlvl_lss_wf.connect(generate_model, 'design_file', estimate_model, 'design_file')
frstlvl_lss_wf.connect(generate_model, 'con_file', estimate_model, 'tcon_file')

# Create an outputspec node
modelfit_outputspec = Node(IdentityInterface(fields=['copes', 'varcopes', 'dof_file',
                                                        'design_image', 'design_file', 'design_cov'
                                                    ], mandatory_inputs=True),
                            name='modelfit_outputspec')
#modelfit_outputspec.plugin_args={'sbatch_args': ('--partition IB_40C_512G --nodelist=n089 --cpus-per-task=1 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
frstlvl_lss_wf.connect(estimate_model, 'copes', modelfit_outputspec, 'copes')
frstlvl_lss_wf.connect(estimate_model, 'varcopes', modelfit_outputspec, 'varcopes')
frstlvl_lss_wf.connect(generate_model, 'design_image', modelfit_outputspec, 'design_image')
frstlvl_lss_wf.connect(generate_model, 'design_file', modelfit_outputspec, 'design_file')
frstlvl_lss_wf.connect(generate_model, 'design_cov', modelfit_outputspec, 'design_cov')
frstlvl_lss_wf.connect(estimate_model, 'dof_file', modelfit_outputspec, 'dof_file')

# Create a datasink node
sinkd = MapNode(DataSink(),
                iterfield=['substitutions', 'modelfit.contrasts.@copes', 'modelfit.contrasts.@varcopes'],
                name='sinkd')
sinkd.inputs.base_directory = sink_dir
#sinkd.plugin_args={'sbatch_args': ('--partition IB_40C_512G --nodelist=n089 --cpus-per-task=2 --account iacc_madlab --qos pq_madlab'), 'overwrite': True}
frstlvl_lss_wf.connect(getsubs, 'subs', sinkd, 'substitutions')
frstlvl_lss_wf.connect(modelfit_outputspec, 'dof_file', sinkd, 'modelfit.dofs')
frstlvl_lss_wf.connect(modelfit_outputspec, 'copes', sinkd, 'modelfit.contrasts.@copes')
frstlvl_lss_wf.connect(modelfit_outputspec, 'varcopes', sinkd, 'modelfit.contrasts.@varcopes')
frstlvl_lss_wf.connect(modelfit_outputspec, 'design_image', sinkd, 'modelfit.design')
frstlvl_lss_wf.connect(modelfit_outputspec, 'design_cov', sinkd, 'modelfit.design.@cov')
frstlvl_lss_wf.connect(modelfit_outputspec, 'design_file', sinkd, 'modelfit.design.@matrix')
frstlvl_lss_wf.connect(subjID_infosource, 'subject_id', sinkd, 'container')


"""
Creates the full workflow
"""

frstlvl_lss_wf.config['execution']['crashdump_dir'] = '/home/sjsuss626/scratch/madlab/emur01/crash/first_level/lvl1_ses-S1_negvneu_lss'
frstlvl_lss_wf.base_dir = work_dir 
frstlvl_lss_wf.run('MultiProc', plugin_args={'n_procs':6})
#frstlvl_lss_wf.run(plugin='SLURM', plugin_args={'sbatch_args': ('--partition 16C_128G --account iacc_madlab --qos pq_madlab'), 'overwrite': True})
