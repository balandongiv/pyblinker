function process_eeg_with_pop_blinker()
% Step 0
% PROCESS_EEG_WITH_POP_BLINKER
% -------------------------------------------------------------------------
% Gold-standard starter script for running EEGLAB's `pop_blinker` on a
% single EEG recording using default-ish settings.
%
% This is the pattern we usually start with when we prepare MATLAB
% reference runs that will later be *reproduced/migrated* in Python
% (e.g. for the `pyblinker` package).
%
% Why this matters:
% - we call `pop_biosig` to load an EDF (consistent, open format)
% - we call `pop_blinker` once, with a deterministic `params` struct
% - we save **exactly** the `EEG` and the `params` we passed in
% - we keep folder layout predictable so the Python side can compare
%
% Notes you must keep in mind for migration work:
% 1) In older code this function was sometimes called `step0_process_EEG`.
%    We're renaming it to something clearer: `process_eeg_with_pop_blinker`.
%    In the docs or migration notes you can still say:
%       "Usually we start with: step0_process_EEG()"
%    but the actual function name here is the improved one.
%
% 2) We try to load `config.m` **if it exists** in the same folder as this
%    file. That file may define:
%       - eeglab_path   : where EEGLAB lives
%       - main_folder   : where to save the .mat output
%       - blinker_dir   : where Blinker should write its artifacts
%    If `config.m` is not there, we fall back to a sensible default layout
%    relative to this file.
%
% 3) We explicitly initialize EEGLAB in **no-GUI** mode:
%          eeglab nogui
%    because this script is meant to run in automated/testing/migration
%    contexts.
%
% 4) We assume we want to process the file:
%       mne_sample_audvis_raw.edf
%    sitting under:
%       .../pyblinker/test_files
%    This matches the "gold" run that the Python side will use.
%
% 5) Output is saved as:
%       process_eeg_popblinker_input.mat
%    (renamed from the older, more opaque
%     `step0_data_input_allChannels_popblinker.mat`)
%
% 6) The `params` we build in the helper `build_blinker_params()` show the
%    fields we expect from MATLAB → Python. This is intentional. Even if
%    Blinker can auto-construct some of these, we **spell them out** so the
%    migration code can see them.
%
% In short: **this script is the canonical, "gold" MATLAB run** for
% pop_blinker that we want to mirror in pyblinker.

% Low cut off hz:1
% High cutt of 20
% There are 5 channels, the number of points is 27800
% The selected channel is no 3
% Channel number: 3, channel label: 003
% Total blinks: 85,  good blinks: 59, good ratio: 0.75
% -------------------------------------------------------------------------

    % ---------------------------------------------------------------------
    % 1. Resolve paths & config via shared helper
    % ---------------------------------------------------------------------
    paths = sharedMigrationPaths(struct( ...
        'DefaultDataSubfolder', 'test_files', ...
        'DefaultOutputSubfolder', 'migration_files', ...
        'DataDirCandidates', {{'data_dir', 'main_folder'}}, ...
        'OutputDirCandidates', {{'main_folder'}}, ...
        'EnsureOutputDir', true));

    data_dir_use = paths.data_dir;
    output_dir = paths.output_dir;
    config_vars = paths.config_vars;

    % ---------------------------------------------------------------------
    % 3. Initialize EEGLAB silently (if path known)
    % ---------------------------------------------------------------------
    eeglab_path = extract_config_path(config_vars, 'eeglab_path');

    if ~isempty(eeglab_path) && isfolder(eeglab_path)
        addpath(genpath(eeglab_path));
        eeglab nogui;
    else
        % if user already has eeglab in path, this will still work
        try
            eeglab nogui;
        catch
            error('EEGLAB could not be started. Make sure it is on your MATLAB path or define eeglab_path in config.m');
        end
    end

    % ---------------------------------------------------------------------
    % 4. Build input and output filenames
    % ---------------------------------------------------------------------
    eeg_file_path = fullfile(data_dir_use, 'mne_sample_audvis_raw.edf');
    if ~exist(eeg_file_path, 'file')
        error('EEG file not found at: %s', eeg_file_path);
    end

    % more descriptive, migration-friendly name
    output_file = fullfile(output_dir, 'process_eeg_popblinker_input.mat');

    % ---------------------------------------------------------------------
    % 5. Load EEG with pop_biosig


    % ---------------------------------------------------------------------
    EEG = pop_biosig(eeg_file_path);
    % find the channel whose label is '003'
    idx = find(strcmp({EEG.chanlocs.labels}, '003')); % we know from our expirement, the representative is 003,

    % Find the indices for labels '003' and '005'
    % idx = find(ismember({EEG.chanlocs.labels}, {'003','005'}));

    % keep only that channel
    EEG = pop_select(EEG, 'channel', idx);

    % ---------------------------------------------------------------------
    % 6. Build Blinker params (explicit, migration-friendly)
    % ---------------------------------------------------------------------
    blinker_dir = extract_config_path(config_vars, 'blinker_dir');
    if ~isempty(blinker_dir) && isfolder(blinker_dir)
        params = build_blinker_params(blinker_dir);
    else
        params = build_blinker_params(output_dir);
    end

    % ---------------------------------------------------------------------
    % 7. Run pop_blinker, save results
    % ---------------------------------------------------------------------
    try
        [EEG, com, blinks, blinkFits, blinkProperties, blinkStatistics, params] = ...
            pop_blinker(EEG, params); %#ok<NASGU,ASGLU> (we only save EEG & params by design)

        save(output_file, 'EEG', 'params');
        disp(['Processing complete. EEG and params saved as: ' output_file]);
    catch ME
        % For migration, it's better to show the exact error
        warning('Error while running pop_blinker: %s', ME.message);
        rethrow(ME);
    end
end


function value = extract_config_path(config_vars, field_name)
%EXTRACT_CONFIG_PATH Safely obtain a string path from config variables.
    value = '';
    if ~isfield(config_vars, field_name)
        return;
    end

    candidate = config_vars.(field_name);
    if isa(candidate, 'string') && isscalar(candidate)
        candidate = char(candidate);
    elseif iscell(candidate) && ~isempty(candidate)
        candidate = candidate{1};
        if isa(candidate, 'string') && isscalar(candidate)
            candidate = char(candidate);
        end
    end

    if ischar(candidate)
        value = candidate;
    end
end

function params = build_blinker_params(blinker_dir)
% BUILD_BLINKER_PARAMS  Construct an explicit Blinker params struct.
% We make it explicit so the Python migration can see every field.

    if ~exist(blinker_dir, 'dir')
        mkdir(blinker_dir);
    end

    params = struct();

    % where Blinker will drop its intermediate / final blink info
    params.blinkerSaveFile  = fullfile(blinker_dir, '_blinks.mat');
    params.blinkerDumpDir   = fullfile(blinker_dir, 'blinkDump');

    % basic meta – these are not strictly required for detection but make
    % the struct complete and predictable
    params.experiment       = 'Experiment1';
    params.subjectID        = 'Subject1_Task1_Experiment1_Rep1';
    params.task             = 'Task1';
    params.uniqueName       = 'Unknown';
    params.startDate        = '01-Jan-2016';
    params.startTime        = '00:00:00';
    params.signalTypeIndicator = 'UseNumbers';

    % signal selection – in many real runs this is populated from EEG.chanlocs
    % Here we just show an explicit, simple case for testing/migration
    % params.signalNumbers    = 3;  % simplest case
    % params.signalLabels     = {'003'};

    % dump / debug flags – off for "gold" run
    params.showMaxDistribution  = false;
    params.dumpBlinkerStructure = false;
    params.dumpBlinkPositions   = false;
    params.dumpBlinkImages      = false;

    % small console hint (useful when running batches)
    disp('Blinker params created:');
    disp(params);
end
