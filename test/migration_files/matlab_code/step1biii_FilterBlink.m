function process_fit_blinks()
% PROCESS_FIT_BLINKS
% -------------------------------------------------------------------------
% Reference implementation for the **FilterBlink** stage (STEP 1biii) in
% the Blinker pipeline.
%
% This function was named **process_fit_blinks** because Blinker performs
% most of its calculations within the `extractBlinksEEG` function, directly
% after the `extractBlinks` call. Based on this design, when running this
% function in debug mode (e.g., stopping at line 72 inside Blinker's code),
% the system will automatically execute both:
%
%   • getBlinkPositions (STEP 1bi)
%   • fitBlinks (STEP 1bii)
%
% Since the core code was originally written as one continuous block,
% we intentionally do not refactor it into separate smaller parts here.
% This preserves the original MATLAB execution flow, ensuring the output
% matches exactly what Blinker produces internally.
%
% -------------------------------------------------------------------------
% About this design decision
% -------------------------------------------------------------------------
% While it is possible to decompose the blink filtration process into
% independent steps such as:
%   • filterByBlinkAmpRatio (STEP 1biii)
%   • filterByGoodBlinkThreshold (STEP 1biv)
%   • filterByGoodRatio (STEP 1bv)
%   • selectBestCandidate (STEP 1bvi)
%
% doing so in MATLAB could risk subtle behavioral differences due to
% reordering or data propagation issues. To maintain 100% accuracy during
% validation, we re-run the entire `extractBlinks` logic as a single unit.
% This ensures that all dependent steps — including
% **GetSignalPosition** and **FitBlinks** — are evaluated exactly as
% Blinker originally intended.
%
% In the Python migration (pyblinker), this block will later be broken down
% into the individual substeps listed above. This function therefore serves
% as the *gold-standard reference* for the MATLAB side, ensuring a
% byte-for-byte comparison between MATLAB and Python results.
%
% -------------------------------------------------------------------------
% Execution Overview
% -------------------------------------------------------------------------
% 1. Resolve paths relative to this file (portable, no hard-coded paths)
% 2. Optionally load `config.m` (overrides defaults if present)
% 3. Initialize EEGLAB silently (nogui) if available
% 4. Build input/output filenames (config paths override defaults)
% 5. Load input `.mat` data (candidateSignals, params, signalType)
% 6. Run the full `extractBlinks(...)` flow (covers STEP 1bi–1bvi)
% 7. Compare the computed result with the MATLAB gold-standard output
%
% -------------------------------------------------------------------------
% Purpose for migration testing
% -------------------------------------------------------------------------
% This script acts as the **reference baseline** for validating the Python
% `pyblinker` implementation. Its outputs represent the expected MATLAB
% ground truth, which should match the Python port when using equivalent
% parameters and logic.
%
% Recommended filename:
%   process_fit_blinks.m
% -------------------------------------------------------------------------

    % ---------------------------------------------------------------------
    % 1. Resolve paths & config via shared helper
    % ---------------------------------------------------------------------
    paths = sharedMigrationPaths(struct( ...
        'DataDirCandidates', {{'migration_data_dir', 'main_folder'}}, ...
        'OutputDirCandidates', {{'main_folder'}}, ...
        'EnsureOutputDir', true));

    data_dir = paths.data_dir;
    output_dir = paths.output_dir;
    config_vars = paths.config_vars;

    % ---------------------------------------------------------------------
    % 3. Initialize EEGLAB silently (if path known)
    % ---------------------------------------------------------------------
    eeglab_path = normalize_config_path(config_vars, 'eeglab_path');
    if ~isempty(eeglab_path) && isfolder(eeglab_path)
        addpath(genpath(eeglab_path));
        eeglab nogui;
    else
        try
            eeglab nogui;
        catch
            error(['EEGLAB could not be started. ' ...
                'Add EEGLAB to your MATLAB path or define eeglab_path in config.m']);
        end
    end

    % ---------------------------------------------------------------------
    % 4. Build input and output filenames
    % ---------------------------------------------------------------------
    gold_output_file = fullfile(data_dir, ...
        'step1biii_data_output_process_step_FilterBlink.mat');
    input_file = fullfile(data_dir, ...
        'step1biii_data_input_process_step_FilterBlink.mat');

    assert(isfile(input_file),  'Input .mat not found: %s', input_file);
    assert(isfile(gold_output_file), 'Gold .mat not found: %s', gold_output_file);

    % ---------------------------------------------------------------------
    % 5. Load input data
    % ---------------------------------------------------------------------
    input_data = loadMigrationFixture(input_file, ...
        {'candidateSignals', 'params', 'signalType'}, 'STEP 1biii input fixture');
    candidateSignals = input_data.candidateSignals;
    params           = input_data.params;
    signalType       = input_data.signalType;

    expected_data = loadMigrationFixture(gold_output_file, {'blinks'}, ...
        'STEP 1biii expected fixture');
    expected_blinks = expected_data.blinks;

    % ---------------------------------------------------------------------
    % 6. Run the full extractBlinks flow (STEP 1bi–1bvi)
    % ---------------------------------------------------------------------
    [blinks, params] = extractBlinks(candidateSignals, signalType, params); %#ok<ASGLU>

    % ---------------------------------------------------------------------
    % 7. Compare computed results with the reference gold output
    % ---------------------------------------------------------------------
    comparison = compareMigrationResults(blinks.signalData, expected_blinks.signalData, ...
        @compareblinkpropertiesstructure, 'Blink signal data');

    if comparison.isEqual
        fprintf('\nBlink structures match the MATLAB gold output ✅\n');
    else
        fprintf('\nBlink structures DO NOT match the MATLAB gold output ❌\n');
        if ~isempty(comparison.details)
            disp('Differences found:');
            disp(comparison.details);
        end
    end

    % Save computed output (for reference or Python comparison)
    computed_output_file = fullfile(output_dir, ...
        'process_fit_blinks_computed_output.mat');
    save(computed_output_file, 'blinks', 'params');
    fprintf('Computed output saved to: %s\n', computed_output_file);
end

function value = normalize_config_path(config_vars, field_name)
%NORMALIZE_CONFIG_PATH Extract a char path from config variables.
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
