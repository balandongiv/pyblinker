function process_extract_blink_statistics_step3()
% PROCESS_EXTRACT_BLINK_STATISTICS_STEP3
% -------------------------------------------------------------------------
% Reference / gold-standard runner for **STEP 3 – extractBlinkStatistics**
% in the Blinker pipeline.
%
% Purpose:
%   After we have:
%     - detected and fitted blinks (STEP 1x),
%     - computed blink properties (STEP 2x),
%     - and possibly applied restrictions (e.g. PAVR, STEP 2d),
%   we finally want to **summarize** everything into a statistics
%   structure/table. That is what `extractBlinkStatistics(...)` does.
%
% This script loads the prepared MATLAB fixture for STEP 3, calls the
% actual `extractBlinkStatistics(...)` function, and exports the result to
% a table (Excel) so we can easily inspect the output or compare it with
% the Python port (`pyblinker`).
%
% What this script does:
%   1. Resolve paths relative to this file
%   2. Optionally load `config.m` so project-specific folders are used
%   3. Initialize EEGLAB silently (if available)
%   4. Load the STEP 3 input fixture:
%        step3_data_input_extractBlinkStatistic.mat
%      which should contain:
%        - blinks
%        - blinkFits
%        - blinkProperties
%        - params
%   5. Call:
%        blinkStatistics = extractBlinkStatistics(...)
%   6. Convert to a table and write to `blinkStatistics.xlsx` (for human
%      inspection / CI artifact)
%
% Recommended filename:
%   process_extract_blink_statistics_step3.m
% -------------------------------------------------------------------------

    % ---------------------------------------------------------------------
    % 1. Resolve paths & config via shared helper
    % ---------------------------------------------------------------------
    paths = sharedMigrationPaths(struct( ...
        'DataDirCandidates', {{'main_folder'}}, ...
        'OutputDirCandidates', {{'main_folder'}}, ...
        'EnsureOutputDir', true));

    data_dir = paths.data_dir;
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
    % 4. Build input file path and load data
    % ---------------------------------------------------------------------
    input_file = fullfile(data_dir, 'step3_data_input_extractBlinkStatistic.mat');
    assert(isfile(input_file), 'Input .mat not found: %s', input_file);

    in_data = loadMigrationFixture(input_file, ...
        {'blinks', 'blinkFits', 'blinkProperties', 'params'}, ...
        'STEP 3 input fixture');
    blinks           = in_data.blinks;
    blinkFits        = in_data.blinkFits;
    blinkProperties  = in_data.blinkProperties;
    params           = in_data.params;

    % ---------------------------------------------------------------------
    % 5. Run the actual function under test
    % ---------------------------------------------------------------------
    blinkStatistics = extractBlinkStatistics( ...
        blinks, blinkFits, blinkProperties, params)

    % ---------------------------------------------------------------------
    % 6. Export to table / Excel for inspection
    % ---------------------------------------------------------------------
    blinkTable = struct2table(blinkStatistics, 'AsArray', true);

    % write into the same (or closest) output dir, not random CWD
    xlsx_file = fullfile(data_dir, 'blinkStatistics.xlsx');
    writetable(blinkTable, xlsx_file);

    fprintf('Blink statistics extracted and written to:\n  %s\n', xlsx_file);
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
