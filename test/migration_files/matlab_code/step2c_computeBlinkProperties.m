function process_compute_blink_properties_step2c()
% PROCESS_COMPUTE_BLINK_PROPERTIES_STEP2C
% -------------------------------------------------------------------------
% Reference / gold-standard runner for **STEP 2c – computeBlinkProperties**
% in the Blinker pipeline.
%
% This script is the cleaned-up, portable version of the original:
%     \test\migration_files\matlab_code\step2c_computeBlinkProperties.m
%
% Its purpose is to:
%   1. Load the MATLAB **input fixture** for this step (the data that
%      computeBlinkProperties() needs),
%   2. Run our local / custom `computeBlinkProperties(...)`,
%   3. Load the MATLAB **expected output** (gold) for this step,
%   4. Compare current results vs the gold output, field by field,
%   5. Report differences so the Python port (pyblinker) can target the
%      same structure and numeric values.
%
% -------------------------------------------------------------------------
% Why this step matters in migration
% -------------------------------------------------------------------------
% After we have:
%   - detected blinks
%   - fit the blink waveforms (STEP 1bii)
% the pipeline typically derives **blink-level metrics** (amplitude,
% width, slopes, velocity-based features, etc.). That is what
% `computeBlinkProperties(...)` does.
%
% For a faithful MATLAB → Python migration we must be able to show:
%   "Given the same blinkFits, signalData, params, srate, blinkVelocity,
%    and peaks, MATLAB produces exactly *this* blinkProps, and Python
%    should produce the same."
%
% This script is the place where we enforce that.
%
% -------------------------------------------------------------------------
% What this script does (summary)
% -------------------------------------------------------------------------
% 1. Resolve paths (prefers config.m, otherwise uses project-relative paths)
% 2. Load **output** fixture (the gold/reference results from MATLAB)
% 3. Load **input** fixture (the data to feed into computeBlinkProperties)
% 4. Call `computeBlinkProperties(...)` from our local MATLAB code
% 5. Compare:
%       - blinkProps           vs blinkProps_output
%       - peaksPosVelZero      vs peaksPosVelZero_output
%       - peaksPosVelBase      vs peaksPosVelBase_output
% 6. Print comparison results in a test-friendly way
%
% -------------------------------------------------------------------------
% Recommended filename:
%   process_compute_blink_properties_step2c.m
% -------------------------------------------------------------------------

    % ---------------------------------------------------------------------
    % 1. Resolve paths & config via shared helper
    % ---------------------------------------------------------------------
    paths = sharedMigrationPaths(struct( ...
        'DataDirCandidates', {{'main_folder'}}, ...
        'OutputDirCandidates', {{'main_folder'}}, ...
        'EnsureOutputDir', true));

    data_dir = paths.data_dir;

    % ---------------------------------------------------------------------
    % 3. Define input/output fixture paths
    % ---------------------------------------------------------------------
    % NOTE:
    %   In the original script, the naming was a bit flipped:
    %     input_file  <- step2c_data_output_computeBlinkProperties.mat
    %     output_file <- step2c_data_input_computeBlinkProperties.mat
    %   That was slightly confusing. Here we keep the **same filenames** so
    %   we don't break existing fixtures, but we name our variables clearly.
    %
    %   - gold_file  : what MATLAB previously produced (authoritative)
    %   - input_file : what we need to FEED into computeBlinkProperties
    % ---------------------------------------------------------------------
    gold_file  = fullfile(data_dir, 'step2c_data_output_computeBlinkProperties.mat');
    input_file = fullfile(data_dir, 'step2c_data_input_computeBlinkProperties.mat');

    assert(isfile(gold_file),  'Gold/reference .mat not found: %s', gold_file);
    assert(isfile(input_file), 'Input .mat not found: %s', input_file);

    % ---------------------------------------------------------------------
    % 4. Load gold/reference data
    % ---------------------------------------------------------------------
    gold_data = loadMigrationFixture(gold_file, ...
        {'blinkProps', 'peaksPosVelZero', 'peaksPosVelBase'}, ...
        'STEP 2c expected fixture');
    blinkProps_gold        = gold_data.blinkProps;
    peaksPosVelZero_gold   = gold_data.peaksPosVelZero;
    peaksPosVelBase_gold   = gold_data.peaksPosVelBase;

    % ---------------------------------------------------------------------
    % 5. Load input data for computeBlinkProperties
    % ---------------------------------------------------------------------
    in_data = loadMigrationFixture(input_file, ...
        {'blinkFits', 'signalData', 'params', 'srate', 'blinkVelocity', 'peaks'}, ...
        'STEP 2c input fixture');
    blinkFits     = in_data.blinkFits;
    signalData    = in_data.signalData;
    params        = in_data.params;
    srate         = in_data.srate;
    blinkVelocity = in_data.blinkVelocity;
    peaks         = in_data.peaks;

    % ---------------------------------------------------------------------
    % 6. Run the actual function under test
    %    (local copy in test/migration_files/matlab_code/computeBlinkProperties.m)
    % ---------------------------------------------------------------------
    [blinkProps, peaksPosVelZero, peaksPosVelBase] = ...
        computeBlinkProperties(blinkFits, signalData, params, ...
                               srate, blinkVelocity, peaks);

    % ---------------------------------------------------------------------
    % 7. Compare structures / matrices against gold
    % ---------------------------------------------------------------------
    fprintf('\n--- STEP 2c: compare blinkProps ---\n');
    comparison_blinkProps = compareMigrationResults(blinkProps, blinkProps_gold, ...
        @compareblinkpropertiesstructure, 'blinkProps');

    if comparison_blinkProps.isEqual
        fprintf('blinkProps ✅ matches gold output\n');
    else
        fprintf('blinkProps ❌ does NOT match gold output\n');
        if ~isempty(comparison_blinkProps.details)
            disp('Differences:');
            disp(comparison_blinkProps.details);
        end
    end

    fprintf('\n--- STEP 2c: compare peaksPosVelZero ---\n');
    comparison_peaksPosVelZero = compareMigrationResults(peaksPosVelZero, ...
        peaksPosVelZero_gold, @matrix_comparator, 'peaksPosVelZero');

    if comparison_peaksPosVelZero.isEqual
        fprintf('peaksPosVelZero ✅ matches gold output\n');
    else
        fprintf('peaksPosVelZero ❌ does NOT match gold output\n');
        if ~isempty(comparison_peaksPosVelZero.details)
            disp(comparison_peaksPosVelZero.details);
        end
    end

    fprintf('\n--- STEP 2c: compare peaksPosVelBase ---\n');
    comparison_peaksPosVelBase = compareMigrationResults(peaksPosVelBase, ...
        peaksPosVelBase_gold, @matrix_comparator, 'peaksPosVelBase');

    if comparison_peaksPosVelBase.isEqual
        fprintf('peaksPosVelBase ✅ matches gold output\n');
    else
        fprintf('peaksPosVelBase ❌ does NOT match gold output\n');
        if ~isempty(comparison_peaksPosVelBase.details)
            disp(comparison_peaksPosVelBase.details);
        end
    end


    % ---------------------------------------------------------------------
    % 8. (Optional) Save computed results for inspection / future diffs
    % ---------------------------------------------------------------------
    % computed_out_file = fullfile(data_dir, ...
    %     'process_compute_blink_properties_step2c_computed_output.mat');
    % save(computed_out_file, ...
    %     'blinkProps', 'peaksPosVelZero', 'peaksPosVelBase');
    % fprintf('\nComputed results saved to: %s\n', computed_out_file);
end

function [is_equal, details] = matrix_comparator(actual, expected)
%MATRIX_COMPARATOR Wrap compare_matrices to produce boolean + detail output.
    details = compare_matrices(actual, expected);
    is_equal = strcmp(details.Status, 'Comparison Results') && isempty(details.Details);
end
