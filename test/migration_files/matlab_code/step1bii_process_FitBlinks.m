function process_fit_blinks_step1bii()
% PROCESS_FIT_BLINKS_STEP1BII
% -------------------------------------------------------------------------
% Reference implementation for the **FitBlinks** stage (STEP 1bii)
% in the Blinker pipeline.
%
% This function reproduces the MATLAB gold-standard behavior for fitting
% detected blinks to their temporal waveforms. It validates the MATLAB
% output from the `fitBlinks()` function and compares it with a stored
% gold-standard reference file.
%
% -------------------------------------------------------------------------
% Background and design note
% -------------------------------------------------------------------------
% In the original Blinker implementation, the FitBlinks step is embedded in
% the `extractBlinksEEG` workflow. For clarity and migration testing, this
% function isolates that logic into a standalone script while maintaining
% identical behavior to the original.
%
% The function:
%   1. Loads configuration and resolves project paths.
%   2. Loads input/output `.mat` files (MATLAB gold data).
%   3. Runs the `fitBlinks()` function using default parameters.
%   4. Compares the generated blink fit structure against the reference.
%   5. Optionally plots one blink for visual verification.
%
% -------------------------------------------------------------------------
% Migration testing purpose
% -------------------------------------------------------------------------
% This script serves as the **gold-standard reference** for validating the
% Python `pyblinker` port of the `fitBlinks()` step (STEP 1bii).
% Its results provide the authoritative baseline for regression testing.
%
% -------------------------------------------------------------------------
% Execution overview
% -------------------------------------------------------------------------
% 1. Resolve paths relative to this file
% 2. Optionally load `config.m` (overrides defaults)
% 3. Load input `.mat` file (candidateSignal, blinkPositions)
% 4. Compute blink fits via `fitBlinks(candidateSignal, blinkPositions)`
% 5. Compare with MATLAB gold-standard output
% 6. Call `plot_blink_fit_example()` for debugging visualization
%
% Recommended filename:
%   process_fit_blinks_step1bii.m
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

    % ---------------------------------------------------------------------
    % 3. Define input/output file paths
    % ---------------------------------------------------------------------
    input_file  = fullfile(output_dir, 'step1bii_data_input_process_FitBlinks.mat');
    output_file = fullfile(output_dir, 'step1bii_data_output_process_FitBlinks.mat');

    assert(isfile(input_file),  'Input .mat file not found: %s', input_file);
    assert(isfile(output_file), 'Output .mat file not found: %s', output_file);

    % ---------------------------------------------------------------------
    % 4. Load data and run FitBlinks
    % ---------------------------------------------------------------------
    input_data = loadMigrationFixture(input_file, ...
        {'candidateSignal', 'blinkPositions'}, 'STEP 1bii input fixture');
    candidateSignal = input_data.candidateSignal;
    blinkPositions  = input_data.blinkPositions;

    % Compute blink fits using MATLAB's gold-standard function
    blinkFits = fitBlinks(candidateSignal, blinkPositions);

    % Load MATLAB gold output
    expected_data = loadMigrationFixture(output_file, {'blinkFits'}, ...
        'STEP 1bii expected fixture');
    blinkFits_expected = expected_data.blinkFits;

    % ---------------------------------------------------------------------
    % 5. Compare computed results with the reference gold output
    % ---------------------------------------------------------------------
    comparison = compareMigrationResults(blinkFits, blinkFits_expected, ...
        @compareblinkpropertiesstructure, 'Blink fit structures');

    if comparison.isEqual
        fprintf('\nBlink fit structures match the MATLAB gold output ✅\n');
    else
        fprintf('\nBlink fit structures DO NOT match the MATLAB gold output ❌\n');
        if ~isempty(comparison.details)
            disp('Differences found:');
            disp(comparison.details);
        end
    end

    % ---------------------------------------------------------------------
    % 6. Optional: Plot example blink (for debugging / visual check)
    % ---------------------------------------------------------------------
    blinkIndex = 3; % Example blink to visualize
    if numel(blinkFits) >= blinkIndex
        plot_blink_fit_example(candidateSignal, blinkFits, blinkIndex);
    else
        fprintf('⚠️ Not enough blink fits to plot (requested #%d)\n', blinkIndex);
    end

    fprintf('\nProcessing complete. Comparison and (optional) plot generated.\n');
end


% -------------------------------------------------------------------------
% Helper Function: plot_blink_fit_example
% -------------------------------------------------------------------------
function plot_blink_fit_example(candidateSignal, blinkFits, blinkIndex)
% PLOT_BLINK_FIT_EXAMPLE
% -------------------------------------------------------------------------
% Utility function for visualizing the blink waveform and fitted points
% generated by `fitBlinks()`. This function is separated for easier
% debugging and reuse during migration testing.
%
% Arguments:
%   candidateSignal : vector of EEG signal data containing blink
%   blinkFits       : struct array returned by fitBlinks()
%   blinkIndex      : integer index of blink to visualize
% -------------------------------------------------------------------------

    assert(blinkIndex <= numel(blinkFits), ...
        'Blink index exceeds available blinks.');

    bf = blinkFits(blinkIndex);

    % Retrieve key feature frames
    maxFrame  = bf.maxFrame;
    leftOuter = bf.leftOuter;  rightOuter = bf.rightOuter;
    leftZero  = bf.leftZero;   rightZero  = bf.rightZero;
    leftBase  = bf.leftBase;   rightBase  = bf.rightBase;

    % Extract time range and data segment
    timeRange    = leftOuter-10 : rightOuter+10;
    blinkSegment = candidateSignal(timeRange);

    % --- Plot ---
    figure('Name', sprintf('Blink Fit Visualization #%d', blinkIndex), ...
           'Color', 'w');
    plot(timeRange, blinkSegment, 'LineWidth', 1.5, 'Color', [0 0 0 0.4]);
    hold on;

    % Annotate all key frames
    plot(maxFrame, candidateSignal(maxFrame), 'ro', 'MarkerSize', 8);
    text(maxFrame, candidateSignal(maxFrame), ' Max Frame', 'VerticalAlignment', 'top');

    plot([leftOuter, rightOuter], candidateSignal([leftOuter, rightOuter]), ...
        'go', 'MarkerSize', 8);
    text(leftOuter, candidateSignal(leftOuter), ' Left Outer', 'VerticalAlignment', 'bottom');
    text(rightOuter, candidateSignal(rightOuter), ' Right Outer', 'VerticalAlignment', 'bottom');

    plot([leftZero, rightZero], candidateSignal([leftZero, rightZero]), ...
        'mo', 'MarkerSize', 8);
    text(leftZero, candidateSignal(leftZero), ' Left Zero', 'VerticalAlignment', 'top');
    text(rightZero, candidateSignal(rightZero), ' Right Zero', 'VerticalAlignment', 'top');

    plot([leftBase, rightBase], candidateSignal([leftBase, rightBase]), ...
        'co', 'MarkerSize', 8);
    text(leftBase, candidateSignal(leftBase), ' Left Base', 'VerticalAlignment', 'bottom');
    text(rightBase, candidateSignal(rightBase), ' Right Base', 'VerticalAlignment', 'bottom');

    % Scatter points for clarity
    scatter(timeRange, blinkSegment, 15, 'b', 'filled');

    % Labels and aesthetics
    xlabel('Frame Index');
    ylabel('Blink Amplitude');
    title(sprintf('Blink Fit Visualization (Blink #%d)', blinkIndex));
    legend({'Signal', 'Feature Points'}, 'Location', 'best');
    grid on;
    hold off;

    fprintf('Plotted blink fit visualization for Blink #%d.\n', blinkIndex);
end
