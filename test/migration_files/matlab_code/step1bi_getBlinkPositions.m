%% Step 1: Process EEG Channel 003
%
% This step isolates and processes EEG data for channel **003** to create
% the MATLAB "gold standard" reference for blink detection validation.
%
% -------------------------------------------------------------------------
% **Purpose**
% -------------------------------------------------------------------------
% To reduce computation time, this step focuses on channel **003** only.
% The data for this channel are extracted from a previously processed
% section and include the following variables:
%
%   - **blinkComp** : [1 × 27800 double] — extracted blink component signal  
%   - **srate**     : [scalar = 100 Hz] — sampling rate of the signal  
%   - **stdThreshold** : [scalar = 1.5] — standard deviation threshold for
%                        blink detection
%
% The blink signal (`blinkComp`) is obtained after applying a band-pass
% filter using the EEGLAB function:
%
%   ```matlab
%   EEG1 = pop_eegfiltnew(EEG1, params.lowCutoffHz, params.highCutoffHz);
%   ```
%
% The filtered data are then passed to Blinker's internal function
% `getCandidateSignals`, which is subsequently used within
% `extractBlinks` to identify candidate blink events.
%
% -------------------------------------------------------------------------
% **Validation Context**
% -------------------------------------------------------------------------
% This script establishes MATLAB's output as the **ground-truth reference**
% for cross-language validation. The same input data will be used in
% the Python implementation (PyBlinker), allowing for direct comparison
% of results between MATLAB and Python frameworks.
%
% The comparison ensures that the Python implementation accurately
% reproduces MATLAB's behavior for:
%   - Filtering (`pop_eegfiltnew`)
%   - Blink component extraction and candidate signal generation
%   - Threshold-based blink detection
%
% -------------------------------------------------------------------------
% **Summary**
% -------------------------------------------------------------------------
% Step 1: Load and process EEG channel 003  
% Step 2: Generate blink component (`blinkComp`)  
% Step 3: Use the MATLAB output as gold-standard reference for Python validation
%
% -------------------------------------------------------------------------
% **Example**
% -------------------------------------------------------------------------
% ```matlab
% % Example usage:
% EEG1 = pop_eegfiltnew(EEG1, params.lowCutoffHz, params.highCutoffHz);
% blinkComp = getCandidateSignals(EEG1, 'channel', '003');
% ```


function step1bi_getBlinkPositions()
    % processBlinkComp loads blinkComp.mat and processes blink positions
    % Resolve configuration & data directories via shared helper
    paths = sharedMigrationPaths(struct( ...
        'DataDirCandidates', {{'main_folder'}}, ...
        'UseOutputDir', false));

    data_dir = paths.data_dir;

    % Define file paths dynamically
    input_file = fullfile(data_dir, 'step1bi_data_input_getBlinkPositions.mat');
    output_file = fullfile(data_dir, 'step1bi_data_output_getBlinkPositions.mat');

    % Load the input data
    data = loadMigrationFixture(input_file, ...
        {'blinkComp', 'srate', 'stdThreshold'}, 'STEP 1bi input fixture');
    
    % Ensure that the required variables are present
    if ~isfield(data, 'blinkComp') || ~isfield(data, 'srate') || ~isfield(data, 'stdThreshold')
        error('The file blinkComp.mat must contain blinkComp, srate, and stdThreshold variables.');
    end
    
    % Extract variables
    blinkComp = data.blinkComp;
    % Convert blinkComp to single precision
    blinkComp = single(blinkComp);
    srate = data.srate;
    srate = single(srate);
    stdThreshold = data.stdThreshold;
    
    % Note: There is a discrepancy in length (4) between `blinkPositions` (83) and `blinkPositions_output` (79).
    % However, the `startBlinks` and `endBlinks` pairs are consistent across the 73 entries in both arrays.
    % This difference is likely due to rounding errors introduced when saving the time series data 
    % in `step1bi_data_input_getBlinkPositionskComp.mat`.
    % We'll cross-check the results in Python to verify consistency between MATLAB and Python outputs.
    % Ensure that corresponding pairs match across both platforms wherever they exist.


    % Call the getBlinkPositions function
    blinkPositions = getBlinkPositions(blinkComp, srate, stdThreshold); % 79 (using output) vs 83 (using input)
    data_output = loadMigrationFixture(output_file, {'blinkPositions'}, ...
        'STEP 1bi expected fixture');
    blinkPositions_output = data_output.blinkPositions; % This is the data output if we run the code from the beginning of pop_blinker
    % Read the note above about unequal length of  the data input and data
    % output
    % Display the blink positions
    disp('Blink Positions (Start Frame, End Frame):');
    disp(blinkPositions');
    
    % Optionally, plot the blinkComp signal and mark the detected blinks
    % plotBlinkPositions(blinkComp, srate, blinkPositions);
    blinkIndex=2;
    % Call plotSingleBlink to visualize the specified blink
    % Call plotSingleBlink to visualize the specified blink with threshold
    plotSingleBlink(blinkComp, srate, blinkPositions, blinkIndex, stdThreshold);

end

function plotBlinkPositions(blinkComp, srate, blinkPositions)
    % plotBlinkPositions plots the blinkComp signal and highlights detected blinks
    
    % Time vector
    t = (0:length(blinkComp)-1) / srate;
    
    % Plot the blinkComp signal
    figure;
    plot(t, blinkComp);
    hold on;
    
    % Highlight the detected blinks
    for i = 1:size(blinkPositions, 2)
        startIdx = blinkPositions(1, i);
        endIdx = blinkPositions(2, i);
        patch([t(startIdx) t(endIdx) t(endIdx) t(startIdx)], ...
              [min(blinkComp) min(blinkComp) max(blinkComp) max(blinkComp)], ...
              'red', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
    
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Blink Component with Detected Blinks');
    legend('Blink Component', 'Detected Blinks');
    hold off;
end

function plotSingleBlink(blinkComp, srate, blinkPositions, blinkIndex, stdThreshold)
    % plotSingleBlink plots a specific blink in blinkComp based on the given index,
    % including 10 points of context on each side and a horizontal line for stdThreshold.
    %
    % Parameters:
    %   blinkComp     - Single precision blink component signal.
    %   srate         - Sampling rate (Hz).
    %   blinkPositions - 2 x n array with start and end frames of detected blinks.
    %   blinkIndex    - Index of the blink to plot (1-based index).
    %   stdThreshold  - Number of standard deviations above the mean for blink detection.
    
    % Validate the blink index
    if blinkIndex < 1 || blinkIndex > size(blinkPositions, 2)
        error('Invalid blink index. Please enter an index between 1 and %d.', size(blinkPositions, 2));
    end
    
    % Calculate the threshold value based on mean and robust std deviation
    mu = mean(blinkComp); 
    robustStdDev = 1.4826 * mad(blinkComp, 1);
    threshold = mu + stdThreshold * robustStdDev;
    
    % Extract start and end indices for the specified blink
    startIdx = blinkPositions(1, blinkIndex);
    endIdx = blinkPositions(2, blinkIndex);
    
    % Determine the range of points to display, adding 10 points before and after
    displayStart = max(1, startIdx - 10); % Ensure we don’t go below index 1
    displayEnd = min(length(blinkComp), endIdx + 10); % Ensure we don’t exceed signal length
    
    % Time vector for the selected range
    t = (displayStart:displayEnd) / srate;
    
    % Plot the specific blink with additional points for context
    figure;
    plot(t, blinkComp(displayStart:displayEnd), 'b', 'LineWidth', 1.5);
    hold on;
    
    % Highlight the blink region
    blinkTime = (startIdx:endIdx) / srate;
    patch([blinkTime(1) blinkTime(end) blinkTime(end) blinkTime(1)], ...
          [min(blinkComp(displayStart:displayEnd)) min(blinkComp(displayStart:displayEnd)) ...
           max(blinkComp(displayStart:displayEnd)) max(blinkComp(displayStart:displayEnd))], ...
          'red', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    
    % Label the start and end of the blink
    text(blinkTime(1), blinkComp(startIdx), 'Start', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    text(blinkTime(end), blinkComp(endIdx), 'End', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
    
    % Add horizontal line for threshold
    yline(threshold, '--', sprintf('Threshold: %.2f', threshold), 'Color', 'k', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
    
    % Labeling the plot
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('Single Blink at Index %d with Context', blinkIndex));
    legend('Blink Signal', 'Blink Region');
    hold off;
end


