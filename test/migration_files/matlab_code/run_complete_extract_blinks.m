function [blinks, params] = processBlinkComp()
% processBlinkComp loads blinkComp.mat (or your raw EEG mat), prepares params,
% and extracts blinks from blinkComp.

    % Load configuration (expects config.m to define main_folder, etc.)
    config;

    % Prepare parameters in a dedicated function
    params = prepareBlinkParams();

    % Load data and extract candidate signals
    input_file = fullfile(main_folder, 'ear_eog_raw_EEG_E8.mat');
    data = load(input_file);
    
    blinkComp = data.blinkComp;

    % Convert blinkComp to single precision
    candidateSignals = single(blinkComp);

    % Choose signal type (as used by extractBlinks)
    signalType = 'SignalNumbers';

    % Extract blinks
    [blinks, params] = extractBlinks(candidateSignals, signalType, params);
    signalData=blinks.signalData;
    output_file = fullfile(main_folder, 'step5_data_output_extract_blinks_rpb.mat');
    save(output_file, 'signalData', '-v7');

end


function params = prepareBlinkParams()
% prepareBlinkParams creates and returns the params struct (same fields + values)

    params = struct();

    params.srate = 100;
    params.stdThreshold = 1.5;

    params.subjectID = 'Subject1_Task1_Experiment1_Rep1';
    params.uniqueName = 'Unknown';
    params.experiment = 'Experiment1';
    params.task = 'Task1';

    params.startDate = '01-Jan-2016';
    params.startTime = '00:00:00';

    params.signalTypeIndicator = 'UseNumbers';
    params.signalNumbers = 1;
    params.signalLabels = {'002'};

    params.excludeLabels = {'exg5','exg6','exg7','exg8','vehicle position'};

    params.dumpBlinkerStructures = 0;
    params.showMaxDistribution = 1;
    params.dumpBlinkImages = 0;
    params.verbose = 1;
    params.dumpBlinkPositions = 0;

    params.fileName = '';  % empty char array

    params.blinkerSaveFile = 'C:\eeg_lab_matlab\eeglab2024.2\_blinks.mat';
    params.blinkerDumpDir  = 'C:\eeg_lab_matlab\eeglab2024.2\blinkDump';

    % params.lowCutoffHz = 1;
    % params.highCutoffHz = 20;

    params.minGoodBlinks = 10;

    params.blinkAmpRange = uint8([3 50]);

    params.goodRatioThreshold = 0.7;
    params.pAVRThreshold = 3;

    params.correlationThresholdTop    = 0.98;
    params.correlationThresholdBottom = 0.9;
    params.correlationThresholdMiddle = 0.95;

    params.keepSignals = 0;
    params.shutAmpFraction = 0.9;

    params.zThresholds = [0.9  2.0;
                          0.98 5.0];

    params.ICSimilarityThreshold = 0.85;
    params.ICFOMThreshold = 1;

    params.numberMaxBins = 80;
end
