function step3_selectChannels()
%STEP3_SELECTCHANNELS Validate channel selection against MATLAB fixtures.

    paths = sharedMigrationPaths(struct( ...
        'DataDirCandidates', {{'main_folder'}}, ...
        'UseOutputDir', false));

    data_dir = paths.data_dir;

    input_compact = loadMigrationFixture(fullfile(data_dir, ...
        'step3a_input_selectChannel_compact.mat'), ...
        {'signalData', 'params'}, 'STEP 3 compact input');
    signalData = input_compact.signalData;
    params = input_compact.params;

    blinks = processBlinkSignalsCompact(signalData, params);

    expected_data = loadMigrationFixture(fullfile(data_dir, ...
        'step3a_input_selectChannel.mat'), {'blinks'}, 'STEP 3 expected output');

    comparison = compareMigrationResults(blinks.signalData, ...
        expected_data.blinks.signalData, @compareblinkpropertiesstructure, ...
        'Selected signal data');

    if comparison.isEqual
        fprintf('Signal data structures match the MATLAB gold output ✅\n');
    else
        fprintf('Signal data structures DO NOT match the MATLAB gold output ❌\n');
        if ~isempty(comparison.details)
            disp(comparison.details);
        end
    end
end
