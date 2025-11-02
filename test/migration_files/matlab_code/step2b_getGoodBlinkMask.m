function step2b_getGoodBlinkMask()
%STEP2B_GETGOODBLINKMASK Validate the good blink mask computation against fixtures.
%
% This helper loads the STEP 2b migration fixtures, computes the good blink
% mask using `get_good_blink_mask`, and verifies the results against the
% stored MATLAB gold outputs.

    % Resolve configuration & directories via shared helper
    paths = sharedMigrationPaths(struct( ...
        'DataDirCandidates', {{'main_folder'}}, ...
        'UseOutputDir', false));

    data_dir = paths.data_dir;

    % Define file paths dynamically
    input_file = fullfile(data_dir, 'step2b_data_input_getGoodBlinkMask.mat');
    output_file = fullfile(data_dir, 'step2b_data_output_getGoodBlinkMask.mat');

    data = loadMigrationFixture(input_file, ...
        {'zThresholds', 'specifiedStd', 'specifiedMedian', 'blinkFits'}, ...
        'STEP 2b input fixture');
    zThresholds = data.zThresholds;
    specifiedStd = data.specifiedStd;
    specifiedMedian = data.specifiedMedian;
    blinkFits = data.blinkFits;

    [goodBlinkMask, specifiedMedian, specifiedStd] = ...
        get_good_blink_mask(blinkFits, specifiedMedian, specifiedStd, zThresholds);

    data_output = loadMigrationFixture(output_file, ...
        {'goodBlinkMask', 'specifiedMedian', 'specifiedStd'}, ...
        'STEP 2b expected fixture');

    goodBlinkMask_output = data_output.goodBlinkMask;
    specifiedMedian_output = data_output.specifiedMedian;
    specifiedStd_output = data_output.specifiedStd;

    comparison_mask = compareMigrationResults(goodBlinkMask, goodBlinkMask_output, ...
        @isequal, 'goodBlinkMask equality');
    fprintf('Good blink mask match: %d\n', comparison_mask.isEqual);

    comparison_median = compareMigrationResults(specifiedMedian, specifiedMedian_output, ...
        @isequal, 'specifiedMedian equality');
    fprintf('Specified median match: %d\n', comparison_median.isEqual);

    comparison_std = compareMigrationResults(specifiedStd, specifiedStd_output, ...
        @isequal, 'specifiedStd equality');
    fprintf('Specified std match: %d\n', comparison_std.isEqual);
end
