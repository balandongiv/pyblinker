function data = loadMigrationFixture(file_path, required_fields, description)
%LOADMIGRATIONFIXTURE Load a MAT fixture, validating its existence and fields.
%   data = loadMigrationFixture(file_path, required_fields, description)
%
% PARAMETERS
%   file_path        - Absolute or relative path to the MAT-file fixture.
%   required_fields  - Cell array (or string) of field names that must be
%                      present after loading the file. Optional.
%   description      - Short label used in error messages. Optional.
%
% RETURNS
%   data - Struct returned by MATLAB's `load` for the given fixture.
%
% This helper standardizes fixture loading across the migration scripts by
% providing consistent error messages and field validation.
%
% See also: loadMigrationFixturePair, compareMigrationResults
%
% Author: pyblinker migration helpers
%
    if nargin < 2 || isempty(required_fields)
        required_fields = {};
    end
    if nargin < 3 || isempty(description)
        description = 'fixture';
    end

    validateattributes(file_path, {'char', 'string'}, {'nonempty'}, mfilename, 'file_path');
    file_path = char(file_path);

    if exist(file_path, 'file') ~= 2
        error('loadMigrationFixture:MissingFile', ...
            '%s not found: %s', description, file_path);
    end

    data = load(file_path);

    if ischar(required_fields) || (isstring(required_fields) && isscalar(required_fields))
        required_fields = {char(required_fields)};
    end

    if ~isempty(required_fields)
        missing = setdiff(required_fields, fieldnames(data));
        if ~isempty(missing)
            error('loadMigrationFixture:MissingFields', ...
                '%s is missing required fields: %s', description, strjoin(missing, ', '));
        end
    end
end

function [input_data, expected_data, comparison] = loadMigrationFixturePair(input_path, expected_path, options)
%LOADMIGRATIONFIXTUREPAIR Load paired input/expected fixtures with validation.
%   [input_data, expected_data, comparison] = loadMigrationFixturePair(...)
%
% PARAMETERS (options struct)
%   InputFields    - Cell array of required fields for the input fixture.
%   ExpectedFields - Cell array of required fields for the expected fixture.
%   Comparator     - Function handle used to compare actual vs. expected
%                    values. Invoked as comparator(actual, expected).
%   ActualValue    - Value produced by the caller to compare against the
%                    expected fixture. When omitted, no comparison is run.
%   ExpectedValue  - Expected value (typically loaded from the fixture) used
%                    with the comparator.
%   ComparisonName - Description used in logs/error messages.
%   FailOnMismatch - Logical flag (default false). When true, a mismatch
%                    triggers an error.
%
% RETURNS
%   input_data    - Struct loaded from input_path.
%   expected_data - Struct loaded from expected_path.
%   comparison    - Struct summarizing comparison results (fields: description,
%                   isEqual, details). Empty when no comparator provided.

    if nargin < 3 || isempty(options)
        options = struct();
    end

    if ~isfield(options, 'InputFields'); options.InputFields = {}; end
    if ~isfield(options, 'ExpectedFields'); options.ExpectedFields = {}; end
    if ~isfield(options, 'ComparisonName'); options.ComparisonName = 'Fixture comparison'; end
    if ~isfield(options, 'FailOnMismatch'); options.FailOnMismatch = false; end

    input_data = loadMigrationFixture(input_path, options.InputFields, 'Input fixture');
    expected_data = loadMigrationFixture(expected_path, options.ExpectedFields, 'Expected fixture');

    comparison = struct();
    has_comparator = isfield(options, 'Comparator') && ~isempty(options.Comparator);
    has_values = isfield(options, 'ActualValue') && isfield(options, 'ExpectedValue');

    if has_comparator && has_values
        comparison = compareMigrationResults(options.ActualValue, options.ExpectedValue, ...
            options.Comparator, options.ComparisonName);

        if options.FailOnMismatch && ~comparison.isEqual
            error('loadMigrationFixturePair:ComparisonFailed', ...
                '%s mismatch when comparing fixtures.', options.ComparisonName);
        end
    end
end

function result = compareMigrationResults(actual, expected, comparator, description)
%COMPAREMIGRATIONRESULTS Standardize comparison handling across scripts.
%   result = compareMigrationResults(actual, expected, comparator, description)
%
% RETURNS
%   result - struct with fields:
%            • description - human readable label for the comparison.
%            • isEqual     - logical indicating comparison success.
%            • details     - comparator-specific details (if provided).

    if nargin < 3 || isempty(comparator)
        comparator = @(lhs, rhs) isequal(lhs, rhs);
    end
    if nargin < 4 || isempty(description)
        description = 'Comparison';
    end

    try
        [is_equal, details] = comparator(actual, expected);
    catch ME
        if contains(ME.message, 'Too many output arguments')
            is_equal = comparator(actual, expected);
            details = [];
        else
            rethrow(ME);
        end
    end

    result = struct('description', description, ...
                    'isEqual', logical(is_equal), ...
                    'details', details);
end
