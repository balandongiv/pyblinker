function paths = sharedMigrationPaths(options)
% SHAREDMIGRATIONPATHS Centralize shared path/config loading logic for migration tests.
%
% This utility factors the repeated MATLAB setup code that existed in:
%   - step1bii_process_FitBlinks.m
%   - step1biii_FilterBlink.m
%   - step2c_computeBlinkProperties.m
%   - step2d_applyPAVRRestriction.m
%   - step3_extractblinkstat.m
%   - process_eeg_with_pop_blinker.m
%
% In addition it now supports callers that previously executed `config.m`
% directly (step0_pop_blinker, step1bi_getBlinkPositions, step2b_getGoodBlinkMask,
% step3_selectChannels) without leaking variables into the caller workspace.
%
% CALL SIGNATURE
%   paths = sharedMigrationPaths(options)
%
% PARAMETERS
%   options (struct, optional) supports the following fields:
%     • DefaultDataSubfolder   - Subfolder (relative to repo root) used when
%                                no config override is found. Default: 'migration_files'.
%     • DefaultOutputSubfolder - Subfolder for output artifacts. Defaults to
%                                DefaultDataSubfolder when omitted.
%     • ConfigFileName         - Name of the config script. Default: 'config.m'.
%     • DataDirCandidates      - Cell array of variable names (from config.m)
%                                to probe for data directory overrides. Default:
%                                {'main_folder'}.
%     • OutputDirCandidates    - Cell array of variable names to probe for
%                                output directory overrides. Default:
%                                {'main_folder'}.
%     • EnsureOutputDir        - Logical flag indicating whether the resolved
%                                output directory should be created when it does
%                                not exist. Default: true.
%     • EnsureDataDir          - Logical flag to create the resolved data
%                                directory when missing. Default: false.
%     • UseOutputDir           - Logical flag that controls whether an output
%                                directory should be resolved at all. Default: true.
%     • ProjectRootOverride    - Optional absolute path that bypasses automatic
%                                repository-root discovery when provided.
%     • ProjectRootMarkers     - Cell array of sentinel files/folders used when
%                                inferring the project root. Default:
%                                {'.git', 'pyproject.toml', 'setup.py'}.
%
% RETURNS
%   paths (struct) containing:
%     • caller_file            - Absolute path of the invoking MATLAB file.
%     • caller_dir             - Directory containing the caller.
%     • project_root           - Repository root inferred from caller path.
%     • config_file            - Absolute path to the config script (if it exists).
%     • config_loaded          - True when config.m was executed.
%     • config_vars            - Struct of variables introduced by config.m.
%     • data_dir_default       - Default migration data directory.
%     • output_dir_default     - Default migration output directory.
%     • data_dir               - Final data directory to use.
%     • output_dir             - Final output directory ('' when UseOutputDir=false).
%     • data_dir_source        - Config variable name supplying the data dir
%                                override, empty when defaults were used.
%     • output_dir_source      - Config variable name supplying the output dir
%                                override.
%
% EXTENSIBILITY
%   Additional migration scripts can call this helper with a tailored
%   options struct instead of duplicating the boilerplate for resolving
%   project-relative paths, loading config.m, and provisioning output
%   directories. Callers should read configuration values from
%   `paths.config_vars` rather than expecting those variables to exist in
%   their workspace.
%
% Example:
%   paths = sharedMigrationPaths(struct(
%       'DataDirCandidates', {{'migration_data_dir', 'main_folder'}}, ...
%       'OutputDirCandidates', {{'main_folder'}}, ...
%       'EnsureOutputDir', true));
%
%   data_dir = paths.data_dir;
%   output_dir = paths.output_dir;
%   config    = paths.config_vars;
%
% Author: pyblinker migration helpers

    if nargin < 1 || isempty(options)
        options = struct();
    end

    if ~isfield(options, 'DefaultDataSubfolder')
        options.DefaultDataSubfolder = 'migration_files';
    end
    if ~isfield(options, 'DefaultOutputSubfolder')
        options.DefaultOutputSubfolder = options.DefaultDataSubfolder;
    end
    if ~isfield(options, 'ConfigFileName')
        options.ConfigFileName = 'config.m';
    end
    if ~isfield(options, 'DataDirCandidates')
        options.DataDirCandidates = {'main_folder'};
    end
    if ~isfield(options, 'OutputDirCandidates')
        options.OutputDirCandidates = {'main_folder'};
    end
    if ~isfield(options, 'EnsureOutputDir')
        options.EnsureOutputDir = true;
    end
    if ~isfield(options, 'EnsureDataDir')
        options.EnsureDataDir = false;
    end
    if ~isfield(options, 'UseOutputDir')
        options.UseOutputDir = true;
    end
    if ~isfield(options, 'ProjectRootMarkers') || isempty(options.ProjectRootMarkers)
        options.ProjectRootMarkers = {'.git', 'pyproject.toml', 'setup.py'};
    end

    stack = dbstack('-completenames');
    if numel(stack) < 2
        error('sharedMigrationPaths:InvalidCaller', ...
              'sharedMigrationPaths must be invoked from another function or script.');
    end
    caller_file = stack(2).file;
    caller_dir = fileparts(caller_file);

    if isfield(options, 'ProjectRootOverride') && ~isempty(options.ProjectRootOverride)
        project_root = options.ProjectRootOverride;
    else
        project_root = locate_project_root(caller_dir, options.ProjectRootMarkers);
    end

    data_dir_default = fullfile(project_root, options.DefaultDataSubfolder);
    output_dir_default = fullfile(project_root, options.DefaultOutputSubfolder);

    config_file = fullfile(caller_dir, options.ConfigFileName);
    config_vars = struct();
    config_loaded = false;

    if exist(config_file, 'file') == 2
        config_vars = execute_config_script(config_file);
        config_loaded = true;
    end

    [data_dir, data_dir_source] = resolve_dir(config_vars, options.DataDirCandidates, ...
                                              data_dir_default, options.EnsureDataDir);

    if options.UseOutputDir
        [output_dir, output_dir_source] = resolve_dir(config_vars, options.OutputDirCandidates, ...
                                                      output_dir_default, options.EnsureOutputDir);
    else
        output_dir = '';
        output_dir_source = '';
    end

    paths = struct();
    paths.caller_file        = caller_file;
    paths.caller_dir         = caller_dir;
    paths.project_root       = project_root;
    paths.config_file        = config_file;
    paths.config_loaded      = config_loaded;
    paths.config_vars        = config_vars;
    paths.data_dir_default   = data_dir_default;
    paths.output_dir_default = output_dir_default;
    paths.data_dir           = data_dir;
    paths.output_dir         = output_dir;
    paths.data_dir_source    = data_dir_source;
    paths.output_dir_source  = output_dir_source;
end

function project_root = locate_project_root(start_dir, markers)
%LOCATE_PROJECT_ROOT Walk upwards from the caller directory until a sentinel is found.
    current = start_dir;
    project_root = start_dir;

    while true
        if any(cellfun(@(marker) path_exists(current, marker), markers))
            project_root = current;
            return;
        end

        parent = fileparts(current);
        if isempty(parent) || strcmp(parent, current)
            % Reached filesystem root; fall back to the original start dir.
            project_root = start_dir;
            return;
        end
        current = parent;
    end
end

function tf = path_exists(root_dir, marker)
    marker_path = fullfile(root_dir, marker);
    tf = (exist(marker_path, 'file') == 2) || (exist(marker_path, 'dir') == 7);
end

function config_vars = execute_config_script(config_file)
%EXECUTE_CONFIG_SCRIPT Run config.m inside an isolated workspace and capture variables.
    runner = @() run(config_file);
    config_vars = capture_script_workspace(runner, {'config_file', 'config_vars', 'runner'});
end

function captured = capture_script_workspace(script_runner, skip_names)
%CAPTURE_SCRIPT_WORKSPACE Execute a script and return variables as a struct.
    if nargin < 2
        skip_names = {};
    end

    script_runner();

    workspace_vars = whos();
    captured = struct();
    skip_all = [skip_names(:); {'script_runner', 'skip_names', ...
                                'workspace_vars', 'idx', 'name', 'captured'}];
    for idx = 1:numel(workspace_vars)
        name = workspace_vars(idx).name;
        if any(strcmp(name, skip_all))
            continue;
        end
        captured.(name) = eval(name);
    end
end

function [dir_path, source_name] = resolve_dir(config_vars, candidates, default_dir, create_if_missing)
%RESOLVE_DIR Select a directory path from config-defined candidates.
    dir_path = '';
    source_name = '';

    if ~iscell(candidates)
        candidates = {candidates};
    end

    for idx = 1:numel(candidates)
        candidate = candidates{idx};
        if isempty(candidate)
            continue;
        end

        if ~(ischar(candidate) || (isstring(candidate) && isscalar(candidate)))
            continue;
        end
        candidate = char(candidate);

        if isfield(config_vars, candidate)
            value = config_vars.(candidate);
            potential_dir = coerce_to_char(value);
            if ~isempty(potential_dir)
                dir_path = potential_dir;
                source_name = candidate;
                break;
            end
        end
    end

    if isempty(dir_path)
        dir_path = default_dir;
    end

    if create_if_missing && ~isfolder(dir_path)
        mkdir(dir_path);
    end
end

function value = coerce_to_char(candidate)
%COERCE_TO_CHAR Convert strings/cells to a usable character vector path.
    value = '';

    if ischar(candidate)
        value = candidate;
    elseif isa(candidate, 'string') && isscalar(candidate)
        value = char(candidate);
    elseif iscell(candidate) && ~isempty(candidate)
        value = coerce_to_char(candidate{1});
    end

    if ~(ischar(value) || (isa(value, 'string') && isscalar(value)))
        value = '';
    end
    if isa(value, 'string')
        value = char(value);
    end
end
