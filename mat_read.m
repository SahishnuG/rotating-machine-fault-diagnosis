% convert_mat_folders_to_csv.m
% Usage: place this file in the parent folder that contains 'acoustic' and 'vibration' folders,
% then run: convert_mat_folders_to_csv
%
% This script:
%  - loops through acoustic and vibration folders
%  - reads the Signal struct in each .mat
%  - constructs Time Stamp using x_values.start_value and x_values.increment
%  - writes CSV files with column names and units described in your message
%  - attempts to parse filename metadata: aaaaNm_bbbb_cccc.mat -> load, condition, severity

function convert_mat_folders_to_csv()
    acoustic_dir = fullfile(pwd, 'acoustic');
    vibration_dir = fullfile(pwd, 'vibration');

    if ~isfolder(acoustic_dir)
        warning('acoustic folder not found at %s', acoustic_dir);
    end
    if ~isfolder(vibration_dir)
        warning('vibration folder not found at %s', vibration_dir);
    end

    fprintf('Processing acoustic folder: %s\n', acoustic_dir);
    process_acoustic_folder(acoustic_dir);

    fprintf('Processing vibration folder: %s\n', vibration_dir);
    process_vibration_folder(vibration_dir);

    fprintf('Done.\n');
end

function process_acoustic_folder(folder)
    if ~isfolder(folder), return; end
    files = dir(fullfile(folder, '*.mat'));
    for k = 1:numel(files)
        fpath = fullfile(folder, files(k).name);
        try
            data = load(fpath);
            % Find the variable named 'Signal' (or first struct if different)
            if isfield(data,'Signal')
                S = data.Signal;
            else
                % try to recover first struct-like var
                vars = fieldnames(data);
                S = [];
                for i=1:numel(vars)
                    v = data.(vars{i});
                    if isstruct(v)
                        S = v; break;
                    end
                end
                if isempty(S)
                    warning('No struct found in %s — skipping', fpath);
                    continue;
                end
            end

            % read x_values and y_values
            xv = S.x_values;
            yv = S.y_values;

            start_val = getfield_safe(xv,'start_value');
            incr = getfield_safe(xv,'increment');
            n = getfield_safe(xv,'number_of_values');
            yvals = getfield_safe(yv,'values');

            % ensure column vector(s)
            if isvector(yvals)
                yvals = yvals(:);
            end

            % safety: if number_of_values is missing, derive from y length
            if isempty(n) && ~isempty(yvals)
                n = size(yvals,1);
            end

            if isempty(start_val) || isempty(incr) || isempty(n)
                error('Missing x_values metadata in %s', fpath);
            end

            t = start_val + (0:(n-1))' * incr; % column vector

            % create table: Time Stamp and values
            T = table;
            T.("Time Stamp") = t;
            T.values = yvals; % if yvals is Nx1

            % optional: add parsed filename metadata as columns
            meta = parse_filename_metadata(files(k).name);
            if ~isempty(meta)
                T.load = repmat({meta.load}, height(T), 1);
                T.condition = repmat({meta.condition}, height(T), 1);
                T.severity = repmat({meta.severity}, height(T), 1);
            end

            % write csv; same base name .csv
            outname = fullfile(folder, [strip_extension(files(k).name) '.csv']);
            writetable(T, outname);
            fprintf('Wrote %s (acoustic) -> %s\n', files(k).name, outname);
        catch ME
            warning('Failed to convert %s: %s', fpath, ME.message);
        end
    end
end

function process_vibration_folder(folder)
    if ~isfolder(folder), return; end
    files = dir(fullfile(folder, '*.mat'));
    for k = 1:numel(files)
        fpath = fullfile(folder, files(k).name);
        try
            data = load(fpath);
            if isfield(data,'Signal')
                S = data.Signal;
            else
                vars = fieldnames(data);
                S = [];
                for i=1:numel(vars)
                    v = data.(vars{i});
                    if isstruct(v)
                        S = v; break;
                    end
                end
                if isempty(S)
                    warning('No struct found in %s — skipping', fpath);
                    continue;
                end
            end

            xv = S.x_values;
            yv = S.y_values;

            start_val = getfield_safe(xv,'start_value');
            incr = getfield_safe(xv,'increment');
            n = getfield_safe(xv,'number_of_values');
            yvals = getfield_safe(yv,'values');

            if isempty(n) && ~isempty(yvals)
                n = size(yvals,1);
            end

            if isempty(start_val) || isempty(incr) || isempty(n)
                error('Missing x_values metadata in %s', fpath);
            end

            t = start_val + (0:(n-1))' * incr;

            % yvals may be Nx4 or NxM. As per your spec, expect 4 columns:
            % x_direction_housing_A, y_direction_housing_A, x_direction_housing_B, y_direction_housing_B
            if size(yvals,2) < 4
                warning('File %s: expected 4 columns for vibration but found %d. Will still write available columns.', files(k).name, size(yvals,2));
            end

            % Build table
            T = table;
            T.("Time Stamp") = t;
            % safe assign column-by-column
            colnames = {'x_direction_housing_A','y_direction_housing_A','x_direction_housing_B','y_direction_housing_B'};
            for c = 1:min(size(yvals,2),4)
                T.(colnames{c}) = yvals(:,c);
            end
            % if there are extra channels, name them channel_5, channel_6, ...
            if size(yvals,2) > 4
                for c = 5:size(yvals,2)
                    cname = sprintf('channel_%d', c);
                    T.(cname) = yvals(:,c);
                end
            end

            % Add filename metadata columns
            meta = parse_filename_metadata(files(k).name);
            if ~isempty(meta)
                T.load = repmat({meta.load}, height(T), 1);
                T.condition = repmat({meta.condition}, height(T), 1);
                T.severity = repmat({meta.severity}, height(T), 1);
            end

            % Write CSV
            outname = fullfile(folder, [strip_extension(files(k).name) '.csv']);
            writetable(T, outname);
            fprintf('Wrote %s (vibration) -> %s\n', files(k).name, outname);
        catch ME
            warning('Failed to convert %s: %s', fpath, ME.message);
        end
    end
end

%% ---- helper functions ----
function val = getfield_safe(s, fld)
    % robustly get a field whether it's nested as a struct or cell
    val = [];
    if isempty(s), return; end
    if isstruct(s) && isfield(s, fld)
        val = s.(fld);
        % if it's a 1x1 struct with nested field 'values' or numeric, try to unpack
        if isstruct(val) && isfield(val,'values')
            val = val.values;
        end
        % if it's a cell, convert
        if iscell(val)
            val = val{1};
        end
    else
        % sometimes fields appear as s(1).x_values etc.
        try
            if iscell(s) && ~isempty(s)
                s2 = s{1};
                if isstruct(s2) && isfield(s2,fld)
                    val = s2.(fld);
                    if isstruct(val) && isfield(val,'values')
                        val = val.values;
                    end
                    if iscell(val), val = val{1}; end
                end
            end
        catch
            % swallow
        end
    end
end

function name = strip_extension(fname)
    [~, name, ~] = fileparts(fname);
end

function meta = parse_filename_metadata(fname)
    % Parse filenames of form: aaaaNm_bbbb_cccc.mat
    % returns struct with fields load, condition, severity (strings) or empty if parse fails
    meta = struct();
    name = strip_extension(fname);
    parts = strsplit(name, '_');
    if numel(parts) >= 3
        meta.load = parts{1};
        meta.condition = parts{2};
        meta.severity = parts{3};
    else
        % can't parse, return empty
        meta = [];
    end
end
