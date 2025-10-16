% convert_current_temp_csvs.m
% Put this file in the project root (same folder that has 'current_temp' and other folders).
% Run: convert_current_temp_csvs
%
% Behavior:
% - Looks for 'current_temp' folder in pwd
% - Creates 'current_temp_csv' folder in pwd (if needed)
% - Reads each CSV in current_temp, handles TDMS-exported headers like /'Log'/'cDAQ...
% - Detects if Time Stamp column exists; if not, creates Time Stamp = (0:N-1)' * dt
%   (default dt = 1.0 second). Change dt below if you know the true sampling interval.
% - Writes a standardized CSV with columns:
%   Time Stamp,Temperature_housing_A,Temperature_housing_B,U-phase,V-phase,W-phase,load,condition,severity

function convert_current_temp_csvs()
    root_dir = pwd;
    input_dir = fullfile(root_dir, 'current_temp');
    output_dir = fullfile(root_dir, 'current_temp_csv');

    if ~isfolder(input_dir)
        error('Input folder not found: %s', input_dir);
    end
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end

    % Default sample interval (seconds) to use if file lacks Time Stamp
    dt_default = 1.0;  % change this to the true dt if known

    files = dir(fullfile(input_dir, '*.csv'));
    if isempty(files)
        fprintf('No CSV files found in %s\n', input_dir);
        return;
    end

    for k = 1:numel(files)
        fname = files(k).name;
        fpath = fullfile(input_dir, fname);
        try
            % read first line to inspect header
            fid = fopen(fpath, 'r');
            if fid < 0
                warning('Failed to open %s', fpath);
                continue;
            end
            firstLine = strtrim(fgetl(fid));
            fclose(fid);

            % Some TDMS->CSV exports put single-line channel header like:
            % /'Log'/'cDAQ9185-1F486B5Mod1/ai0',/'Log'/'cDAQ9185-1F486B5Mod1/ai1',...
            % We want to map them to the 5 expected columns (no timestamp) or detect a Time Stamp.
            % We'll try to read the CSV robustly with readtable.
            opts = detectImportOptions(fpath, 'NumHeaderLines', 0);
            % force reading as text first to avoid weird column names
            Traw = readtable(fpath, opts);

            % If the very first column name looks like a path (contains 'Log' or starts with "/'"),
            % we assume there is no Time Stamp and the file columns correspond to:
            %  [Temperature_housing_A, Temperature_housing_B, U-phase, V-phase, W-phase]
            colnames = Traw.Properties.VariableNames;

            % clean column names (remove quotes, slashes)
            cleaned = clean_colnames(colnames);

            hasTimeStamp = any(contains(lower(cleaned), 'time')) || any(strcmpi(cleaned, 'TimeStamp'));

            if hasTimeStamp
                % Ensure first column is a Time Stamp-like column; if not exactly named Time Stamp, rename.
                % Find the time column index
                timeIdx = find(contains(lower(cleaned), 'time'), 1, 'first');
                timeVec = Traw{:, timeIdx};
                % Remove the existing time column from Traw columns for reassignment below
                nonTimeIdx = setdiff(1:size(Traw,2), timeIdx);
                dataMatrix = Traw{:, nonTimeIdx};
                dataCols = cleaned(nonTimeIdx);
            else
                % No explicit time column; assume the table has 5 columns corresponding to:
                % Temperature_housing_A, Temperature_housing_B, U-phase, V-phase, W-phase
                dataMatrix = Traw{:,:};
                dataCols = cleaned;
                N = size(dataMatrix,1);
                timeVec = (0:(N-1))' * dt_default;
            end

            % Now map columns to target names. We expect 5 data columns.
            % If file has more or fewer columns, we'll try sensible mapping:
            % Prefer to pick columns by matching names, otherwise use order.
            targetNames = {'Temperature_housing_A', 'Temperature_housing_B', 'U-phase', 'V-phase', 'W-phase'};

            % Create table starting with Time Stamp
            T = table();
            T.("Time Stamp") = timeVec;

            % Map data columns into the 5 target columns
            M = size(dataMatrix,2);
            mapped = false(1,5);

            % 1) match by words in column names if possible
            for i = 1:M
                cname = dataCols{i};
                lname = lower(cname);
                if contains(lname, 'temp') && ~mapped(1)
                    T.("Temperature_housing_A") = dataMatrix(:, i);
                    mapped(1) = true;
                elseif contains(lname, 'housing') && contains(lname, 'b') && ~mapped(2)
                    T.("Temperature_housing_B") = dataMatrix(:, i);
                    mapped(2) = true;
                elseif contains(lname, 'u') && contains(lname, 'phase') && ~mapped(3)
                    T.("U-phase") = dataMatrix(:, i);
                    mapped(3) = true;
                elseif contains(lname, 'v') && contains(lname, 'phase') && ~mapped(4)
                    T.("V-phase") = dataMatrix(:, i);
                    mapped(4) = true;
                elseif contains(lname, 'w') && contains(lname, 'phase') && ~mapped(5)
                    T.("W-phase") = dataMatrix(:, i);
                    mapped(5) = true;
                end
            end

            % 2) fill remaining target columns by order from left to right
            j = 1; % index in dataMatrix
            for tname_i = 1:numel(targetNames)
                if ~isfield(T, targetNames{tname_i})
                    % find next unused column in dataMatrix
                    while j <= M && any(ismember(dataCols{j}, T.Properties.VariableNames))
                        j = j + 1;
                    end
                    if j <= M
                        T.(targetNames{tname_i}) = dataMatrix(:, j);
                        j = j + 1;
                    else
                        % if not enough columns, fill with NaNs
                        T.(targetNames{tname_i}) = NaN(height(T),1);
                    end
                end
            end

            % Add filename metadata columns (load, condition, severity)
            meta = parse_filename_metadata(fname);
            if ~isempty(meta)
                T.load = repmat({meta.load}, height(T), 1);
                T.condition = repmat({meta.condition}, height(T), 1);
                T.severity = repmat({meta.severity}, height(T), 1);
            end

            % Write out CSV to output_dir
            outname = fullfile(output_dir, [strip_extension(fname) '.csv']);
            writetable(T, outname);
            fprintf('Converted: %s -> %s\n', fname, outname);

        catch ME
            warning('Failed to process %s: %s', fname, ME.message);
        end
    end

    fprintf('Finished processing %d files.\n', numel(files));
end

%% ===== Helper functions =====
function s = strip_extension(fname)
    [~, s, ~] = fileparts(fname);
end

function cleaned = clean_colnames(colnames)
    % remove slashes, quotes, spaces and make simpler forms for matching
    cleaned = cell(size(colnames));
    for i=1:numel(colnames)
        c = colnames{i};
        % common TDMS CSV gives header like: /'Log'/'cDAQ9185-1F486B5Mod1/ai0'
        % remove slashes and single quotes
        c = strrep(c, '/', '');
        c = strrep(c, '''', '');
        c = strrep(c, '"', '');
        c = strrep(c, '?', '');
        c = strtrim(c);
        % replace non-alphanum with underscore
        c = regexprep(c, '[^a-zA-Z0-9]', '_');
        cleaned{i} = c;
    end
end

function meta = parse_filename_metadata(fname)
    % Parse filenames of form: aaaaNm_bbbb_cccc.*  ->  load, condition, severity
    meta = struct();
    name = strip_extension(fname);
    parts = strsplit(name, '_');
    if numel(parts) >= 3
        meta.load = parts{1};
        meta.condition = parts{2};
        meta.severity = parts{3};
    else
        meta = [];
    end
end
