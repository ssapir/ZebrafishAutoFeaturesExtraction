%% params
clear all;
global threshold min_diff max_diff win DATA_DIR;

if ispc  % windows
    DATA_DIR = "\\ems.elsc.huji.ac.il\avitan-lab\Lab-Shared\Data\";
else
    DATA_DIR = "/ems/elsc-labs/avitan-l/Lab-Shared/Data/";
end

% Thresholds for detection, degrees. Above 23 is hunt (mean value), 
% and each eye is between 10-90 deg.
threshold = 23;
min_diff  = 10;
max_diff  = 90;

% For calculation of events - fix segments which are too short (try to
% unify or remove if can't)
win = 120; % min event length is 120. Max is 780. mean is 420

save = true;

% todo read args?
fish_name = "20211108-f1";
subdir    = "TemperatureEffects";
excel_postfix = "-automatic-whole-movie-frames.csv";

%% Load data
% Loading function add needed headers to table (hard coded - note)
[fish_data, output_excel_props, fish_folder] = read_fish_and_excel(fish_name, subdir);

diff_real = sort(fish_data.events{1}.head.eyes_head_dir_diff_ang_in_deg', 1);
diff_abs  = sort(abs(diff_real), 1);
statuses  = fish_data.events{1}.fish_tracking_status_list;

%% Calc hunt
[startFrame, endFrame] = calculate_hunting(diff_abs, diff_real, statuses);

%% Create empty table (with correct size and field types)
newTable = array2table(zeros(length(startFrame),...
    length(output_excel_props.Properties.VariableNames)),...
    'VariableNames',output_excel_props.Properties.VariableNames);
varsCateg = [];
for i = 1 : size(output_excel_props,2)
    if strcmpi(class(output_excel_props(1,i).Variables), 'categorical')
        varsCateg = [varsCateg output_excel_props.Properties.VariableNames(i)];
    end
end
newTable = convertvars(newTable,varsCateg,"categorical");
for field = varsCateg  % quickfix strings to be empty (and not <undefined>)
    newTable.(field{1}) = strings(length(newTable.startFrame), 1);
end

%% Fill table with indication of automatic vs manual fit
newTable.startFrame            = startFrame;
newTable.endFrame              = endFrame;

% copy metadata (if exists)
newTable.fishName(:)           = fish_data.name;
newTable.age(:)                = fish_data.age_dpf;
newTable.NumberOfParamecia(:)  = fish_data.num_of_paramecia_in_plate;
newTable.AcclimationTime(:)    = fish_data.acclimation_time_min;
newTable.Feeding(:)            = fish_data.feeding_str;

%% Save
if save
    output_excel_path = fullfile(fish_folder, ...
        strjoin([fish_name, excel_postfix], ''));
    if ~isfile(output_excel_path)
        writetable(newTable, output_excel_path);
    else
        disp(["Already exists won't override: ", output_excel_path]);
    end
end

%% Functions
function [fish_data, f2frames, fish_folder] = read_fish_and_excel(fish_name, subdir)
    global DATA_DIR;
    if isempty(subdir)
        subdir = "FeedingAssay2020";
    end
    DATA_DIR = DATA_DIR + subdir;
    
    if isempty(fish_name)
        fish_name="20200720-f2";
    end
    
    fish_folder = fullfile(DATA_DIR, fish_name, '');
    mat_file_path = fullfile(fish_folder, 'processed_data_whole_movie', ...
        strjoin([fish_name,'_preprocessed.mat'], ''));
    disp(fish_folder);
    disp(mat_file_path);
    
    % Set up the Import Options and import the data
    opts = delimitedTextImportOptions("NumVariables", 21);

    % Specify range and delimiter
    opts.DataLines = [1, Inf];
    opts.Delimiter = ",";

    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";

    fish_data = load(mat_file_path).fish_data;

    % Specify column names and types (important mostly due to types)
%    opts.VariableNames = ["fishName", "age", "NumberOfParamecia", "AcclimationTime", "RecordingTime", "fps", "Feeding", "Origin", "Status0123", "AcquisitionComments", "startFrame", "endFrame", "endFramecutPoint", "outcome", "nBout", "visualField", "complexhunt", "GoingfoCollipse", "ReadyToCut", "Cut", "CuttingComments"];
%    opts.VariableTypes = ["categorical", "double", "double", "double", "double", "double", "categorical", "categorical", "string", "string", "double", "double", "double", "double", "double", "double", "string", "string", "double", "double", "categorical"];

% categorical data is empty strings
opts.VariableNames = ["fishName", "age", "NumberOfParamecia", "AcclimationTime", ...
    "RecordingTime", "fps", "Feeding", "Origin", "Status0123", ...
    "AcquisitionComments", "startFrame", "endFrame", "endFramecutPoint", ...
    "outcome", "nBout", "visualField", "complexhunt", "GoingfoCollipse", ...
    "ReadyToCut", "Cut", "CuttingComments", "AnalysisMovieComments", "is_an_event", ...
    "startFrame-1", "endFrame-1", "endFramecutPoint-1", "outcome-1", "complexhunt-1", "CuttingComments-1", ...
    "startFrame-2", "endFrame-2", "endFramecutPoint-2", "outcome-2", "complexhunt-2", "CuttingComments-2", ...
    "startFrame-3", "endFrame-3", "endFramecutPoint-3", "outcome-3", "complexhunt-3", "CuttingComments-3"];
opts.VariableTypes = ["categorical", "double", "double", "double", ...
    "double", "double", "categorical", "categorical", "double", ...
    "double", "double", "double", "double", ...
    "double", "double", "double", "double", "categorical", ...
    "double", "double", "categorical", "categorical", "double", ...
    "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", ...
    "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", ...
    "categorical", "categorical", "categorical", "categorical", "categorical", "categorical"];
    % Import the data
    f2frames = table('Size', [1 length(opts.VariableNames)], 'VariableNames', ...
        opts.VariableNames, 'VariableTypes', opts.VariableTypes);
end

function [startFrame, endFrame] = calculate_hunting(diff_abs, diff_real, statuses)

   global threshold min_diff max_diff win;
     
   % ss is logical status 
   is_hunt = zeros(length(statuses),1); 
   is_hunt(diff_abs(1,:) >= min_diff & diff_abs(1,:) <= max_diff & ...
           diff_abs(2,:) >= min_diff & diff_abs(2,:) <= max_diff & ...
           mean(diff_abs,1) >= threshold & mean(diff_abs,1) <= max_diff & ...
           statuses & ...
           sum(sign(diff_real), 1) == 0) = 1;
    
    % Find start-end (at least win size)
    A2 = movmedian(is_hunt, 3);  % correct small holes with moving median
    B2 = movsum(A2, [win 0]); 
    A  = zeros(size(B2));
    A(B2 >= round(win / 12)) = 1;  % threshold detection as >10 frames of the window
    % find edges (tf marks end of segments)
    B  = cumsum(A);
    tf = B(2:end) == B(1:end-1) & A(1:end-1)==1;
    tf = [tf ; A(end)];
    result         = diff([0 ; B(tf==1)]);  % length of segment
    endFrame_tmp   = find(tf);
    startFrame_tmp = endFrame_tmp - result;

    % Unify segments which are too close (win diff)
    startFrame_unif = startFrame_tmp;
    endFrame_unif   = endFrame_tmp;
    frame_diff      = startFrame_unif(2:end) - endFrame_unif(1:end-1);
    startFrame      = startFrame_unif;
    endFrame        = endFrame_unif;
    remove_ind      = zeros(size(startFrame));
    unify_ind       = find(frame_diff<=win)';
    for i = flip(unify_ind) % flip since we pass backwards the endFrames inds
        endFrame(i) = endFrame(i+1); % merge ends
        remove_ind(i + 1) = 1; % mark this to be removed
    end
    startFrame(remove_ind==1) = [];
    endFrame(remove_ind==1)   = [];

    % Move if still too short (<10 frames)
    short = find((endFrame-startFrame) <= 10);
    startFrame(short) = [];
    endFrame(short)   = [];
end