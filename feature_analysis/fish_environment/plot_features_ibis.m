%% Load data
clear all;
DIR = '\\ems.elsc.huji.ac.il\avitan-lab\Lab-Shared\Analysis\FeedingAssaySapir\dataset_extracted_features\';
no  = load([DIR, 'combined_age_v2_hit_miss_abort_outcome_all_paramecia_bbox_no_target_heatmap_data_features_in_fov.mat']);
all = load([DIR, 'combined_age_v2_hit_miss_abort_outcome_all_paramecia_bbox_all_heatmap_data_features_in_fov.mat']);
tar = load([DIR, 'combined_age_v2_hit_miss_abort_outcome_all_paramecia_bbox_target_only_heatmap_data_features_in_fov.mat']);

%% Loop over features x ibis
global age_str outcome_str distance_str ibi_str angle_str
% Parse matlab field name as visual string name (global lambda functions)
% Field names should have specific structure, therefore here we convert it
% back to readable&printable format
age_str      = @(f) replace(replace(f, "a_", ""), "_", "-");
outcome_str  = @(f) replace(f, "_", "-");
distance_str = @(f) replace(replace(replace(f, "a_", ""), "_dot_", "."), "_", "-");
ibi_str      = @(f) replace(f, "__", "-");
angle_str    = @(f) replace(f, "_", "-");

% Get all available fields
ages        = fieldnames(all);
outcomes    = fieldnames(all.(ages{1}));
distances   = fieldnames(all.(ages{1}).(outcomes{1}));
angles      = fieldnames(all.(ages{1}).(outcomes{1}).(distances{1}));
features    = fieldnames(all.(ages{1}).(outcomes{1}).(distances{1}).(angles{1}));
ibi_names   = fieldnames(all.(ages{1}).(outcomes{1}).(distances{1}).(angles{1}).(features{1}));

%% Convert data to table, of specific checked feature (if more convenient- allow to use stats, intersections, etc)

%feature = 'distance_min_mm'; feature_name = "Min distance (mm)";
feature = 'n_paramecia'; feature_name = "N paramecia";

result = convert_data_to_table(all, feature);

%% Reorder table
o_ind = find(cellfun(@(n) string(n), result.Properties.VariableNames) == 'outcome_group');
a_ind = find(cellfun(@(n) string(n), result.Properties.VariableNames) == 'age_group');
result = [result(:,o_ind) result(:,a_ind) result(:, setxor([o_ind, a_ind], 1:size(result,1)))];

%% Plot feature
wanted_distances = [{"0-1.5mm"}, {"1.5-3mm"}];
wanted_angles = angles;
SEM = @(data) nanstd(data)./sqrt(size(data,2));
subset_table = @(T, age, out) cellfun( @(f) strcmp(f, age), [T.age_group]) & cellfun( @(f) strcmp(f, out), [T.outcome_group]);

for age = unique(result.age_group)'
    for out = unique(result.outcome_group)'
        curr = result(subset_table(result, age, out),:);
        figure; ind = 1; ax = [];
        for d = wanted_distances
            for a = wanted_angles'
                name_features = cellfun(@(n) append(d{:}, "_", a{:}, "_", ibi_str(n)), ibi_names);
                arr = nan(length([curr.([name_features(1)])]), length(ibi_names));
                for i = 1:length(ibi_names)
                    arr(:, i) = curr.([name_features(i)]);
                end
                ax = [ax subplot(length(wanted_distances), length(wanted_angles), ind)]; ind = ind + 1;
                %plot(arr', 'o-'); % Hard to see results when plot all => mean +sem
                hold on;
                plot(nanmean(arr), 'o-'); 
                errorbar(1:length(ibi_names), nanmean(arr), SEM(arr));
                ylabel(feature_name); xlabel("N ibi");
                title(append(d{:}, " : ", angle_str(a{:})))
            end
        end
        sgtitle(append(age{:}, " ", out))
    end
end

%% Functions
function result = convert_data_to_table(data, feature)
    % Convert nested struct to table.
    % The table contains metadata of age x outcome, and column per feature
    % (angle x distance x n_ibi)
    arguments
        data (1,1) struct
        feature (1, 1) string 
    end
    
    global age_str outcome_str distance_str ibi_str angle_str

    ages        = fieldnames(data);
    outcomes    = fieldnames(data.(ages{1}));
    distances   = fieldnames(data.(ages{1}).(outcomes{1}));
    angles      = fieldnames(data.(ages{1}).(outcomes{1}).(distances{1}));
    features    = fieldnames(data.(ages{1}).(outcomes{1}).(distances{1}).(angles{1}));
    ibi_names   = fieldnames(data.(ages{1}).(outcomes{1}).(distances{1}).(angles{1}).(features{1}));
    result = table();
    for age = ages'
        for outcome = outcomes'
            inner_table = table();
            for distance = distances'
                for angle = angles'
                    curr_subset = data.(age{:}).(outcome{:}).(distance{:}).(angle{:}).(feature);
                    if isempty(fieldnames(curr_subset))
                        continue
                    end
                    arr = nan(length(curr_subset.(ibi_names{1})), length(ibi_names));
                    for i = 1:length(ibi_names)
                        arr(:, i) = curr_subset.(ibi_names{i});
                    end
                    name_features = cellfun(@(n) append(distance_str(distance), "_", angle, "_", ibi_str(n)), ibi_names);
                    temp_table = array2table(arr, 'VariableNames', name_features);
                    indices = 1:size(temp_table,1);
                    temp_table.key=indices';
                    temp_table(:, 'outcome_group') = {outcome_str(outcome)};
                    temp_table(:, 'age_group')     = {age_str(age)};
                    if isempty(inner_table)
                        inner_table = temp_table;
                    elseif ~isempty(temp_table)
                        inner_table = innerjoin(inner_table, temp_table, "Keys", {'key', 'outcome_group', 'age_group'});
                    else
                        disp(temp_table)  % error
                    end
                end
            end

            if isempty(result)
                result = inner_table;
            elseif ~isempty(inner_table)
                result = [result; inner_table];
            else
                disp(inner_table)  % error
            end    
        end
    end

end