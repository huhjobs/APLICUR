% ===== Configuration =====
dataset_path = '../datasets/smallexampledata';  % path to dataset folder

% Auto-assign the smallest unused positive integer as results subfolder name
existing = dir('./results');
existing = existing([existing.isdir]);
nums = str2double({existing.name});
nums = nums(~isnan(nums) & nums > 0);
n = num2str(min(setdiff(1:max([nums, 0])+1, nums)));

% Results will be saved under ./results/smallexampledata/<n>/
[~, dataset_folder] = fileparts(dataset_path);
resultsavepath = fullfile(dataset_folder, n);

% ===== Dataset filtering =====
d = dir(dataset_path);
d = d(~[d.isdir]);                          % keep files only
d = d(~startsWith({d.name}, '.'));          % remove hidden files (e.g. .DS_Store)

% Uncomment one of the following to select a subset of datasets:

% % Multiple conditions:
% d = d((contains({d.name}, 'sp1e-02_12k10k_i') | ...
%        contains({d.name}, 'sp1e-02_12k10k_x')) & ...
%        contains({d.name}, '_incoh'));

% Single condition:
% d = d(contains({d.name}, 'sp1e-02_6k100_x_incoh'));

% % No filter (run all datasets):
% % (leave d as-is)

% ===== Run experiments =====
for k = 1:numel(d)
    mat_file_path = fullfile(dataset_path, d(k).name);
    disp(mat_file_path);
    run_experiments('DATASETPATH', mat_file_path, 'SUBDIR', resultsavepath);
end