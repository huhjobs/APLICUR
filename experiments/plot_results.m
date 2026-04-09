close all; clear; clc;

% ===== Paper-friendly plot defaults =====
set(groot, 'defaultLineLineWidth', 2.2);
set(groot, 'defaultAxesLineWidth', 1.2);
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultTextFontSize', 11);
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultLineMarkerSize', 7);
addpath('../export_fig-master');

% ===== Configuration =====
foldername = "smallexampledata/1";   % results subfolder under ./results/

% Collect dataset subdirectories
path = sprintf("./results/%s", foldername);
d = dir(path);
datalist = {d([d.isdir]).name};
datalist = datalist(~ismember(datalist, {'.','..'}));
disp(datalist)

% Run plotting for each dataset
for i = 1:length(datalist)
    doit(foldername, datalist{i})
end


function doit(foldername, datasetname)

    % Output directory for plots
    outfoldername = sprintf("%s", foldername); % Change to customize saving directory name
    [~, finalFolder] = fileparts(outfoldername);
    outdir = sprintf('./resultplots/%s/', outfoldername);
    if ~exist(outdir, 'dir'); mkdir(outdir); end

    datasetnametxt = string(datasetname);

    % Plot/save flags
    savefig = true;
    plotspectra = true;   % Singular value spectra (requires DEBUG mode in run_<alg>.m)
    plotrestimes = true;   % Residual vs time
    plotacctimes = true;   % Time-to-accuracy
    plotresvecs = true;   % Residual vs iterations
    xlimval = 0;    % Set >0 to cap x-axis in plotrestimes

    % ===== Load result files =====
    folder_path = sprintf('./results/%s/%s*/', foldername, datasetname);
    mat_files.basic = dir(fullfile(folder_path, "output_basic.mat"));

    % Select algorithm set:
    algs = ["lsqr", "aplicur_svdfree", "aplicur", "npcg", "bdp"];          % benchmark mode
    % algs = ["lsqr", "aplicur_svdfree", "aplicur", "npcg", "bdp_sparse"];   % sparse benchmark mode
    % algs = ["aplicur_fixed_under", "aplicur_fixed", "aplicur_fixed_over"]; % rank estimation mode
    % algs = ["aplicur", "aplicur_singleshot"];                              % scheduling mode
    % algs = ["aplicur_under", "aplicur_correct", "aplicur_over"];           % rank estimation (full alg) mode
    % algs = ["lsqr", "aplicur_svdfree", "aplicur"];                         % svd-free mode

    for alg = algs
        mat_files.(alg)     = dir(fullfile(folder_path, "output_" + alg + "_1*.mat"));
        to_files.(alg)      = dir(fullfile(folder_path, "terminal_output_" + alg + ".txt"));
        timedet_files.(alg) = dir(fullfile(folder_path, "timing_detail_" + alg + ".txt"));
    end

    % Line styles and markers (one entry per algorithm)
    colors     = lines(length(algs));
    linestyles = {'-', '-', ':', '-.', '--', '--'};
    markers    = {'o', 'none', 'none', 'none', 'none', 'none'};

    % Load .mat files into dict_arr, skipping missing ones
    fns = fieldnames(mat_files);
    fns = fns(cellfun(@(fn) ~isempty(mat_files.(fn)), fns));
    if ~ismember('npcg', fns)
        algs(algs == "npcg") = "nope";
    end
    for i = 1:numel(fns)
        fieldname = fns{i};
        data = load(fullfile(mat_files.(fieldname).folder, mat_files.(fieldname).name));
        dict_arr.(fieldname) = data.output;
    end

    if isempty(dict_arr), return; end
    d = dict_arr;
    num_algs = length(algs);

    % Save terminal outputs to file
    diary_file = sprintf('%sterminal_outputs.txt', outdir);
    fclose(fopen(diary_file, 'w'));
    diary(diary_file);
    for fieldname = fieldnames(to_files)'
        to_path = fullfile(to_files.(fieldname{1}).folder, to_files.(fieldname{1}).name);
        fprintf('\n\n%s\n', datasetnametxt);
        type(to_path);
    end

    % Print target ranks
    fprintf("\nmu = %.2e\n", d.basic.mu);
    for l = 1:num_algs
        alg = algs{l};
        try
            fprintf("Target rank in %s: %d\n", alg, d.(alg).ls);
        catch, end
    end
    diary off;

    % Escape underscores for LaTeX titles
    datasetnametxt = strrep(datasetnametxt, '_', '\_');


    %% === Plot 1: Singular Value Spectra ===
    if plotspectra
        figure('Visible','off','Units','inches','Position',[1 1 3.25 2.4], ...
               'PaperUnits','inches','PaperPosition',[0 0 3.25 2.4],'PaperSize',[3.25 2.4]);
        hold on;

        h_lines = [];
        % Plot original spectrum
        h_lines = [h_lines, plot(1:length(d.basic.svdAs), d.basic.svdAs, '-', ...
            'Color', 'k', 'LineWidth', 0.5, 'DisplayName', 'original')];

        % Plot preconditioned spectrum for each algorithm (if available)
        for a = 1:num_algs
            alg = algs{a};
            if ~contains(string(alg), 'basic') && ~(string(alg) == "nope") && isfield(d.(alg), 'precsv')
                svs = d.(alg).precsv;
                if string(alg) == 'npcg', svs = sqrt(svs); end
                h = plot(1:d.basic.n, svs, '-', ...
                    'Marker', markers{a}, 'MarkerSize', 2.5, ...
                    'MarkerIndices', round(linspace(1, d.basic.n, 20)), ...
                    'LineStyle', linestyles{mod(a-1, numel(linestyles)) + 1}, ...
                    'Color', colors(a,:), ...
                    'DisplayName', [pretty_alg_name(alg) sprintf(' ($\\kappa=%.1f$)', svs(1)/svs(end))]);
                h_lines = [h_lines, h];
            end
        end

        xlabel('$i$'); ylabel('$\sigma_i$');
        set(gca, 'YScale', 'log'); grid on;
        leg = legend(h_lines, 'Location', 'northeast');
        leg.FontSize = 8.5; leg.ItemTokenSize = [18, 8]; leg.LineWidth = 0.8;
        leg.Position(3) = leg.Position(3) * 1.1;
        hold off;

        if savefig
            outsubdir = fullfile(outdir, sprintf('spectra%s', finalFolder));
            if ~exist(outsubdir, 'dir'), mkdir(outsubdir); end
            print(gcf, '-dpdf', fullfile(outsubdir, sprintf('spectra %s.pdf', datasetname)), '-r300');
        end
    end


    %% === Plot 2: Relative Residual vs Time ===
    if plotrestimes
        figure('Visible','off','Units','inches','Position',[1 1 3.25 2.4], ...
               'PaperUnits','inches','PaperPosition',[0 0 3.25 2.4],'PaperSize',[3.25 2.4]);
        hold on;

        b_norm = norm(d.basic.bs);
        h_lines = [];

        for a = 1:num_algs
            alg = algs{a};
            if string(alg) == "basic" || string(alg) == "nope", continue; end

            % Offset timestamps by preconditioning time if applicable
            if contains(string(alg), 'npcg') || contains(string(alg), 'bdp')
                d.(alg).timestamps = d.(alg).timestamps + d.(alg).timings.prec.total;
            end

            target_rank = 0;
            if string(alg) == "npcg"
                target_rank = d.npcg.ls;
            elseif contains(string(alg),"aplicur")
                target_rank = d.(alg).ls;
            elseif string(alg) == "bdp"
                target_rank = d.basic.n;
            end

            ts = d.(alg).timestamps;
            if contains(string(alg), 'aplicur')
                rvec = d.(alg).resvecs_woreg_ts / b_norm;
            else
                rvec = d.(alg).resvecs_woreg / b_norm;
            end

            h = plot(ts, rvec, '-', ...
                'Marker', markers{a}, 'MarkerSize', 2.5, ...
                'MarkerIndices', round(linspace(1, length(ts), 20)), ...
                'LineStyle', linestyles{mod(a-1, numel(linestyles)) + 1}, ...
                'Color', colors(a,:), ...
                'DisplayName', [pretty_alg_name(alg) sprintf('\\ (rk=%d)', target_rank)]);
            h_lines = [h_lines, h];
        end

        % Reference line: optimal residual
        if d.basic.mu
            ref_val = d.basic.relress_woreg;
            leg_str = '$\|A x_\mu - b\|_2 / \|b\|_2$';
        else
            ref_val = d.basic.relress;
            leg_str = '$\|A x_\ast - b\|_2 / \|b\|_2$';
        end
        h_ref = yline(ref_val, '-');
        h_ref.Annotation.LegendInformation.IconDisplayStyle = 'off';
        h_lines = [h_lines, plot(NaN, NaN, '-', 'Color', 'k', 'LineWidth', 0.5, 'DisplayName', leg_str)];

        xlabel('Time (s)'); ylabel('Rel.\ Residual');
        set(gca, 'YScale', 'log'); grid on;
        if xlimval, xlim([0 xlimval]); end
        leg = legend(h_lines, 'Location', 'northeast');
        leg.FontSize = 8.5; leg.ItemTokenSize = [18, 8]; leg.LineWidth = 0.8;
        hold off;

        if savefig
            outsubdir = fullfile(outdir, sprintf('restimes%s', finalFolder));
            if ~exist(outsubdir, 'dir'), mkdir(outsubdir); end
            print(gcf, '-dpdf', fullfile(outsubdir, sprintf('restime %s.pdf', datasetname)), '-r300');
        end
    end


    %% === Plot 4: Relative Residual vs Iterations ===
    if plotresvecs
        figure('Visible','off','Units','inches','Position',[1 1 3.25 2.4], ...
               'PaperUnits','inches','PaperPosition',[0 0 3.25 2.4],'PaperSize',[3.25 2.4]);
        hold on;

        b_norm = norm(d.basic.bs);
        h_lines = [];

        for a = 1:num_algs
            alg = algs{a};
            if string(alg) == "basic" || string(alg) == "nope", continue; end

            if contains(string(alg), 'npcg') || contains(string(alg), 'bdp')
                d.(alg).timestamps = d.(alg).timestamps + d.(alg).timings.prec.total;
            end

            target_rank = 0;
            if string(alg) == "npcg",             target_rank = d.npcg.ls;
            elseif contains(string(alg),"aplicur"), target_rank = d.(alg).ls;
            elseif string(alg) == "bdp",           target_rank = d.basic.n; end

            rvec = d.(alg).resvecs_woreg / b_norm;
            h = plot(1:length(rvec), rvec, '-', ...
                'Color', colors(a,:), ...
                'DisplayName', [pretty_alg_name(alg) sprintf(' (rk=%d)', target_rank)]);
            h_lines = [h_lines, h];
        end

        xlabel('Iterations'); ylabel('Rel.\ Residual');
        set(gca, 'YScale', 'log'); grid on;
        leg = legend(h_lines, 'Location', 'northeast');
        leg.FontSize = 8.5; leg.ItemTokenSize = [8, 8]; leg.LineWidth = 0.8;
        hold off;

        if savefig
            outsubdir = fullfile(outdir, sprintf('resvecs%s', finalFolder));
            if ~exist(outsubdir, 'dir'), mkdir(outsubdir); end
            print(gcf, '-dpdf', fullfile(outsubdir, sprintf('resvec %s.pdf', datasetname)), '-r300');
        end
    end


    %% === Plot 3: Time-to-Accuracy ===
    if plotacctimes
        b_norm = norm(d.basic.bs);

        % Desired accuracy levels as fractions of the log-scale range to optimal
        desiredratios = [1, 4/5, 3/5, 2/5, 1/5, 1/10, 1/20, 1/50, 1/100];
        if ~d.basic.mu
            d.basic.relress_woreg = d.basic.relress;
        end
        desired_accs = 10.^(log10(d.basic.relress_woreg) + desiredratios .* (-log10(d.basic.relress_woreg)));

        % For each algorithm, find the first time it reaches each accuracy level
        result_mat = NaN(num_algs, numel(desired_accs));
        for a = 1:num_algs
            alg = algs{a};
            if contains(string(alg), 'basic') || string(alg) == "nope", continue; end

            if contains(string(alg), 'npcg') || contains(string(alg), 'bdp')
                d.(alg).timestamps = d.(alg).timestamps + d.(alg).timings.prec.total;
            end

            ts = d.(alg).timestamps;
            if contains(string(alg), 'aplicur')
                if ~d.basic.mu, d.(alg).resvecs_woreg_ts = d.(alg).tsresvecs; end
                relresvec = d.(alg).resvecs_woreg_ts / b_norm;
            else
                if ~d.basic.mu, d.(alg).resvecs_woreg = d.(alg).resvecs; end
                relresvec = d.(alg).resvecs_woreg / b_norm;
            end

            if ~isempty(relresvec)
                for k = 1:numel(desired_accs)
                    idx = find(relresvec < desired_accs(k), 1, 'first');
                    if ~isempty(idx), result_mat(a, k) = ts(idx); end
                end
            end
        end

        % Plot time-to-accuracy curves
        desired_accs = desired_accs(:)';
        fig = figure('Visible','off','Units','inches','Position',[1 1 3.25 2.4], ...
                     'PaperUnits','inches','PaperPosition',[0 0 3.25 2.4],'PaperSize',[3.25 2.4]);
        hold on;
        cols = lines(size(result_mat, 1));

        for r = 1:size(result_mat, 1)
            times = result_mat(r, :);
            valid = ~isnan(times);
            if any(valid)
                semilogx(desired_accs(valid), times(valid), '-o', ...
                    'Color', cols(r,:), 'MarkerSize', 6, ...
                    'DisplayName', pretty_alg_name(algs{r}));
            end
        end

        set(gca, 'XScale', 'log', 'YScale', 'log', 'XDir', 'reverse');
        xlabel('Desired accuracy'); ylabel('Time (s)');
        grid on; box on;
        leg = legend('Location', 'best');
        leg.FontSize = 8.5; leg.ItemTokenSize = [8, 8]; leg.LineWidth = 0.8;
        hold off;

        if savefig
            outsubdir = fullfile(outdir, sprintf('acctimes%s', finalFolder));
            if ~exist(outsubdir, 'dir'), mkdir(outsubdir); end
            outfig = fullfile(outsubdir, sprintf('%s_restime_plot.pdf', datasetname));
            print(fig, outfig, '-dpdf');
            close(fig);
            fprintf('Saved plot for %s -> %s\n', datasetname, outfig);
        end
    end

end

% Uncomment the version matching your chosen algs set above.

% % --- Rank estimation mode ---
% function s = pretty_alg_name(s)
%     s = regexprep(s, 'aplicur_fixed_under', 'under');
%     s = regexprep(s, 'aplicur_fixed_over', 'over');
%     s = regexprep(s, 'aplicur_fixed', 'correct');
% 
%     % Optional: clean up repeated underscores
%     s = regexprep(s, '__+', '_');
% 
%     % Escape for LaTeX
%     s = strrep(s, '_', '\_');
% end

% % --- Scheduling mode ---
% function s = pretty_alg_name(s)
%     s = regexprep(s, 'aplicur_singleshot', 'w/o scheduling');
%     s = regexprep(s, 'aplicur', 'w/ scheduling');
% 
%     % Optional: clean up repeated underscores
%     s = regexprep(s, '__+', '_');
% 
%     % Escape for LaTeX
%     s = strrep(s, '_', '\_');
% end
% 
% % --- Rank estimation (fully adaptive) mode ---
% function s = pretty_alg_name(s)
%     s = regexprep(s, 'aplicur_under', 'under');
%     s = regexprep(s, 'aplicur_over', 'over');
%     s = regexprep(s, 'aplicur_correct', 'correct');
% 
%     % Optional: clean up repeated underscores
%     s = regexprep(s, '__+', '_');
% 
%     % Escape for LaTeX
%     s = strrep(s, '_', '\_');
% end

% % --- SVD-free mode ---
% function s = pretty_alg_name(s)
%     s = regexprep(s, 'aplicur_svdfree', 'svd-free');
%     s = regexprep(s, 'aplicur', 'svd-based');
% 
%     s = regexprep(s, 'lsqr', 'LSQR');
%     % % Optional: clean up repeated underscores
%     % s = regexprep(s, '__+', '_');
%     % 
%     % % Escape for LaTeX
%     % s = strrep(s, '_', '\_');
% 
%     % s = sprintf('$%s$',s);
% end

% --- Benchmark mode (default) ---
function s = pretty_alg_name(s)
    s = regexprep(s, '_singleshot|shot', '_single');

    s = regexprep(s, '_isp', '');
    s = regexprep(s, '_sparse', '');

    s = regexprep(s, 'lsqr', 'LSQR');
    s = regexprep(s, 'npcg', 'NPCG');
    s = regexprep(s, 'aplicur', 'APLICUR');
    s = regexprep(s, 'bdp', 'Blendenpik');

    s = regexprep(s, '_svdfree', '-sf');

    % Optional: clean up repeated underscores
    s = regexprep(s, '__+', '_');

    % Escape for LaTeX
    s = strrep(s, '_', '\_');
end