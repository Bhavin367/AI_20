clear; clc;

gait_file = '../gait_rad.csv';
if ~isfile(gait_file)
    error('Could not find %s. Run run_all.py (or main.py) in Python first.', gait_file);
end

gait = readmatrix(gait_file);
fprintf('Loaded %d frames x %d joints from %s\n', size(gait, 1), size(gait, 2), gait_file);

n_cycles = 2;
for c = 1:n_cycles
    for i = 1:3:size(gait, 1)
        plot_spider_pose(gait(i, :));
        title(sprintf('Cycle %d/%d - Frame %d/%d', c, n_cycles, i, size(gait, 1)));
        drawnow;
        pause(0.03);
    end
end
