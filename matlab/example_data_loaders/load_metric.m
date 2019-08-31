function [metric_lh,metric_rh] = load_metric(type)

P = mfilename('fullpath');
brainspace_path = regexp(P,'.*BrainSpace','match','once');
data_path = [brainspace_path filesep 'shared' filesep 'data' filesep 'main_group'];

metric = load([data_path filesep 'conte69_32k_' lower(type) '.csv']);
metric_lh = metric(1:end/2);
metric_rh = metric(end/2+1:end); 
 
