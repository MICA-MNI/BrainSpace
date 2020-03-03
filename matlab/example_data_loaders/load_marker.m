function [metric_lh,metric_rh] = load_marker(name)
% LOAD_MARKER   loads metric data.
%
%   [metric_lh,metric_rh] = LOAD_MARKER(name) loads data on the cortical
%   surface. Set to 'thickness' for cortical thickness, 'curvature' for
%   curvature, or 't1wt2w' for t1w/t2w intensity. Left hemispheric data is
%   stored in metric_lh, and right in metric_rh. 
%
%   For more information, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/data_loaders/load_metric.html">ReadTheDocs</a>.

P = mfilename('fullpath');
brainspace_path = fileparts(fileparts(fileparts(P)));
data_path = [brainspace_path filesep 'shared' filesep 'data' filesep 'main_group'];

metric = load([data_path filesep 'conte69_32k_' lower(name) '.csv']);
metric_lh = metric(1:end/2);
metric_rh = metric(end/2+1:end); 
 
