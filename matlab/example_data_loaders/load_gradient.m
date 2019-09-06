function data = load_gradient(name,idx)
% LOAD_GRADIENT   loads template gradients.
%
%   data = LOAD_GRADIENT(name,idx) loads template gradients. For functional
%   connectivity gradients set name to 'fc', for microstructural profile
%   covariance gradients set it to 'mpc'. idx can be set to 1 for the first
%   gradient or 2 for the second. 
%
%   For more information, please consult our <a
%   href="https://brainspace.readthedocs.io/en/stable/pages/matlab_doc/data_loaders/load_template.html">ReadTheDocs</a>.

P = mfilename('fullpath');
brainspace_path = regexp(P,'.*BrainSpace','match','once');
data_path = [brainspace_path filesep 'shared' filesep 'template_gradients'];

data = load([data_path filesep 'conte69_32k_' name '_gradient' num2str(idx) '.csv']);

end