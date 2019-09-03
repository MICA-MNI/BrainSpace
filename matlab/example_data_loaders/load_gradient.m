function data = load_template(name,idx)

P = mfilename('fullpath');
brainspace_path = regexp(P,'.*BrainSpace','match','once');
data_path = [brainspace_path filesep 'shared' filesep 'template_gradients'];

data = load([data_path filesep 'conte69_32k_' name '_gradient' num2str(idx) '.csv']);

end