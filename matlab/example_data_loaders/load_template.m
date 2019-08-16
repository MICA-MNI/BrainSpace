function data = load_template(type,number)

P = mfilename('fullpath');
brainspace_path = regexp(P,'.*BrainSpace','match','once');
data_path = [brainspace_path filesep 'shared' filesep 'data' filesep 'template_gradients'];

data = load([data_path filesep 'conte69_32k_' type '_gradient' num2str(number) '.csv']);

end