function download_data(data_file)
% Downloads the online data.
data_path = fileparts(data_file);
if ~exist(data_path,'dir')
    mkdir(data_path);
end

data_file = [data_path filesep 'figure_data.mat'];
url = 'https://box.bic.mni.mcgill.ca/s/fzHPtKsWgOt9TT9/download';
if ~exist(data_file,'file')
    disp('Did not find requisite data, downloading...');
    mkdir(data_path);
    websave(data_file,url);
    disp('Done.'); 
end