function obj = fit(obj,connectivity_matrix,varargin)
% Computes the kernel, embedding, and alignment according the
% properties of the calling object. A data matrix or cell array of data
% matrices is obligatory as input. Valid name-value pairs of optional
% arguments are as follows:
%
%   - Kernel arguments
%       - Sparsity (default: 90) 
%           - Any numeric between 0 and 100
%       - Tolerance (default: 1e-6)
%           - Any numeric
%       - Gamma (default: 1/number_of_features)
%           - Any numeric
%   - Approach arguments
%       - diffusion_time (default: 0)
%           - Any positive numeric or zero
%       - alpha (default: 0.5)
%           - Numeric in the range [0 1].
%   - Alignment arguments
%       - niterations (default: 10)
%           - Any integer
%       - reference (default: nan)
%           A matrix with equivalent size to that of the gradient matrix.
%
% For complete documentation please consult our <a
% href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/main_functionality/gradientmaps.html">ReadTheDocs</a>.

% Deal with varargin
kernel_arg = {};
approach_arg = {};
alignment_arg = {};
for ii = 1:2:numel(varargin)
    switch lower(varargin{ii})
        case {'sparsity','tolerance','gamma'}
            kernel_arg{end+1} = varargin{ii};
            kernel_arg{end+1} = varargin{ii+1};
        case {'diffusion_time','alpha'}
            approach_arg{end+1} = varargin{ii};
            approach_arg{end+1} = varargin{ii+1};
        case {'niterations','reference'}
            alignment_arg{end+1} = varargin{ii};
            alignment_arg{end+1} = varargin{ii+1};
        otherwise
            error('Unknown name-value pair.');            
    end
end

% Make sure connectivity_matrix is in cell format
if ~iscell(connectivity_matrix)
    connectivity_matrix = {connectivity_matrix};
end

% Most of MATLAB's operations default are column-wise, but we define rows to
% be seeds. The easiest solution is to just transpose at the start, rather
% than transpose at every operation.
for ii = 1:numel(connectivity_matrix)
    connectivity_matrix{ii} = connectivity_matrix{ii}';
end
    
disp('Running gradient analysis...');
if isa(obj.method.kernel,'char')
    disp(['Kernel: ' obj.method.kernel]);
else
    disp(['Kernel: ' func2str(obj.method.kernel)]);
end
if isa(obj.method.approach,'char')
    disp(['Approach: ' obj.method.approach]);
else
    disp(['Approach: ' func2str(obj.method.approach)]);
end
if isa(obj.method.alignment,'char')
    disp(['Alignment: ' obj.method.alignment]);
else
    disp(['Alignment: ' func2str(obj.method.alignment)]);
end

% Concatenate matrices for joint alignment. 
if strcmp(obj.method.alignment,'Joint Alignment')
    try
        tmp = cat(2,connectivity_matrix{:});
    catch ME
        if strcmp(ME.identifier,'MATLAB:catenate:dimensionMismatch')
            error('Joint alignment requires that matrices have the same number of features.')
        else
            rethrow(ME)
        end
    end
    size_connectivity = cellfun(@(x)size(x,2),connectivity_matrix);
    clearvars connectivity_matrix
    connectivity_matrix{1} = tmp; 
end
N = numel(connectivity_matrix);

for ii = 1:N
    % Apply the kernel
    kernel_data = obj.kernels(connectivity_matrix{ii},kernel_arg{:});

    % Check for Infs or NaNs in the kernel data
    if any(isnan(kernel_data(:))) || any(isinf(kernel_data(:)))
        error('Detected NaNs or Infs in the kernel data.');
    end
    
    % Run the embedding
    if isa(obj.method.approach,'char')
        [obj.gradients{ii}, obj.lambda{ii}] = ...
            approaches(obj,kernel_data,approach_arg{:});  
    else
        obj.gradients{ii} = obj.method.approach(kernel_data); 
    end
    disp('Stored (unaligned) results in the gradients field.');
end

%Run the alignment
if strcmp(obj.method.alignment,'Procrustes Analysis')
    obj.aligned = procrustes_alignment(obj.gradients,alignment_arg{:});
    disp('Stored aligned results in the aligned field.');
elseif strcmp(obj.method.alignment,'Joint Alignment')
    for ii = 1:numel(size_connectivity)
        if ii == 1
            nmin = 1;
        else
            nmin = sum(size_connectivity(1:ii-1))+1; 
        end
        nmax = sum(size_connectivity(1:ii)); 
        obj.aligned{ii} = obj.gradients{1}(nmin:nmax,:); 
    end
    obj.gradients = {}; 
elseif isa(obj.method.alignment,'function_handle')
    obj.aligned = obj.method.alignment(obj);
elseif strcmp(obj.method.alignment,'None')
    % Pass
else
    error('Unknown alignment method.');
end

end
