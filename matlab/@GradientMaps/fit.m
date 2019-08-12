function obj = fit(obj,connectivity_matrix,varargin)
% Computes the kernel, manifold learning, and alignment according the
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
%   - Manifold arguments
%       - diffusiontime (default: 0)
%           - Any positive numeric or zero
%       - alpha (default: 0.5)
%           - Numeric in the range [0 1].
%   - Alignment arguments
%       - niterations (default: 100)
%           - Any integer
%       - first_alignment_target (default: none)
%           A matrix with equivalent size to that of the gradient matrix.


% Deal with varargin
kernel_arg = {};
manifold_arg = {};
alignment_arg = {};
for ii = 1:2:numel(varargin)
    switch lower(varargin{ii})
        case {'sparsity','tolerance','gamma'}
            kernel_arg{end+1} = varargin{ii};
            kernel_arg{end+1} = varargin{ii+1};
        case {'diffusiontime','alpha'}
            manifold_arg{end+1} = varargin{ii};
            manifold_arg{end+1} = varargin{ii+1};
        case {'niterations','first_alignment_target'}
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

disp('Running gradient analysis...');
if isa(obj.method.kernel,'char')
    disp(['Kernel: ' obj.method.kernel]);
else
    disp(['Kernel: ' func2str(obj.method.kernel)]);
end
if isa(obj.method.manifold,'char')
    disp(['Manifold: ' obj.method.manifold]);
else
    disp(['Manifold: ' func2str(obj.method.manifold)]);
end
if isa(obj.method.alignment,'char')
    disp(['Alignment: ' obj.method.alignment]);
else
    disp(['Alignment: ' func2str(obj.method.alignment)]);
end

% Concatenate matrices for manifold alignment. 
if strcmp(obj.method.alignment,'Manifold Alignment')
    tmp = cat(2,connectivity_matrix{:});
    size_connectivity = cellfun(@(x)size(x,1),connectivity_matrix);
    clearvars connectivity_matrix
    connectivity_matrix{1} = tmp; 
end
N = numel(connectivity_matrix);

for ii = 1:N
    % Apply the kernel
    kernel_data = obj.kernels(connectivity_matrix{ii},kernel_arg{:});

    % Run the embedding
    if isa(obj.method.manifold,'char')
        [obj.gradients{ii}, obj.lambda{ii}] = ...
            manifolds(obj,kernel_data,manifold_arg{:});  
    else
        obj.gradients{ii} = obj.method.manifold(kernel_data); 
    end
    disp('Stored (unaligned) results in the gradients field.');
    

end

%Run the alignment
if strcmp(obj.method.alignment,'Procrustes Analysis')
    obj.aligned = procrustes_alignment(obj.gradients,alignment_arg{:});
    disp('Stored aligned results in the aligned field.');
elseif strcmp(obj.method.alignment,'Manifold Alignment')
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