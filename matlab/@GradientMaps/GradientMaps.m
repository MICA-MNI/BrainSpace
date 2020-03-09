classdef GradientMaps
% obj = BrainSpace(varargin)
%
% Class definition and constructor for BrainSpace. The following
% name-value pairs are allowed as input.
%
%   - kernel (default: normalized angle)
%       - 'p','pearson'
%       - 'sm','spearman'
%       - 'g','gaussian'
%       - 'na','normalized angle'
%       - 'cs','cosine similarity'
%       - '','none'
%       - a function handle
%   - approach (default: diffusion embedding)
%       - 'dm','diffusion embedding'
%       - 'le','laplacian eigenmap'
%       - 'pca','principal component analysis'
%       - a function handle
%   - alignment (default: none)
%       - 'none',''
%       - 'pa','procrustes analysis'
%       - 'ja','joint alignment'
%       - a function handle
%   - n_components (default: 10)
%       - Any natural number.
%   - random_state (default: nan)
%       - Any valid input for MATLAB's "rng" function or nan for no
%           initialization.
%
% For complete documentation, including descriptions of this object's
% properties and methods please consult our <a
% href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/main_functionality/gradientmaps.html">ReadTheDocs</a>.
%
% See also: GRADIENTMAPS.FIT
    
    properties (SetAccess = private)
        method
        gradients
        aligned
        lambda        
    end
    
    properties (Access = private)
        random_state
    end
    
    methods
        %% Constructor
        function obj = GradientMaps(varargin)
            disp('Launching BrainSpace, the gradient connectivity toolbox.');
            disp('');
            
            % Parse input
            in_fun = @(x) isa(x,'char') || isa(x,'function_handle');
            p = inputParser;
            addParameter(p, 'kernel', 'normalized angle', in_fun);
            addParameter(p, 'approach', 'diffusion embedding', in_fun);
            addParameter(p, 'alignment', 'none', in_fun);
            addParameter(p, 'n_components', 10, @isnumeric);
            addParameter(p, 'random_state', nan);
            
            parse(p, varargin{:});
            R = p.Results;
            
            % Set the properties
            obj = obj.set( ...
                'kernel',       R.kernel, ...
                'approach',     R.approach, ...
                'alignment',    R.alignment, ...
                'random_state', R.random_state, ...
                'n_components', R.n_components);
        end
    end
    methods(Access = private)
        
        %% Private methods
        function obj = set(obj,varargin)
            % obj2 = SET(obj,varargin)
            %
            % Private function to set the properties of the BrainSpace
            % object.
            
            change_string = {};
            for ii = 1:2:numel(varargin)
                switch lower(varargin{ii})
                    case 'kernel'
                        if isa(varargin{ii+1},'function_handle')
                            obj.method.kernel = varargin{ii+1};
                            change_string{end+1} = ('Set the kernel to a custom function handle.');
                        else
                            switch lower(varargin{ii+1})
                                case {'none',''}
                                    obj.method.kernel = 'None';
                                case {'p','pearson'}
                                    obj.method.kernel = 'Pearson';
                                case {'sm','spearman'}
                                    obj.method.kernel = 'Spearman';
                                case {'g','gaussian'}
                                    obj.method.kernel = 'Gaussian';
                                case {'cs','cosine','cosine similarity','cossim','cosine_similarity','cosinesimilarity'}
                                    obj.method.kernel = 'Cosine Similarity';
                                case {'na','normalized angle','normalizedangle','normangle','normalized_angle'}
                                    obj.method.kernel = 'Normalized Angle';
                                otherwise
                                    error('Unknown kernel. Valid kernels are: ''none'', ''pearson'', ''spearman'', ''Gaussian'', ''cosine similarity'', and ''normalized angle''');
                            end
                            change_string{end+1} = (['Set the kernel to: ' obj.method.kernel '.']);
                        end
                        
                    case 'approach'
                        if isa(varargin{ii+1},'function_handle')
                            obj.method.approach = varargin{ii+1};
                            change_string{end+1} = ('Set the approach to a custom function handle.');
                        else
                            switch lower(varargin{ii+1})
                                case {'pca','principalcomponentanalysis','principal component analysis'}
                                    obj.method.approach = 'Principal Component Analysis';
                                case {'dm','diffusion embedding','diffusionembedding','diffemb'}
                                    obj.method.approach = 'Diffusion Embedding';
                                case {'le','laplacian eigenmap','laplacian eigenmaps','lapeig','laplacianeigenmaps','laplacianeigenmap'}
                                    obj.method.approach = 'Laplacian Eigenmap';
                                otherwise
                                    error('Unknown approach. Valid approaches are: ''principal component analysis'', ''diffusion embedding'', and ''laplacian eigenmap''');
                            end
                            change_string{end+1} = (['Set the approach to: ' obj.method.approach '.']);
                        end
                    case 'alignment'
                        if isa(varargin{ii+1},'function_handle')
                            obj.method.kernel = varargin{ii+1};
                            change_string{end+1} = ('Set the alignment to a custom function handle.');
                        else
                            switch lower(varargin{ii+1})
                                case {'','none'}
                                    obj.method.alignment = 'None';
                                case {'pa','procrustes','procrustes analysis','procrustesanalysis'}
                                    obj.method.alignment = 'Procrustes Analysis';
                                case {'ja','joint','jointalignment'}
                                    obj.method.alignment = 'Joint Alignment';
                            end
                            change_string{end+1} = (['Set the alignment to: ' obj.method.alignment '.']);
                        end
                        
                    case 'random_state'
                        obj.random_state = varargin{ii+1};
                        if ~isnan(varargin{ii+1})
                            change_string{end+1} = ['Set the random state initialization to: ' num2str(varargin{ii+1}) '.'];
                        else
                            change_string{end+1} = ['No random state initialization set.'];
                        end
                        
                    case 'n_components'
                        obj.method.n_components = varargin{ii+1};
                        change_string{end+1} = ['Set the number of requested components to: ' num2str(varargin{ii+1}) '.'];
                        
                    otherwise
                        error('Unknown property. Valid properties are: ''connectivitymatrix'', ''kernel'', ''approach'', and ''nullmodel''.');
                end
            end
            for ii = 1:numel(change_string)
                disp(change_string{ii})
            end
            disp(' ')
        end
        % -------------------------------------
        % -------------------------------------
        % -------------------------------------
        function kernel_data = kernels(obj,data,varargin)
            % Applies kernel to the data. Known kernels are "none", "Cosine
            % Similarity", and "Normalized Angle".
            p = inputParser;
            addParameter(p, 'sparsity', 90, @isnumeric);
            addParameter(p, 'tolerance', 1e-6, @isnumeric);
            addParameter(p, 'gamma', 1/size(data,1), @isnumeric);
            
            parse(p, varargin{:});
            kernel = obj.method.kernel;
            
            % Check zero vectors in input data.
            if any(all(data==0))
                error('Input data contains a zero vector. Gradients cannot be computed for these vectors.')
            end
            
            % Sparsify input data. 
            disp(['Running with sparsity parameter: ' num2str(p.Results.sparsity)]);
            sparse_data = data;
            sparse_data(data < prctile(data,p.Results.sparsity)) = 0; 
            
            % If a custom function, just run the custom function.
            if isa(kernel,'function_handle')
                kernel_data = kernel(data);
                return
            end
            
            switch kernel
                case 'None'
                    if p.Results.sparsity ~= 0
                        warning('Using a none kernel with a matrix sparsification will likely lead to an asymmetric matrix. Consider setting the sparsity parameter to 0.');
                    end
                    kernel_data = sparse_data;
                case {'Pearson','Spearman'}
                    kernel_data = corr(sparse_data,'type',kernel);
                case 'Gaussian'
                    disp(['Running with gamma parameter: ' num2str(p.Results.gamma) '.']);
                    kernel_data = exp(-p.Results.gamma .* squareform(pdist(sparse_data').^2));
                case {'Cosine Similarity','Normalized Angle'}
                    cosine_similarity = 1-squareform(pdist(sparse_data','cosine'));
                    switch kernel
                        case 'Cosine Similarity'
                            kernel_data = cosine_similarity;
                        case 'Normalized Angle'
                            kernel_data = 1-acos(cosine_similarity)/pi;
                    end
                otherwise
                    error('Unknown kernel method');
            end
                        
            % Check for negative numbers.
            if any(kernel_data(:) < 0)
                disp('Found negative numbers in the kernel matrix. These will be set to zero.');
                kernel_data(kernel_data < 0) = 0; 
            end
            
            % Check for vectors of zeros. 
            if any(all(kernel_data == 0))
                error(['After thresholding, a complete vector in the kernel ' ...
                    'matrix consists of zeros. Consider using a kernel that ' ...
                    'does not allow for negative numbers (e.g. normalized angle).']); 
            end
            
            if ~issymmetric(kernel_data)
                if max(max(abs(kernel_data - kernel_data'))) < p.Results.tolerance
                    kernel_data = tril(kernel_data) + tril(kernel_data,-1)';
                else
                    error('Asymmetry in the affinity matrix is too large. Increase the tolerance. Alternatively, are you using a ''none'' kernel with a non-zero sparsity parameter? This may result in errors.');
                end
            end
        end
        % -------------------------------------
        % -------------------------------------
        % -------------------------------------
        function [embedding, lambda] = approaches(obj, data, varargin)
            % [embedding, result] = approaches(obj, data, varargin)
            %
            % Computes the embedded data. This function should not be
            % called directly; it should only be called from the
            % run_analysis method
            %% Check input arguments.
            
            p = inputParser;
            addParameter(p, 'alpha'         , 0.5       , @isnumeric    );
            addParameter(p, 'diffusion_time' , 0         , @isnumeric    );
            
            % Parse the input
            parse(p, varargin{:});
            in = p.Results;
            
            % If a custom function, just run the custom function.
            if isa(obj.method.approach,'function_handle')
                embedding = obj.method.approach(data);
                return
            end
            
            if ~issymmetric(data)
                if max(max(abs(data - data'))) > eps % floating point issues. 
                    error('Affinity matrix is not symmetric.')
                else
                    data = tril(data) + tril(data,-1)'; % Attempt to force symmetry
                end
            end
            
            % Check if the graph is connected. Large matrices may remain
            % floating point asymmetric despite the above check, so only
            % use lower.
            if ~all(conncomp(graph(abs(data),'lower')) == 1) 
                error('Graph is not connected.')
            end
            
            %% Embedding.
            % Set the random state for reproducibility.
            if ~isnan(obj.random_state) 
                rng(obj.random_state)
            end
            
            % Run manifold learning
            switch obj.method.approach
                case 'Principal Component Analysis'
                    [~, embedding, ~, ~, lambda] = pca(data);
                    embedding = embedding(:,1:obj.method.n_components);
                case 'Laplacian Eigenmap'
                    disp(['Requested ' num2str(obj.method.n_components) ' components.']);
                    [embedding, lambda] = laplacian_eigenmaps(data, obj.method.n_components);
                case 'Diffusion Embedding'
                    disp(['Running with alpha parameter: ' num2str(in.alpha)]);
                    disp(['Running with diffusion time: ' num2str(in.diffusion_time)]);
                    [embedding, lambda] = diffusion_mapping(data, obj.method.n_components, in.alpha, in.diffusion_time);
                otherwise
                    error('Unknown manifold technique.');
            end
        end
    end
end
