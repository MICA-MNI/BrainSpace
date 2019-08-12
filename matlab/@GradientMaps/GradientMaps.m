classdef GradientMaps
    % obj = BrainSpace(varargin)
    %
    % Class definition and constructor for BrainSpace. The following
    % name-value pairs are allowed as input.
    %
    %       - kernel (default: normalized angle)
    %           - 'pearson'
    %           - 'spearman'
    %           - 'gaussian'
    %           - 'normalized angle'
    %           - 'cosine similarity'
    %           - 'none'
    %       - manifold (default: diffusion embedding)
    %           - 'diffusion embedding'
    %           - 'laplacian eigenmap'
    %           - 'principal component analysis'
    %       - alignment (default: false)
    %           - true
    %           - false
    %       - n_components (default: 2)
    %           - Any positive integer.
    %       - random_state (default: 0)
    %           - Any valid input for MATLAB's "rng" function.
    %
    %
    % If you use this toolbox, please cite ...
    %
    % For questions regarding this toolbox, contact the corresponding
    % author of [[[ Publication ]]
    %
    % See also: GRADIENT.ALIGN GRADIENT.FIT, GRADIENT.MSR,
    % GRADIENT.SPINTEST, BRAINSPACE_SAMPLE_RUN.
    
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
            addParameter(p, 'manifold', 'diffusion embedding', in_fun);
            addParameter(p, 'alignment', 'none', in_fun);
            addParameter(p, 'n_components', 2, @isnumeric);
            addParameter(p, 'random_state', 0);
            
            parse(p, varargin{:});
            R = p.Results;
            
            % Set the properties
            obj = obj.set( ...
                'kernel',       R.kernel, ...
                'manifold',     R.manifold, ...
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
                                case 'pearson'
                                    obj.method.kernel = 'Pearson';
                                case 'spearman'
                                    obj.method.kernel = 'Spearman';
                                case 'gaussian'
                                    obj.method.kernel = 'Gaussian';
                                case {'cs','cosine','cosine similarity','cossim','cosine_similarity'}
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
                                case {'pca','principalcomponentananalysis','principal component analysis'}
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
                                case 'none'
                                    obj.method.alignment = 'None';
                                case {'pa','procrustes','procrustes analysis','procrustesanalysis'}
                                    obj.method.alignment = 'Procrustes Analysis';
                                case {'j','joint','jointalignment'}
                                    obj.method.alignment = 'Joint Alignment';
                            end
                            change_string{end+1} = (['Set the alignment to: ' obj.method.alignment '.']);
                        end
                        
                    case 'random_state'
                        obj.random_state = varargin{ii+1};
                        change_string{end+1} = ['Set the random state initialization to: ' num2str(varargin{ii+1}) '.'];
                        
                    case 'n_components'
                        obj.method.n_components = varargin{ii+1};
                        change_string{end+1} = ['Set the number of requested components to: ' num2str(varargin{ii+1}) '.'];
                        
                    otherwise
                        error('Unknown property. Valid properties are: ''connectivitymatrix'', ''kernel'', ''manifold'', and ''nullmodel''.');
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
            
            % If a custom function, just run the custom function.
            if isa(kernel,'function_handle')
                kernel_data = kernel(data);
                return
            end
            
            sparse_data = data .* bsxfun(@gt, data, prctile(data,p.Results.sparsity));
            switch kernel
                case 'None'
                    kernel_data = sparse_data;
                case {'Pearson','Spearman'}
                    kernel_data = corr(sparse_data,'type',kernel);
                case 'Gaussian'
                    disp(['Running with gamma parameter: ' num2str(p.Results.gamma) '.']);
                    kernel_data = exp(-p.Results.gamma .* squareform(pdist(sparse_data').^2));
                case {'Cosine Similarity','Normalized Angle'}
                    disp(['Running with sparsity parameter: ' num2str(p.Results.sparsity)]);
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
            
            kernel_data(kernel_data < 0) = 0; 
            
            if ~issymmetric(kernel_data)
                if max(max(abs(kernel_data - kernel_data'))) < p.Results.tolerance
                    kernel_data = tril(kernel_data) + tril(kernel_data,-1)';
                else
                    error('Asymmetry in the affinity matrix is too large. Increase the tolerance. If this does not help, then please report this as a bug.');
                end
            end
        end
        % -------------------------------------
        % -------------------------------------
        % -------------------------------------
        function [embedding, result] = approaches(obj, data, varargin)
            % [embedding, result] = approaches(obj, data, varargin)
            %
            % Computes the embedded data. This function should not be
            % called directly; it should only be called from the
            % run_analysis method
            %% Check input arguments.
            
            p = inputParser;
            addParameter(p, 'alpha'         , 0.5       , @isnumeric    );
            addParameter(p, 'diffusionTime' , 0         , @isnumeric    );
            
            % Parse the input
            parse(p, varargin{:});
            in = p.Results;
            
            % If a custom function, just run the custom function.
            if isa(obj.method.approach,'function_handle')
                embedding = obj.method.approach(data);
                return
            end
            
            if ~issymmetric(data)
                error('Affinity matrix is not symmetric.')
            end
            
            % Check if the graph is connected.
            if ~all(conncomp(graph(abs(data))) == 1)
                error('Graph is not connected.')
            end
            
            %% Embedding.
            % Set the random state for reproducibility.
            rng(obj.random_state);
            
            % Run manifold learning
            switch obj.method.approach
                case 'Principal Component Analysis'
                    [~, embedding, ~, ~, result] = pca(data);
                    embedding = embedding(:,1:obj.method.n_components);
                case 'Laplacian Eigenmap'
                    disp(['Requested ' num2str(obj.method.n_components) ' components.']);
                    [embedding, result] = laplacian_eigenmap(data, obj.method.n_components);
                case 'Diffusion Embedding'
                    disp(['Running with alpha parameter: ' num2str(in.alpha)]);
                    disp(['Running with diffusion time: ' num2str(in.diffusionTime)]);
                    [embedding, result] = diffusion_embedding(data, obj.method.n_components, in.alpha, in.diffusionTime);
                otherwise
                    error('Unknown manifold technique.');
            end
        end
    end
    %%
%     methods(Access = public)
%         function fit_help(obj)
%             
%             % Kernel arguments
%             if isa(obj.method.kernel,'function_handle')
%                 disp('You''ve provided a custom function handle for the kernel, no name-value arguments are accepted.');
%             else
%                 disp(['The following are optional name-value pairs to ''fit'' for the ' obj.method.kernel ' kernel.']);
%                 if strcmp(obj.method.kernel,'None')
%                     disp('This kernel accepts no additional arguments.');
%                 else
%                     disp('This kernel accepts the name ''sparsity'' with a numeric value between [0 100] (default: 90).');
%                     disp('This kernel accepts the name ''tolerance'' with any positive numeric value (default: 1e-6).');
%                 end
%                 if strcmp(obj.method.kernel,'Gaussian')
%                     disp('This kernel accepts the name ''gamma'' with a numeric value (default: 1/size(data,1)).');
%                 end
%             end
%             disp(' ')
%             
%             % Manifold arguments
%             if isa(obj.method.kernel,'function_handle')
%                 disp('You''ve provided a custom function handle for the kernel, no name-value arguments are accepted.');
%             else
%                 disp(['The following are optional name-value pairs to ''fit'' for the ' obj.method.approach ' manifold.']);
%                 if strcmp(obj.method.approach,'Principal Component Analysis')
%                     disp('This manifold accepts no additional arguments.');
%                 elseif strcmp(obj.method.approach,'Laplacian Eigenmap')
%                     disp('This manifold accepts no additional arguments.');
%                 elseif strcmp(obj.method.approach,'Diffusion Embedding')
%                     disp('This manifold accepts the name ''alpha'' with a numeric value between [0 1] (default: 0.5).');
%                     disp('This manifold accepts the name ''diffusiontime'' with any positive numeric value or zero (default: 0).');
%                 end
%             end
%             disp(' ');
%         end
%     end
end