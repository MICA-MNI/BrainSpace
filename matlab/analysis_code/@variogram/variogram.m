classdef variogram
    %% Properties 
    properties(SetAccess = private)
        D
        deltas
        kernel
        pv
        nh
        resample
        b
        random_state
        ns
        knn
        verbose
    end
    
    properties(Hidden, Access = private)
        disort
        u_prctile     
    end

    %% Public Methods
    methods
        %%% Constructor %%%
        function obj = variogram(varargin)
            % Deal with input.
            is_square_numeric = @(x)size(x,1)==size(x,2) && isnumeric(x) && numel(size(x))==2;
            valid_kernel = @(x) ismember(lower(x),{'exp','gaussian','uniform','invdist'});
            p = inputParser;
            addRequired( p, 'D'                         , is_square_numeric);
            addParameter(p, 'deltas'    , 0.1:0.1:0.9   , @(x) all(x>0 & x<=1));
            addParameter(p, 'kernel'    , 'exp'         , valid_kernel);
            addParameter(p, 'pv'        , 25            , @(x) x>0 && x<=100 && isscalar(x));
            addParameter(p, 'nh'        , 25            , @(x)isscalar(x) && isnumeric(x));
            addParameter(p, 'resample'  , false         , @islogical);
            addParameter(p, 'b'         , nan           , @(x)isscalar(x) && isnumeric(x));
            addParameter(p, 'random_state', nan);
            addParameter(p, 'ns'        , inf           , @(x)isscalar(x) && isnumeric(x));
            addParameter(p, 'knn'       , 1000          , @(x)isscalar(x) && isnumeric(x));
            addParameter(p, 'verbose'   , false         , @(x)islogical(x) && isscalar(x));
            
            % Assign input to object properties. 
            parse(p, varargin{:});
            f = fieldnames(p.Results);
            for ii = 1:numel(f)
                obj.(f{ii}) = p.Results.(f{ii});
            end  

            if obj.ns <= obj.knn
                error('The number of samples must be higher than the number of nearest neighbors.');
            end
        end
        
        %%% Fitting function %%%
        function surrs = fit(obj,x,n)
            % Set default n to 1000.
            if ~exist('n','var')
                n = 1000;
            end
            
            % Set random state.
            if ~isnan(obj.random_state)
                rng(obj.random_state)
            end
                        
            % Initialize variables.
            alpha = zeros(numel(obj.deltas),1);
            beta = zeros(numel(obj.deltas),1);
            rsquared = zeros(numel(obj.deltas),1);
            surrs = zeros(size(x,1),n);
            
            % Only compute the sorted distance matrix once because sorting
            % large matrices takes a while.
            [~,obj.disort] = sort(obj.D,2);
            
            % Compute true variogram (dense).
            if isinf(obj.ns)
                v = obj.compute_variogram(x);
                [utrunc,uidx,h] = obj.prepare_smooth_variogram(); 
                smvar = obj.smooth_variogram(utrunc,uidx,v,h);
            end
            
            % Generate surrograte maps.
            for ii = 1:n
                if obj.verbose
                    if ii ~= 1 && ii ~=n
                        fprintf(repmat('\b',1,s));
                    end
                    s = fprintf('Generating surrogate map %d of %d.\r',ii,n);
                end
                % Compute true variogram and permuted map (sampled).
                if ~isinf(obj.ns)
                    idx = randperm(size(x,1),obj.ns);
                    v = obj.compute_variogram(x(idx));
                    [utrunc,uidx,h] = obj.prepare_smooth_variogram(idx);
                    smvar = obj.smooth_variogram(utrunc,uidx,v,h);                 
                end
                
                % Compute permuted map.
                [x_perm,~] = obj.permute_map(x,~isnan(x));
                
                for jj = 1:numel(obj.deltas)
                    % Smooth permuted map.
                    sm_x_perm = obj.smooth_map(x_perm,jj);
                    
                    % Calculate empirical variogram
                    if ~isinf(obj.ns)
                        vperm = obj.compute_variogram(sm_x_perm(idx));
                    else
                        vperm = obj.compute_variogram(sm_x_perm);
                    end
                    smvar_perm = obj.smooth_variogram(utrunc,uidx,vperm,h);
                    
                    % Fit linear regression between smoothed variograms
                    [alpha(jj),beta(jj),rsquared(jj)] = obj.local_regression(smvar_perm,smvar);
                end
                
                % Select best-fit model and regression parameters.
                [~,idx] = max(rsquared);
                %dopt = obj.deltas(idx);
                aopt = alpha(idx);
                bopt = beta(idx);
                
                % Transform and smooth permuted map using best-fit parameters.
                sm_xperm_best = obj.smooth_map(x_perm,idx);
                surrs(:,ii) = sqrt(abs(bopt)) * sm_xperm_best + sqrt(abs(aopt)) * randn(numel(x),1);
            end
            
            % Resample to the input values. 
            if obj.resample
                sorted_map = sort(x);
                for ii = 1:n
                    [~,idx_resample] = ismember(surrs(:,ii),sort(surrs(:,ii)));
                    surrs(:,ii) = sorted_map(idx_resample);
                end
            end
        end
    end
    
    %% Private methods
    methods(Access = private)
        function smvar = smooth_variogram(obj,utrunc,uidx,v,h)  
            % Smooths the variograms. 
            
            vtrunc = v(uidx);
                        
            % Auto-estimate b if it's nan. 
            if isnan(obj.b)
                b_valid = 3 * (h(end)-h(1)) / (numel(h)-1);
            else
                b_valid = obj.b;
            end
            
            du = abs(utrunc - h);
            w = exp(-(2.68 * du / b_valid).^2/2);
            smvar = sum(w.*vtrunc) ./ sum(w);
        end
        
        function [utrunc,uidx,h] = prepare_smooth_variogram(obj,idx)
            % Originally part of smooth_variogram. 
            % Set to a separate function as this only needs to run once
            % per permutation. 
            if nargin > 1
                D_subsample = obj.D(idx,idx);
            else
                D_subsample = obj.D;
            end
            
            upper_triangle = triu(ones(size(D_subsample),'logical'),1);
            u = D_subsample(upper_triangle);
            uidx = u < prctile(u,obj.pv);
            utrunc = u(uidx);
            h = linspace(min(utrunc(:)),max(utrunc(:)),obj.nh);
        end
        
        function sm_map = smooth_map(obj,x,idx_delta)
            % Smooth x using delta proportion of nearest neighbors.     
            
            % Get distances and values of k nearest neighbours
            if isinf(obj.ns)
                k = floor(obj.deltas(idx_delta) * size(obj.D,1));
            else
                k = floor(obj.deltas(idx_delta) * obj.knn);
            end
            jkn = obj.disort(:,2:k+1);
            jkn_idx = jkn + (0:size(obj.D,1):(size(jkn,1)-1)*size(obj.D,1))'; % Convert row indices to matrix indices. 
            dkn = obj.D(jkn_idx);            
            xkn = x(jkn);
            
            % Compute kernel weights and smoothed map.
            weights = obj.smoothing_kernel(obj.kernel, dkn);
            sm_map = sum(weights .* xkn,2) ./ sum(weights,2);
        end
    end
    
    methods(Static, Access = private)
        function [data_perm,mask_perm] = permute_map(data,mask)
            % Return randomly permuted brain map.
            perm_idx = randperm(numel(data));
            data_perm = data(perm_idx);
            if nargin > 1
                mask_perm = mask(perm_idx);
            end
        end
        
        function v = compute_variogram(x)
            % Computes variograms based on input data x.
            diff_ij = x-x';
            upper_triangle = triu(ones(size(diff_ij),'logical'),1);
            v = 0.5 * diff_ij(upper_triangle).^2;
        end
        
        function smooth_d = smoothing_kernel(type,d)
            % Applies a smoothing kernel to distance matrix d.
            switch lower(type)
                case 'exp'
                    smooth_d = exp(-d ./ max(d,[],2));
                case 'gaussian'
                    smooth_d = exp(-1.25 * (d ./ max(d,[],2)));
                case 'invdist'
                    smooth_d = d.^-1;
                case 'uniform'
                    smooth_d = ones(size(d)) / size(d,1);
                otherwise
                    error('Unknown kernel type')
            end
        end
        
        function [intercept,slope,rsquared] = local_regression(x,y)
            % Computes the intercept, beta and resquared of a regression of
            % x on y. 
            
            % Ascertain column vectors.
            if ~isvector(x) || ~isvector(y)
                error('x and y must be vectors');
            end              
            x = x(:); y = y(:);
            
            % Compute betas
            x = [ones(size(x,1),1),x];
            B = (x' * x) ^ -1 * x' * y; 
            intercept = B(1); 
            slope = B(2);
            
            % Compute R^2
            rsquared = 1 - sum((y - (intercept + slope * x(:,2))).^2) / sum((y-mean(y)).^2); 
        end
    end
end
