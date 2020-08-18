classdef variogram
    
    properties
        D
        deltas
        kernel
        pv
        nh
        resample
        b
        random_state
        n
        dense
    end
    
    methods
        function obj = variogram(varargin)
            % Deal with input.
            is_square_numeric = @(x)size(x,1)==size(x,2) && isnumeric(x) && numel(size(x))==2;
            valid_kernel = @(x) ismember(lower(x),{'exp','gaussian','uniform','invdist'});
            p = inputParser;
            addRequired( p, 'D'                         , is_square_numeric);
            addParameter(p, 'deltas'    , 0.1:0.1:0.9   , @(x) x>0 && x<=1);
            addParameter(p, 'kernel'    , 'exp'         , valid_kernel);
            addParameter(p, 'pv'        , 25            , @(x) x>0 && x<=100 && isscalar(x));
            addParameter(p, 'nh'        , 25            , @(x)isscalar(x) && isnumeric(x));
            addParameter(p, 'resample'  , false         , @islogical);
            addParameter(p, 'b'         , nan           , @(x)isscalar(x) && isnumeric(x));
            addParameter(p, 'random_state', nan);
            addParameter(p, 'dense'     , true          , @islogical);
            
            % Assign input to object properties. 
            parse(p, varargin{:});
            f = fieldnames(p.Results);
            for ii = 1:numel(f)
                obj.(f{ii}) = p.Results.(f{ii});
            end  
            
            % Check for unimplemented stuff
            if ~obj.dense 
                error('The sparse implementation has not been ported to MATLAB yet.')
            end 
        end
        
        %% Fitting functions
        function surrs = fit(obj,x,n)
            % Set default n to 1000.
            if ~exist('n','var')
                n = 1000;
            end
            
            % Set random state
            if ~isnan(obj.random_state)
                rng(obj.random_state)
            end
            
            mask = ~isnan(x);
            
            % Initialize variables
            alpha = zeros(numel(obj.deltas),1);
            beta = zeros(numel(obj.deltas),1);
            rsquared = zeros(numel(obj.deltas),1);
            surrs = zeros(size(x,1),n);
            
            % Compute true variogram
            v = obj.compute_variogram(x);
            smvar = obj.smooth_variogram(v);
            
            % Generate surrograte maps.
            for ii = 1:n
                % Create a permuted map
                [x_perm,~] = obj.permute_map(x,mask);
                for jj = 1:numel(obj.deltas)
                    % Smooth permuted map.
                    sm_x_perm = obj.smooth_map(x_perm,jj);
                    
                    % Calculate empirical variogram
                    vperm = obj.compute_variogram(sm_x_perm);
                    smvar_perm = obj.smooth_variogram(vperm);
                    
                    % Fit linear regression between smoothed variograms
                    [alpha(jj),beta(jj),rsquared(jj)] = obj.local_regression(smvar_perm,smvar);
                end
                
                % Select best-fit model and regression parameters.
                [~,idx] = min(rsquared);
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
                    [~,idx] = ismember(surrs(:,ii),sort(surrs(:,ii)));
                    surrs(:,ii) = sorted_map(idx);
                end
            end
        end
        
        function smvar = smooth_variogram(obj,v)            
            % Truncate u and v. 
            upper_triangle = triu(ones(size(obj.D),'logical'),1);
            u = obj.D(upper_triangle);
            uidx = u < prctile(u,obj.pv);
            
            utrunc = u(uidx);
            vtrunc = v(uidx);
            
            h = linspace(min(utrunc(:)),max(utrunc(:)),obj.nh);
            
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
        
        function sm_map = smooth_map(obj,x,deltaidx)
            % Smooth x using delta proportion of nearest neighbors.
            
            % Find indices of first kmax elements of each for of dist matrix
            [~,disort] = sort(obj.D,2);
            
            % Get distances and values of k nearest neighbours
            k = floor(obj.deltas(deltaidx) * size(obj.D,1));
            jkn = disort(:,2:k+1);
            jkn_idx = jkn + (0:size(obj.D,1):(size(jkn,1)-1)*size(obj.D,1))'; % Convert row indices to matrix indices. 
            dkn = obj.D(jkn_idx);            
            xkn = x(jkn);
            
            % Compute kernel weights and smoothed map.
            weights = obj.smoothing_kernel(obj.kernel, dkn);
            sm_map = sum(weights .* xkn,2) ./ sum(weights,2);
        end
    end
    
    %% Static methods.
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
