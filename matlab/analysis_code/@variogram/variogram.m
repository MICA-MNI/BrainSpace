classdef variogram
% VARIOGRAM Generates brain maps with similar spatial autocorrelation.
%
%   obj = VARIOGRAM(D,varargin) initializes the variogram object based on
%   symmetric distance matrix D. Valid name-value pairs are:
%   deltas : 1-dimensional vector, default [0.1,0.2,...,0.9]
%       Proportion of neighbors to include for smoothing, in (0, 1]
%   kernel : character vector, default 'exp'
%       Kernel with which to smooth permuted maps:
%           'gaussian' : Gaussian function.
%           'exp' : Exponential decay function.
%           'invdist' : Inverse distance.
%           'uniform' : Uniform weights (distance independent).
%   pv : scalar (float), default 25
%       Percentile of the pairwise distance distribution at which to
%       truncate during variogram fitting.
%   nh : scalar (integer), default 25
%       Number of uniformly spaced distances at which to compute
%       variograms.
%   resample : logical, default false
%       Resample surrogate maps' values from target brain map
%   b : scalar (float) or nan, default nan
%       Gaussian kernel bandwidth for variogram smoothing. If None, set to
%       three times the spacing between variogram x-coordinates.
%   random_state : any valid input for the rng() function. Set to nan for
%       no random initialization.
%   ns : scalar (integer) or inf, default inf. 
%       Number of samples to use when subsampling the brainmap. Set to inf
%       to use the entire brainmap. 
%   knn : scalar (integer), default 1000
%       Number of nearest neighbours to use when smoothing the map. knn
%       must be smaller than ns. 
%   num_workers: number of workers in the parallel pool, default 0.
%       Sets the number of workers in a parallel pool. Only available if
%       the parallel processing toolbox is installed. Note that if another
%       pool is running with a different number of workers, then this pool
%       will be shut down. The parallel pool is not closed at the end of
%       the script. 
%
%   Public Methods: surrogates = obj.fit(x,n) generates n surrogate maps
%   for brain map x. x must have the same length as D. n is set to 1000 by
%   default. 
%
%   Example usage:
%   x = rand(100,1);
%   D = rand(100);
%   D = D + D'; % make distance matrix symmetric.
%   obj = variogram(D);
%   surrogates = obj.fit(x); 
%
%   ADD A READTHEDOCS LINK!

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
        num_workers
    end
    
    properties(Hidden, Access = private)
        disort
    end

    %% Public Methods
    methods
        %%% Constructor %%%
        function obj = variogram(varargin)
            % Deal with input.
            is_square_numeric = @(x)size(x,1)==size(x,2) && isnumeric(x) && numel(size(x))==2;
            p = inputParser;
            addRequired( p, 'D'                         , is_square_numeric);
            addParameter(p, 'deltas'    , 0.1:0.1:0.9   , @(x) all(x>0 & x<=1));
            addParameter(p, 'kernel'    , 'exp'         , @valid_kernel);
            addParameter(p, 'pv'        , 25            , @(x) x>0 && x<=100 && isscalar(x));
            addParameter(p, 'nh'        , 25            , @(x)isscalar(x) && isnumeric(x));
            addParameter(p, 'resample'  , false         , @islogical);
            addParameter(p, 'b'         , nan           , @(x)isscalar(x) && isnumeric(x));
            addParameter(p, 'random_state', nan);
            addParameter(p, 'ns'        , inf           , @(x)isscalar(x) && isnumeric(x));
            addParameter(p, 'knn'       , 1000          , @(x)isscalar(x) && isnumeric(x));
            addParameter(p, 'num_workers', 0             , @(x)isscalar(x) && isnumeric(x));
            
            % Assign input to object properties. 
            parse(p, varargin{:});
            f = fieldnames(p.Results);
            for ii = 1:numel(f)
                obj.(f{ii}) = p.Results.(f{ii});
            end  
            
            if ~isinf(obj.ns) && obj.ns >= size(obj.D,1)
                error('The number of samples must be smaller than the number of nodes or infinite.');
            end
            
            if obj.num_workers ~= 0 && ~isnan(obj.random_state)
                warning('Due to the way that random number initialization works on parallel pools, the results of this script will not be identical on separate runs even with the random state defined.');
            end
        end
        
        %%% Fitting function %%%
        function surrs = fit(obj,x,n)
            % Start the parallel pool.
            if obj.num_workers ~= 0
                if exist('parpool.m', 'file')  
                    p = gcp('nocreate');
                        if isempty(p)
                            parpool(obj.num_workers);   
                        elseif p.NumWorkers ~= obj.num_workers
                            delete(p);
                            parpool(obj.num_workers);
                        end
                else
                    warning('Could not find the parallel processing toolbox. Continuing without parallel processing.')
                end
            end
            
            % Set default n to 1000.
            if ~exist('n','var')
                n = 1000;
            end
            
            % Set random state.
            if ~isnan(obj.random_state)
                rng(obj.random_state)
            end
                        
            % Initialize variables.
            surrs = zeros(size(x,1),n);
            
            % Only compute the sorted distance matrix once because sorting
            % large matrices takes a while.
            [~,obj.disort] = sort(obj.D,2);
            
            % Compute true variogram (dense). 
            % Put inside a temporary structure to deal with parfor issues;
            % parfor doesn't accept variables by the same name being
            % defined both inside and outside the loop, even if they are
            % defined in mutually exclusive if statements. 
            if isinf(obj.ns)
                v_out = obj.compute_variogram(x);
                [tmp.utrunc,tmp.uidx,tmp.h] = obj.prepare_smooth_variogram(); 
                tmp.smvar = obj.smooth_variogram(tmp.utrunc,tmp.uidx,v_out,tmp.h);
            else
                % Par-for wants a variable called "tmp" even if its unused.
                tmp = struct();
            end
           
            % Generate surrograte maps.
            parfor (ii = 1:n, obj.num_workers)
            %for ii = 1:n
                % Initialize variables
                aopt = nan;
                bopt = nan;
                rsquared_best = -inf; 
                idxopt = nan;
                
                % Compute true variogram and permuted map (sampled).
                if ~isinf(obj.ns) %#ok<PFBNS>
                    perm = randperm(size(x,1),obj.ns);
                    v = obj.compute_variogram(x(perm));
                    [utrunc,uidx,h] = obj.prepare_smooth_variogram(perm);
                    smvar = obj.smooth_variogram(utrunc,uidx,v,h);                 
                else
                    perm = nan; % This line prevents MATLAB from throwing a par-for warning - the line itself doesn't do anything. 
                    utrunc = tmp.utrunc; %#ok<PFBNS>
                    uidx = tmp.uidx;
                    h = tmp.h;
                    smvar = tmp.smvar;
                end
                
                % Compute permuted map.
                [x_perm,~] = obj.permute_map(x,~isnan(x));
                
                for jj = 1:numel(obj.deltas)
                    % Smooth permuted map.
                    sm_x_perm = obj.smooth_map(x_perm,jj);
                    
                    % Calculate empirical variogram
                    if ~isinf(obj.ns)
                        vperm = obj.compute_variogram(sm_x_perm(perm));
                    else
                        vperm = obj.compute_variogram(sm_x_perm);
                    end
                    smvar_perm = obj.smooth_variogram(utrunc,uidx,vperm,h);
                    
                    % Fit linear regression between smoothed variograms
                    [alpha,beta,rsquared] = obj.local_regression(smvar_perm,smvar);
                    
                    if rsquared > rsquared_best
                        aopt = alpha; 
                        bopt = beta;
                        idxopt = jj;
                    end      
                end
                
                % Transform and smooth permuted map using best-fit parameters.
                sm_xperm_best = obj.smooth_map(x_perm,idxopt);
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
            if isa(type, 'function_handle')
                smooth_d = type(d); 
            else
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

function is_valid = valid_kernel(x)
    is_valid = isa(x, 'function_handle');
    if ~is_valid
        is_valid = ismember(lower(x), {'exp','gaussian','uniform','invdist'});
    end
end
