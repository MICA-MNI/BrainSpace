function Y_rand = moran_randomization(Y,MEM,n_rep,varargin)
% MORAN_RANDOMIZATION   Null model for spatially auto-correlated data.
%   y_rand = MICA_MORAN_RANDOMIZATION(y,MEM,n_rep,varargin) computes
%   random values x_rand with similar spatial properties as the input data
%   x. x is a n-by-1 vector of observations, MEM are the Moran
%   eigenvectors, n_rep is a scalar denoting the amount of randomized
%   datasets in the output. 
%
%   Valid name-value pairs are:
%       'procedure' : either 'singleton', or 'pair'.
%       'joint' : either true or false.
%       'random_state' : any argument accepted by rng() or nan. 
%
%   The procedure can be either 'singleton' or 'pair'. 
%
%   References:
%       [1] Wagner, H. H., & Dray, S. (2015). Generating spatially
%           constrained null models for irregularly spaced data using Moran
%           spectral randomization methods. Methods in Ecology and
%           Evolution, 6(10), 1169-1178.
%
%   See also: COMPUTE_MEM
%
%   For more information, please consult our <a
% href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/main_functionality/moran_randomization.html">ReadTheDocs</a>.


%% Deal with the input.  
p = inputParser;
addParameter(p, 'procedure', 'singleton');
addParameter(p, 'joint', true, @islogical);
addParameter(p, 'random_state', nan);
parse(p, varargin{:});

procedure = p.Results.procedure;
joint = p.Results.joint;

if ~isnan(p.Results.random_state)
    rng(p.Results.random_state); 
end

procedure = lower(procedure);
if ~any(procedure == ["singleton","pair"])
    error('Procedure must be either singleton or pair.');
end

% Get correlations between x and the eigenvectors. 
r_xV = corr(Y,MEM,'row','pairwise')';

n = size(MEM,1);

% Select which procedure to use. 
switch procedure
    case 'singleton'
        % Run singleton procedure.
        a = singletonProcedure(r_xV,n_rep,joint);
    case 'pair'
        if size(Y,2) > 1
            error('Multivariate resampling has not been implemented for the pair procedure.');
        end
        a = nan(size(n,n_rep));
        rem = mod(n,2);
        pairs = nan(2,(n-rem)/2,n_rep);
        for loop = 1:n_rep
            % Make pairs and find the remainder. 
            shuffle = randperm(n);
            pairs(:,:,loop) = reshape(shuffle(1:end-rem),2,[]);
            solo  = shuffle(end-rem+1:end);

            % For each pair, compute the power spectrum a (See practical
            % implementation section, Ref [1]).
            ix = pairs(1,:,loop);
            iy = pairs(2,:,loop);
            R2_xVk = r_xV(ix).^2 + r_xV(iy).^2; 
            R_xVk = sqrt(R2_xVk);
            angle = rand(size(R_xVk))*2*pi; % Relaxes condition C2'. 
            a(ix,1,loop) = R_xVk .* cos(angle);
            a(iy,1,loop) = R_xVk .* sin(angle); 
            
            % For remaining data, use singleton procedure. 
            a(solo,1,loop) = singletonProcedure(r_xV(solo),1,joint);
        end
end

% Compute the simulated data, match the mean and standard deviation. 
Y_rand = nan(size(Y,1),size(Y,2),n_rep);
for ii = 1:size(a,2)
    Y_rand(:,ii,:) = mean(Y(:,ii)) + std(Y(:,ii)) .* ((n-1)^0.5*MEM*squeeze(a(:,ii,:)));
end
end

function a = singletonProcedure(x,numRandomizations,joint)
    % Singleton procedure simply randomizes the sign of r_xV.
    if joint
        sign = randi([0,1], size(x,1), 1, numRandomizations)*2 - 1;
    else
        sign = randi([0,1], size(x,1), size(x,2), numRandomizations)*2 - 1;
    end
    a = sign .* x;  
end
