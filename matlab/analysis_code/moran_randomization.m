function Y_rand = moran_randomization(Y,V,nRandomizations,procedure,joint)
% MORAN_RANDOMIZATION   Null model for spatially auto-correlated data.
%   y_rand = MICA_MORAN_RANDOMIZATION(y,W,nRandomizations,procedure)
%   computes random values x_rand with similar spatial properties as the
%   input data x. x is a n-by-1 vector of observations, V are the Moran
%   eigenvectors, nRandomizations is a scalar denoting the amount of
%   randomized datasets in the output, and procedure is the method used to
%   compute randomized data. 
%
%   The procedure can be either 'singleton' or 'pair'. 
%
%   Written by Reinder Vos de Wael (Mar, 2019)
%
%   References:
%       [1] Wagner, H. H., & Dray, S. (2015). Generating spatially
%           constrained null models for irregularly spaced data using Moran
%           spectral randomization methods. Methods in Ecology and
%           Evolution, 6(10), 1169-1178.
%
%   See also: COMPUTE_MEM

%% Deal with the input.  

procedure = lower(procedure);
if ~any(procedure == ["singleton","pair"])
    error('Procedure must be either singleton or pair.');
end

% Get correlations between x and the eigenvectors. 
r_xV = corr(Y,V,'row','pairwise')';

n = size(V,1);

% Select which procedure to use. 
switch procedure
    case 'singleton'
        % Run singleton procedure.
        a = singletonProcedure(r_xV,nRandomizations,joint);
    case 'pair'
        if size(Y,2) > 1
            error('Multivariate resampling has not been implemented for the pair procedure.');
        end
        a = nan(size(n,nRandomizations));
        rem = mod(n,2);
        pairs = nan(2,(n-rem)/2,nRandomizations);
        for loop = 1:nRandomizations
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
Y_rand = nan(size(Y,1),size(Y,2),nRandomizations);
for ii = 1:size(a,2)
    Y_rand(:,ii,:) = mean(Y(:,ii)) + std(Y(:,ii)) .* ((n-1)^0.5*V*squeeze(a(:,ii,:)));
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
