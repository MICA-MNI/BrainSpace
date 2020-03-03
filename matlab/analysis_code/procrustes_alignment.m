function aligned = procrustes_alignment(gradients,varargin)
% PROCRUSTES_ALIGNMENT   Performs a Procrustes alignment between gradients.
%
%   aligned = PROCRUSTES_ALIGNMENT(gradients,varargin) performs a singular
%   value decomposition of the vectors in cell array gradients to align
%   them. Valid name-value pairs are  the number of iterations
%   'nIterations' (default: 10) and the reference (default: first cell of
%   the gradients array). 
%
%   For more information, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/support_functions/procrustes_alignment.html">ReadTheDocs</a>.

% Parse input
p = inputParser;
addParameter(p, 'nIterations', 10, @isnumeric);
addParameter(p, 'reference',nan);
parse(p, varargin{:});

% Grab embeddings.
for ii = 1:numel(gradients)
    embs{ii} = gradients{ii};
end

% Set the first alignment target.
if isnan(p.Results.reference)
    first_target = embs{1};
else
    first_target = p.Results.reference;
end

realigned = procrustes_analysis(embs, p.Results.nIterations, first_target);

% Store aligned data in cell format. 
for ii = 1:size(realigned,3)
    aligned{ii} = realigned(:,:,ii);
end

end
function [realigned, xfms] = procrustes_analysis(embeddings,nIterations,firstTarget)
% PROCRUSTES_ANALYSIS   Align components using Procrustes method.
%   realigned = PROCRUSTES_ANALYSIS(embeddings,nIterations) aligns the
%   components of embeddings using Procrustes method. Embeddings is a cell
%   array, each cell array contains an m-by-n array where m is the number
%   of vertices and n the number of components of one subject. Components
%   are aligned iteratively using nIterations iterations. The first
%   alignment aligns all subjects to the components in the first cell of
%   embeddings. Subsequent alignments align all components to the mean
%   realigned components of the previous iteration.
%
%   realigned = PROCRUSTES_ANALYSIS(embeddings,nIterations,firstTarget)
%   aligns all components to the provided first target on the first
%   iteration (e.g. components of group average), rather than the first
%   subject.
%
%   [realigned, xfms] = PROCRUSTES_ANALYSIS(embeddings,nIterations,firstTarget)
%   also returns the final transformation matrices xfms. 
%
%   MATLAB adaptation of the Satrajit Ghosh's Python code
%   (https://github.com/satra/mapalign). 
%
%   Written by Reinder Vos de Wael (Jul 2017)

% Check if equal number of components in all subjects.
try 
    cat(3,embeddings{:});    
catch err
    if strcmp(err.identifier,'MATLAB:catenate:dimensionMismatch')
        warning('Target and source embedding have a different number of dimensions. Attempting to solve anyway.')
    else
        rethrow(err)
    end
end

% Set starting parameters. 
if exist('firstTarget','var')
    target = firstTarget;
else
    target = embeddings{1};
end
realigned = target; 

for ii = 1:nIterations
    % Calculate new target and clear workspace. 
    if ii > 1
        target = mean(realigned,3); 
        clearvars realigned xfms 
    end
    
    % Run realignment. 
    for jj = 1:numel(embeddings)
        if ii == 1 && jj == 1 && ~exist('firstTarget','var'); continue; end 
        [U,~,V] = svd(target.' * embeddings{jj},0);
        
        % This corresponds with the Satrajit Python code; Numpy seems to do
        % this implicitly in its svd.
        U = U(:, 1:min(size(U,2), size(V,2)));
        V = V(:, 1:min(size(U,2), size(V,2)));
        
        % Calculate the transformation.
        xfms{jj} = V * U';
        
        % Apply the transformation. 
        realigned(:,:,jj) = embeddings{jj} * xfms{jj};
    end
end
end
