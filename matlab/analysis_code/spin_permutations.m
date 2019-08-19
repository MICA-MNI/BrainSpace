function Y_rand = spin_permutations(Y,spheres,permutation_number,varargin)
% Compute designated # of permutations/spins of the input surface data.
% Uniformly samples from all possible rotations and applies them to the
% data. 
%
% Original code by Aaron Alexander-Bloch & Siyuan Liu 
% Modified for use in the BrainSpace toolbox by Reinder Vos de Wael

% Make sure input is in the correct format. 
p = inputParser;
addParameter(p, 'parcellation', [],   @isnumeric);
addParameter(p, 'type', 'freesurfer', @ischar);
parse(p, varargin{:});
parcellation = p.Results.parcellation;
type = lower(p.Results.type);

if ~iscell(Y)
    Y = {Y}; 
end

if ~iscell(spheres)
    spheres = {spheres};
end

if numel(Y) ~= numel(spheres)
    error('Found a different number of data cells than sphere cells.');
end

if ~isempty(parcellation)
    if ~iscell(parcellation)
        parcellation = {parcellation};
    end
end

if numel(spheres) > 2
    error('Does not support more than two surfaces.')
end

if ~ismember(type,{'freesurfer','civet'})
    error('Type must be ''freesurfer'' or ''civet''.');
end

% Convert the surface to SurfStat so we can read the vertex data and build
% the KD Tree.
for ii = 1:numel(spheres)
    S{ii} = convert_surface(spheres{ii},'SurfStat');
    vertices{ii} = S{ii}.coord';    

    % Check for correct data dimensions. 
    if size(Y{ii},1) ~= size(vertices{ii},1)
        if size(Y{ii},2) == size(vertices{ii},1)
            error(['Found a different number of data points than vertices.' ...
                ' The number of rows in Y should be equal to the number of vertices.' ...
                ' It looks like you may be using row vectors, try transposing your data.'])
        else
            error(['Found a different number of data points than vertices. ' ...
                'The number of rows in Y should be equal to the number of vertices.']);
        end
    end
    
    % If parcellated data on sphere, get centroids of vertices.
    if ~isempty(parcellation)
        % Get Euclidean mean of points within each parcel.
        euclidean_mean = parcellationmean(vertices{ii}, parcellation{ii},'ignorewarning')';
        
        % parcellationmean adds nan-columns if the parcellations are not 1:N. 
        euclidean_mean(any(isnan(euclidean_mean),2),:) = [];
        
        % Find the centroid i.e. closest point to the Euclidean mean. 
        idx = knnsearch(vertices{ii},euclidean_mean);
        vertices{ii} = vertices{ii}(idx,:);   
    end
    % Initalize the KD Tree
    tree{ii} = KDTreeSearcher(vertices{ii});
end

Y_rand=cell(numel(spheres),1);
I1 = diag([-1 1 1]);

%permutation starts
disp('Running Spin Permutation');

for j=1:permutation_number
    %the updated uniform sampling procedure
    A = normrnd(0,1,3,3);
    [rotation, temp] = qr(A);
    rotation = rotation * diag(sign(diag(temp)));
    if(det(rotation)<0)
        rotation(:,1) = -rotation(:,1);
    end
    
    % Rotate the data. 
    for ii = 1:numel(spheres)
        if ii == 2 && ~strcmp(type,'civet') % Civet does not need a flipped rotation.
            rotation = I1 * rotation * I1; % Flip Y-Z plane for right hemisphere.
        end
        rotated_vertices = vertices{ii}*rotation;
        nearest_neighbour = knnsearch(tree{ii}, rotated_vertices); % added 2019-06-18 see home page
        Y_rand{ii}= cat(3,Y_rand{ii}, Y{ii}(nearest_neighbour,:));
    end
end
end