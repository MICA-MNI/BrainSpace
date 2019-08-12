function S2 = convert_surface(S,type)
% Converts surfaces to MATLAB or SurfStat format.

if nargin < 2
    type = 'SurfStat';
end

% If multiple surfaces provided in a cell, loop over all. 
if iscell(S)
    for ii = 1:numel(S)       
        S2{ii} = convert_surface(S{ii},type);
    end
    return
end

% If input is a char array, load the file. 
if ischar(S)
    if endsWith(S,'.gii')
        S = gifti(S);
    elseif endsWith(S, '.mat')
        S = load(S);
    else
        try
            S = SurfStatReadSurf1(S);
        catch
            error('Could not read surface.');
        end
    end
end

% Convert to a common format. 
f = fieldnames(S);
if ismember('tri',f) && ismember('coord',f) && ismember('faces',f) && ismember('vertices',f)
    error('Could not determine input surface type.');
elseif ismember('faces',f) && ismember('vertices',f)
    faces = S.faces;
    vertices = S.vertices;
elseif ismember('tri',f) && ismember('coord',f)
    faces = S.tri;
    vertices = S.coord';
else
    error('Could not determine input surface type.');
end

% Convert to requested format. 
switch lower(type)
    case 'surfstat'
        S2.tri = faces;
        S2.coord = vertices';
    case 'matlab'
        S2.faces = faces;
        S2.vertices = vertices;
    otherwise
        error('Unknown output type requested. Options are: ''surfstat'' and ''matlab''.');
end
end
