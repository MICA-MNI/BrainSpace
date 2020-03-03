function S2 = convert_surface(S,varargin)
% CONVERT_SURFACE   Converts surfaces to MATLAB or SurfStat format and 
%                   writes new surface files.
%
%   S2 = CONVERT_SURFACE(S) converts surface S to SurfStat format. S can
%   either be a file (.gii, .mat, .obj, Freesurfer), a loaded variable (in
%   SurfStat or MATLAB format), or a cell array containing multiple of the
%   former.
%
%   S2 = CONVERT_SURFACE(S,'format',F) allows for specifying the output
%   format F, either 'SurfStat' or 'MATLAB'.
%
%   S2 = CONVERT_SURFACE(S,'path',P) will write a file to path P. Supported
%   formats are .gii, .obj, .mat and Freesurfer. Only one surface can be
%   provided in S when writing surfaces. 
%
%   For more information, pleasse consult our <a
% href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/support_functions/convert_surface.html">ReadTheDocs</a>.

p = inputParser;
check_input = @(x) iscell(x) || ischar(x);
addParameter(p,'format','SurfStat', check_input);
addParameter(p,'path','', @ischar)
parse(p, varargin{:});

format = p.Results.format;
path = p.Results.path;

% If multiple surfaces provided in a cell, loop over all.
if iscell(S)
    if ~isempty(path) && numel(S) > 1
        error('Multiple inputs are not supported for surface writing');
    end
    for ii = 1:numel(S)
        S2{ii} = convert_surface(S{ii},'format',format,'path',path);
    end
    return
end

% If input is a char array, load the file. 
if ischar(S)
    if endsWith(S,'.gii')
        try
            S = gifti(S);
        catch ME
            if strcmp(ME.identifier,'MATLAB:UndefinedFunction')
                error('Could not find GIFTI library. Is it installed? If yes, is it on your path?');
            else
                rethrow(ME);
            end
        end
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

% Convert to requested variable output format. 
switch lower(format)
    case 'surfstat'
        S2.tri = faces;
        S2.coord = vertices';
    case 'matlab'
        S2.faces = faces;
        S2.vertices = vertices;
    otherwise
        error('Unknown output type requested. Options are: ''surfstat'' and ''matlab''.');
end


% Write surface
if ~isempty(path)
    if endsWith(path,'.gii')
        % Write gifti
        try
            gii = gifti();
        catch ME
            if strcmp(ME.identifier,'MATLAB:UndefinedFunction')
                error('Could not find GIFTI library. Is it installed? If yes, is it on your path?');
            else
                rethrow(ME);
            end
        end
        gii.vertices = vertices;
        gii.faces = faces;
        save(gii,path);
    elseif endsWith(path,'.mat')
        % Write matlab file
        switch lower(format)
            case 'surfstat'
                tri = faces; 
                coord = vertices';
                save(path,'tri','coord');
            case 'matlab'
                save(path,'faces','vertices');
        end
    else
        % Assume .obj or Freesurfer
        s_tmp.tri = faces; 
        s_tmp.coord = vertices';
        SurfStatWriteSurf1(path,s_tmp);    
    end
end
