function SC = combine_surfaces(S1,S2,format)
% COMBINE_SURFACES   combines two surfaces into a single structure. 
% 
%   SC = COMBINE_SURFACES(S1,S2) takes two surfaces in a format readable
%   by convert_surface and combines them into a single one. 
%   
%   SC = COMBINE_SURFACES(S1,S2,format) outputs surface in the designated
%   format; either 'SurfStat' (default) or 'MATLAB'.
%
%   This script is part of the BrainSpace toolbox. For more information
%   please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/main_functionality/combine_surfaces.html">ReadTheDocs</a>.


if nargin < 3
    format = 'surfstat';
end

% Make sure all surfaces are in the same format.
S1c = convert_surface(S1);
S2c = convert_surface(S2);

SC.coord = [S1c.coord,S2c.coord];
SC.tri = [S1c.tri;S2c.tri + size(S1c.coord,2)];

% Convert to output type
if ~strcmp(format,'surfstat')
    SC = convert_surface(SC,format);
end
