function S = combine_surfaces(S1,S2,type)

if nargin < 3
    type = 'surfstat';
end

% Make sure all surfaces are in the same format.
S1c = convert_surface(S1,'surfstat');
S2c = convert_surface(S2,'surfstat');

S.coord = [S1c.coord,S2c.coord];
S.tri = [S1c.tri;S2c.tri + size(S1c.coord,2)];

% Convert to output type
if ~strcmp(type,'surfstat')
    S = convert_surface(S,type);
end
