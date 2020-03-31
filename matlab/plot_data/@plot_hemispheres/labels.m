function labels(obj,varargin)
% Sets hemisphere labels for plot_hemispheres. Labeltext is taken from
% obj.labeltext. Text name-value pairs can be provided. If none are
% provided, then the defaults are: 
%   Rotation: 90
%   Units: Normalized
%   HorizontalAlignment: 'Center'
%   FontName: DroidSans
%   FontSize 18
%   otherwise default text() parameters. 
%
% For more information, please consult our <a
% href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/visualization/plot_hemispheres.html">ReadTheDocs</a>.

% Get default properties
properties = struct( ...
    'Rotation',90, ...
    'Units','Normalized', ...
    'HorizontalAlignment','center', ...
    'FontName', 'DroidSans', ...
    'FontSize',18);

for ii = 1:2:numel(varargin)
    properties.(varargin{ii}) = varargin{ii+1}; 
end

% Convert property structure to cell array.
C=[fieldnames(properties).'; struct2cell(properties).'];
C=C(:).';

% Set new labels. 
for ii = 1:numel(obj.labeltext)
    % If a label exists already modify only the properties.
    % Rather complex if-statement because there's many exception cases that
    % can error.
    if ismember('text',fieldnames(obj.handles))
        if numel(obj.handles.text) >= ii 
            if ishandle(obj.handles.text(ii))
                set(obj.handles.text(ii)        , ...
                    'String'    , obj.labeltext{ii}, ...
                    C{:});
                continue
            end
        end
    end
    % If no labels exist yet, then simply build some. 
    obj.handles.text(ii) = text(obj.handles.axes(ii,1), ...
        -.2, .5, obj.labeltext{ii}, C{:});
end
drawnow
end
