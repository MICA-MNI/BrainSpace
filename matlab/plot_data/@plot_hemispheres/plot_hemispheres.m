classdef plot_hemispheres < handle
    % PLOT_HEMISPHERES   Plots data on the cortical surface.
    %
    %   obj = PLOT_HEMISPHERES(data,surface,varargin) plots column vectors of the
    %   n-by-m data matrix on surfaces provided in one or two element cell
    %   arary surface. Vertices included in the surfaces must sum to n or the
    %   length of the parcellation vector (see below). All handles to the
    %   graphics objects are in the structure h.
    %
    %   Valid name-value pairs:
    %       - 'labeltext': A length m cell array containing text labels to add to
    %       the surfaces of each column vector.
    %
    %       - 'parcellation': A v-by-1 column vector containing a parcellation
    %       scheme, with identical numbers denoting same parcel.
    %
    %       - 'views': A character vector containing the views you wish to
    %       see. Valid options are  l(ateral), m(edial), i(nferior),
    %       s(uperior), a(nterior), and p(osterior). When supplying a
    %       single surface, the lateral/medial designations must be
    %       inverted for the right hemisphere. Default: 'lm'.
    %
    %   For legacy usage: the handles formerly stored in obj are now stored
    %   in obj.handles.
    %
    %   For more information, please consult our <a
    %   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/visualization/plot_hemispheres.html">ReadTheDocs</a>.
    
    properties
        handles
        labeltext
    end
    
    properties(SetAccess = private)
        data
        surface
        parcellation
        views
    end
    
    properties(Dependent)
        plotted_data
    end
    
    methods
        function obj = plot_hemispheres(varargin)
            % Initialize the object.
            p = inputParser;
            addRequired(p,'data')
            addRequired(p,'surface')
            addParameter(p,'labeltext',[]);
            addParameter(p,'parcellation',[],@isnumeric);
            addParameter(p,'views','lm', @ischar);
            parse(p,varargin{:})
            
            % Set properties.
            obj.handles = struct('figure',gobjects(0),'axes',gobjects(0),'trisurf',gobjects(0), ...
                'camlight',gobjects(0),'colorbar',gobjects(0),'text',gobjects(0));
            obj.surface = p.Results.surface;
            obj.data = p.Results.data;
            obj.parcellation = p.Results.parcellation;
            obj.labeltext = p.Results.labeltext;
            obj.views = p.Results.views;
            
            % Check whether input is correct.
            if ~isempty(obj.parcellation)
                obj.check_data_parcellation();
            end
            obj.check_surface();
            
            % Plot figure.
            obj.plotter()
        end
        
        %% Set methods
        % Set surface.
        function set.surface(obj,surface)
            surface = convert_surface(surface,'Format','SurfStat');
            if ~iscell(surface)
                surface = {surface};
            end
            if numel(obj.surface) > 3
                error('More than two surfaces are not accepted.');
            end
            obj.surface = surface;
        end
        
        % Set parcellation.
        function set.parcellation(obj,parcellation)
            if ~isempty(parcellation)
                if parcellation ~= round(parcellation)
                    error('parcellation may only consist of zeros and positve integers.')
                end
                if ~isvector(parcellation) 
                    error('parcellation must be a vector.')
                end
                parcellation = double(parcellation(:));
            end
            obj.parcellation = parcellation;
        end
        
        % Set labeltext.
        function set.labeltext(obj,labeltext)
            if ~isempty(labeltext)
                if ischar(labeltext)
                    labeltext = {labeltext};
                end
                if ~iscell(labeltext) && ~isstring(labeltext)
                    error('labeltext must be a char, cell, or string array');
                end
                if numel(labeltext) ~= size(obj.data,2)
                    error('Must have as many labels as columns in the data matrix.')
                end
                obj.labeltext = labeltext;
                if ismember('figure',fieldnames(obj.handles))
                    if ishandle(obj.handles.figure)
                        obj.labels();
                    end
                end
            end
        end
        
        function set.views(obj,views)
            views = lower(views);
            if numel(unique(views)) ~= numel(views)
                warning('Detected duplicate characters in views; removing duplicates.');
                views = unique(views);
            end
            for ii = 1:numel(views)
                if ~ismember(views(ii),{'l','m','s','i','a','p'})
                    error('Valid characters or views are: l(ateral), m(edial), i(nferior), s(uperior), a(nterior), p(osterior)');
                end
            end
            obj.views = views;
        end
        
        % Get to-plot data
        function plotted_data = get.plotted_data(obj)
            if isempty(obj.parcellation)
                plotted_data = obj.data;
            else
                plotted_data = parcel2full(obj.data,obj.parcellation);
                plotted_data(isnan(plotted_data)) = -Inf;
            end
        end
        
    end
    
    % Private methods for input checks. 
    methods(Access=private)
        function check_data_parcellation(obj)
            % Check for correct size of parcellation.
            sz = size(obj.data,1);
            if max(obj.parcellation) ~= sz
                error('The highest number in the parcellation scheme must be equivalent to length of the first dimension of the parcellated data.');
            end
            
            % Check if all numbers are included.
            if ~all(ismember(1:sz,obj.parcellation))
                warning('Some parcel numbers are missing, some data may not be included in the output.')
            end
        end
        
        function check_surface(obj)
            N1 = sum(cellfun(@(x)size(x.coord,2),obj.surface));
            N2 = size(obj.plotted_data,1);
            
            if N1 ~= N2
                error('Number of vertices on the surface and number of data points do not match.');
            end
            if size(obj.data,2) > 5
                warning('Plotting more than five data vectors may result in off-screen axes');
            end
        end
    end
end

