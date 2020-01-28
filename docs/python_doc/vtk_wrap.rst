
.. _pypage-vtk_wrapping:

VTK wrapping
===========================

Surface mesh functionality provided in BrainSpace is built on top of the
`Visualization Toolkit (VTK) <https://vtk.org/>`_. This document introduces the
wrapping interface used by BrainSpace to work with VTK objects.


- :ref:`pysec-vtk_wrapping-interface`
- :ref:`pysec-vtk_wrapping-pipeline`
- :ref:`pysec-vtk_wrapping-data_object`
- :ref:`pysec-vtk_wrapping-plotting`



.. _pysec-vtk_wrapping-interface:

Wrapping interface
------------------

:class:`.BSVTKObjectWrapper` is the base class for all wrappers implemented in
BrainSpace. Wrapping a VTK object is done with the :func:`.wrap_vtk` function.
When wrapping a VTK object, if the corresponding wrapper does not exist, it
falls back to :class:`.BSVTKObjectWrapper`. Check the API for the complete list
of wrappers implemented in the current version of BrainSpace.

The :class:`.BSVTKObjectWrapper` is a wrapper that extends the Python object
wrapper in VTK
:class:`~vtkmodules.numpy_interface.dataset_adapter.VTKObjectWrapper`
to provide easier access to VTK setter and getter class methods. The wrapper,
since it is a subclass of
:class:`~vtkmodules.numpy_interface.dataset_adapter.VTKObjectWrapper`,
holds a reference to the VTK object in the
:attr:`.BSVTKObjectWrapper.VTKObject` attribute. And further includes the
following functionality:

#. :meth:`~.BSVTKObjectWrapper.setVTK` and :meth:`~.BSVTKObjectWrapper.getVTK` to invoke several setter/getter methods on the VTK object: ::

    >>> import vtk

    >>> # Let's create a sphere with VTK
    >>> s = vtk.vtkSphere()
    >>> s
    (vtkCommonDataModelPython.vtkSphere)0x7f610d222f48

    >>> # And check the default values
    >>> s.GetRadius()
    0.5
    >>> s.GetCenter()
    (0.0, 0.0, 0.0)

    >>> # We are going to wrap the sphere
    >>> from brainspace.vtk_interface import wrap_vtk
    >>> ws = wrap_vtk(s)

    >>> # ws is an instance of BSVTKObjectWrapper
    >>> ws
    <brainspace.vtk_interface.base.BSVTKObjectWrapper at 0x7f60cd7d6f60>

    >>> # and holds a reference to the VTK sphere
    >>> ws.VTKObject
    (vtkCommonDataModelPython.vtkSphere)0x7f610d222f48

    >>> # Now we can invoke getter methods as follows:
    >>> ws.getVTK('radius', center=None)
    {'radius': 0.5, 'center': (0.0, 0.0, 0.0)}

    >>> # and set different values
    >>> ws.setVTK(radius=2, center=(1.5, 1.5, 1.5))
    <brainspace.vtk_interface.base.BSVTKObjectWrapper at 0x7f60cd7d6f60>

    >>> # To check that everything works as expected
    >>> s.GetRadius()
    2.0
    >>> s.GetCenter()
    (1.5, 1.5, 1.5)

    >>> # these methods can be invoked on the wrapper, which forwards
    >>> # them to the VTK object
    >>> ws.GetRadius()
    2.0


#. Calling VTK setters and getters can also be treated as attributes,
   by overloading :meth:`~.BSVTKObjectWrapper.__setattr__` and :meth:`~.BSVTKObjectWrapper.__getattr__`: ::

    >>> # we can access to the radius and center
    >>> ws.radius
    2.0
    >>> ws.center
    (1.5, 1.5, 1.5)

    >>> # and even set new values
    >>> ws.radius = 8
    >>> ws.center = (10, 10, 10)

    >>> # check that everything's ok
    >>> s.GetRadius()
    8.0
    >>> s.GetCenter()
    (10.0, 10.0, 10.0)

This functionality is available for all methods that start with Get/Set. To see
all the methods available, :class:`~.BSVTKObjectWrapper` holds a
dictionary :attr:`~.BSVTKObjectWrapper.vtk_map` for each wrapped class: ::

    >>> ws.vtk_map.keys()
    dict_keys(['set', 'get'])

    >>> # for getter methods
    >>> ws.vtk_map['get']
    {'addressasstring': 'GetAddressAsString',
     'center': 'GetCenter',
     'classname': 'GetClassName',
     'command': 'GetCommand',
     'debug': 'GetDebug',
     'globalwarningdisplay': 'GetGlobalWarningDisplay',
     'mtime': 'GetMTime',
     'radius': 'GetRadius',
     'referencecount': 'GetReferenceCount',
     'transform': 'GetTransform'}

Note that this approach is case-insensitive. But we recommend using camel case,
at least, for methods with more that one word: ::

    >>> # To access the reference count, for example, these are the same
    >>> ws.referencecount
    1
    >>> ws.ReferenceCount
    1

:func:`.wrap_vtk` provides a nice way to wrap and simultaneously set the values for a VTK object: ::

    >>> ws2 = wrap_vtk(vtk.vtkSphere(), radius=10, center=(5, 5, 0))
    >>> ws2.getVTK('radius', 'center')
    {'radius': 10.0, 'center': (5.0, 5.0, 0.0)}

    >>> ws2.VTKObject.GetRadius()
    10.0
    >>> ws2.VTKObject.GetCenter()
    (5.0, 5.0, 0.0)


In VTK, among setter methods, we have state methods with the form **Set**\ Something\ **To**\ Value.
Using the previous functionality, these methods can be called as follows: ::

    >>> # Let's create a mapper
    >>> m = vtk.vtkPolyDataMapper()

    >>> # This class has several state methods to set the color mode
    >>> [m for m in dir(m) if m.startswith('SetColorModeTo')]
    ['SetColorModeToDefault',
     'SetColorModeToDirectScalars',
     'SetColorModeToMapScalars']

    >>> # The default value is
    >>> m.GetColorModeAsString()
    'Default'

    >>> # Now we are going to wrap the VTK object
    >>> wm = wrap_vtk(m)
    >>> wm
    <brainspace.vtk_interface.wrappers.BSPolyDataMapper at 0x7f60ada07828>

    >>> # and change the default value as we know so far
    >>> # Note that we use None because the method accepts no arguments
    >>> wm.colorModeToMapScalars = None  # same as wm.SetColorMoteToMapScalars()
    >>> wm.GetColorModeAsString()
    'MapScalars'

    >>> # state methods can also be used as follows
    >>> wm.colorMode = 'DirectScalars'  # which is the default
    >>> wm.GetColorModeAsString()
    'Default'

    >>> # This can be used when wrapping
    >>> m2 = wrap_vtk(vtkPolyDataMapper(), colorMode='mapScalars', scalarVisibility=False)
    >>> m2.getVTK('colorModeAsString', 'scalarVisibility')
    {'colorModeAsString': 'MapScalars', 'scalarVisibility': 0}


In the example above, the wrapper class of our mapper is no longer :class:`~.BSVTKObjectWrapper`
but :class:`~.BSPolyDataMapper`. This is because :func:`.wrap_vtk` looks for a
convenient wrapper by searching the hierarchy of wrappers in a bottom-up fashion,
and :class:`~.BSPolyDataMapper` is a wrapper that is already implemented in the
current version of BrainSpace. We can also see this with VTK actor: ::

    >>> wa = wrap_vtk(vtk.vtkActor())
    >>> wa
    <brainspace.vtk_interface.wrappers.BSActor at 0x7f60cd749e80>

    >>> # When a wrapper exists, the VTK object can be created directly
    >>> from brainspace.vtk_interface.wrappers import BSActor
    >>> wa2 = BSActor()
    <brainspace.vtk_interface.wrappers.BSActor at 0x7f60cce8fac8>

    >>> # and can be created with arguments
    >>> # for example, setting the previous mapper
    >>> wa3 = BSActor(mapper=wm)
    >>> wa3.mapper.VTKObject is wm.VTKObject
    True
    >>> wa3.mapper.VTKObject is m
    True

:class:`.BSActor` is a special wrapper, because calls to setter and getter methods can
be forwarded also to its property (i.e., GetProperty()). Methods are first
forwarded to the VTK object of the actor, but if they do not exist, they are
forwarded then to the property. As of the current version, this is only implemented
for :class:`.BSActor`: ::

    >>> # To see the opacity using the VTK object
    >>> wa3.VTKObject.GetProperty().GetOpacity()
    1.0

    >>> # this wa3.VTKObject.GetOpacity() raises an exception

    >>> # Now, using the wrapper
    >>> wa3.GetOpacity()
    1.0
    >>> # or
    >>> wa3.opacity
    1.0

    >>> # and we can set the opacity
    >>> wa3.opacity = 0.25
    >>> wa3.VTKObject.GetProperty().GetOpacity()
    0.25

The advantage of this approach over existing packages that build over VTK is that
we do not need to learn about all the new API. If the user is familiar with VTK,
then using this approach is straightforward, we can invoke the setter and getter
methods by simply stripping the Get/Set prefixes.



.. _pysec-vtk_wrapping-pipeline:

Pipeline liaisons
------------------

VTK workflow is based on connecting (a source to) several filters (and to a sink).
This often makes the code very cumbersome. Let's see a dummy example: ::

    >>> # Generate point cloud
    >>> point_source = vtk.vtkPointSource()
    >>> point_source.SetNumberOfPoints(25)

    >>> # Build convex hull from point cloud
    >>> delauny = vtk.vtkDelaunay2D()
    >>> delauny.SetInputConnection(point_source.GetOutputPort())
    >>> delauny.SetTolerance(0.01)

    >>> # Smooth convex hull
    >>> smooth_filter = vtk.vtkWindowedSincPolyDataFilter()
    >>> smooth_filter.SetInputConnection(delauny.GetOutputPort())
    >>> smooth_filter.SetNumberOfIterations(20)
    >>> smooth_filter.FeatureEdgeSmoothingOn()
    >>> smooth_filter.NonManifoldSmoothingOn()

    >>> # Compute normals
    >>> normals_filter = vtk.vtkPolyDataNormals()
    >>> normals_filter.SetInputConnection(smooth_filter.GetOutputPort())
    >>> normals_filter.SplittingOff()
    >>> normals_filter.ConsistencyOn()
    >>> normals_filter.AutoOrientNormalsOn()
    >>> normals_filter.ComputePointNormalsOn()

    >>> # Execute pipeline
    >>> normals_filter.Update()

    >>> # Get the output
    >>> output1 = normals_filter.GetOutput()
    >>> output1
    (vtkCommonDataModelPython.vtkPolyData)0x7f60cceabb28

    >>> output1.GetNumberOfPoints()
    25

For these scenarios, Brainspace provides the :func:`.serial_connect` function
to serially connect several filters and skip the boilerplate of connecting
filters. The previous example can be rewritten as follows: ::

    >>> from brainspace.vtk_interface.pipeline import serial_connect

    >>> # Generate point cloud
    >>> point_source = wrap_vtk(vtk.vtkPointSource, numberOfPoints=25)

    >>> # Build convex hull from point cloud
    >>> delauny = wrap_vtk(vtk.vtkDelaunay2D, tolerance=0.01)

    >>> # Smooth convex hull
    >>> smooth_filter = wrap_vtk(vtk.vtkWindowedSincPolyDataFilter,
    ...                          numberOfIterations=20, featureEdgeSmoothing=True,
    ...                          nonManifoldSmoothing=True)

    >>> # Compute normals
    >>> normals_filter = wrap_vtk(vtk.vtkPolyDataNormals, splitting=False,
    ...                           consistency=True, autoOrientNormals=True,
    ...                           computePointNormals=True)

    >>> # Execute and get the output
    >>> output2 = serial_connect(point_source, delauny, smooth_filter,
    ...                          normals_filter, as_data=True)
    >>> output2
    <brainspace.vtk_interface.wrappers.BSPolyData at 0x7f60a3f5fa20>

    >>> output2.GetNumberOfPoints()
    25

First, note that we can simply provide the VTK class instead of the object
to :func:`.wrap_vtk`. Furthermore, the output object of the previous pipeline
is a polydata. This brings us to one of the most important wrappers in
BrainSpace, :class:`.BSPolyData`, a wrapper for VTK polydata objects.


.. _pysec-vtk_wrapping-data_object:

Data object wrappers
---------------------

BrainSpace is intended primarily to work with triangular surface meshes of the brain.
Hence, the importance of :class:`.BSPolyData`. VTK already provides very good
wrappers for VTK data objects in the :mod:`~vtkmodules.numpy_interface.dataset_adapter`
module. In BrainSpace, these wrappers are simply extended to incorporate the
aforementioned functionality of the :class:`.BSVTKObjectWrapper` and some
additional features: ::

    >>> # Using the output from previous example
    >>> output2.numberOfPoints
    25
    >>> output2.n_points
    25

    >>> # Check if polydata has only triangles
    >>> output2.has_only_triangle
    True

    >>> # Since a polydata can hold vertices, lines, triangles, and their
    >>> # poly versions, polygons are returned as a 1D array
    >>> output2.Polygons.shape
    (144,)

    >>> # we further provide an additional method to recover the polygons:
    >>> output2.GetCells2D().shape
    (36, 3)

    >>> # this method raises an exception if the polydata holds different
    >>> # cell or polygon types, this can be checked with
    >>> output2.has_unique_cell_type
    True

    >>> # to get the cell types
    >>> output2.cell_types
    array([5])

    >>> vtk.VTK_TRIANGLE == 5
    True

    >>> # To get the point data
    >>> output2.PointData.keys()
    ['Normals']

    >>> output2.point_keys
    ['Normals']

    >>> output2.PointData['Normals'].shape
    (25, 3)

    >>> # or
    >>> output2.get_array(name='Normals', at='point').shape
    (25, 3)

    >>> # we do not have to specify the attributes
    >>> output2.get_array(name='Normals')
    (25, 3)

    >>> # raises exception if name is in more than one attribute (e.g., point
    >>> # and cell data)
    >>> dummy_cell_normals = np.zeros((output2.n_cells, 3))
    >>> output2.append_array(dummy_cell_normals, name='Normals', at='cell')

    >>> output2.get_array(name='Normals') # Raise exception!


Most properties and methods of :class:`.BSPolyData` are inherited
from :class:`.BSDataSet`. Check out their documentations for more information.



.. _pysec-vtk_wrapping-plotting:

Plotting
------------------

BrainSpace offers two high-level plotting functions: :func:`.plot_surf` and
:func:`.plot_hemispheres`. These functions are based on the wrappers of the corresponding
VTK objects. We have already seen above the :class:`~.BSPolyDataMapper` and
:class:`~.BSActor` class wrappers. Here we will show how rendering is performed
using these wrappers. The base class for all plotters is :class:`.BasePlotter`, with
subclasses :class:`.Plotter` and :class:`.GridPlotter`.
These classes forward unknown set/get requests to :class:`.BSRenderWindow`, which
is a wrapper of :class:`~vtk.vtkRenderWindow`. Let's see a simple example: ::

    >>> ipth = 'path_to_surface'

    >>> from brainspace.mesh import mesh_io as mio
    >>> from brainspace.plotting.base import Plotter
    >>> # from brainspace.vtk_interface.wrappers import BSLookupTable

    >>> surf = mio.read_surface(ipth, return_data=True)

    >>> yeo7_colors = np.array([[0, 0, 0, 255],
    ...                         [0, 118, 14, 255],
    ...                         [230, 148, 34, 255],
    ...                         [205, 62, 78, 255],
    ...                         [120, 18, 134, 255],
    ...                         [220, 248, 164, 255],
    ...                         [70, 130, 180, 255],
    ...                         [196, 58, 250, 255]], dtype=np.uint8)

    >>> # create a plotter with 2 rows and 2 columns
    >>> # other arguments are forwarded to BSRenderWindow (i.e., vtkRenderWindow)
    >>> p = Plotter(n_rows=2, n_cols=2, try_qt=False, size=(400, 400))

    >>> # Add first renderer, span all columns of first row
    >>> ren1 = p.AddRenderer(row=0, col=None, background=(1,1,1))

    >>> # Add actor (actor created if not provided)
    >>> ac1 = ren1.AddActor()

    >>> # Set mapper (mapper created if not provided)
    >>> m1 = ac1.SetMapper(inputDataObject=surf, colorMode='mapScalars',
    ...                    scalarMode='usePointFieldData',
    ...                    interpolateScalarsBeforeMapping=False,
    ...                    arrayName='yeo7', useLookupTableScalarRange=True)

    >>> # Set mapper lookup table (created if not provided)
    >>> lut1 = m1.SetLookupTable(numberOfTableValues=8, Range=(0, 7),
    ...                          table=yeo7_colors)

    >>> # Add second renderer in first column of second row
    >>> ren2 = p.AddRenderer(row=1, col=0, background=(1,1,1))
    >>> ac2 = ren2.AddActor(opacity=1, edgeVisibility=0)
    >>> m2 = ac2.SetMapper(inputData=surf)

    >>> # plot in notebook
    >>> p.show(interactive=False, embed_nb=True)

See that we can use :meth:`~.Plotter.AddRenderer` to create the renderer if it is
not provided as an argument. The same happens with methods :meth:`~.BSRenderer.AddActor`,
:meth:`~.BSActor.SetMapper` and :meth:`~.BSPolyDataMapper.SetLookupTable`, from
:class:`.BSRenderer`, :class:`.BSActor` and :class:`.BSPolyDataMapper` wrapper
classes, respectively.

The only difference between :class:`.Plotter` and  :class:`.GridPlotter` is that
in the latter a renderer is restricted to a single entry, cannot span more than one
entry.