Python Package Introduction
===========================
This document gives a basic walkthrough of BrainSpace python package.

- :ref:`Working with gradients<Working with gradients>`
- :ref:`VTK wrapping<VTK wrapping>`
- :ref:`Plotting<Plotting>`

Working with gradients
----------------------

:class:`.GradientMaps` is the main class that offers all the functionality to
work with gradients. This class builds the affinity matrix, performs the
embedding and aligns the gradients. This class follows closely the `API of scikit-learn
objects <https://scikit-learn.org/dev/developers/contributing.html#apis-of-scikit-learn-objects>`_.

#. Let's first generate two random symmetric matrices using scikit-learn :func:`~sklearn.datasets.make_spd_matrix`::

    >>> from sklearn.datasets import make_spd_matrix

    >>> x1 = make_spd_matrix(100)
    >>> x2 = make_spd_matrix(100)
    >>> x1.shape
    (100, 100)


#. Next, we build a :class:`.GradientMaps` object::

    >>> from brainspace.gradient import GradientMaps

    >>> # We build the affinity matrix using 'normalized_angle',
    >>> # use Laplacian eigenmaps (i.e., 'le') to find the gradients
    >>> # and align gradients using procrustes
    >>> gm = GradientMaps(n_gradients=2, approach='le', kernel='normalized_angle',
    ...                   align='procrustes')


#. Now we can compute the gradients for the two datasets by invoking the :meth:`~.GradientMaps.fit` method::

    >>> # Note that multiple datasets are passed as a list
    >>> gm.fit([x1, x2])
    GradientMaps(align='procrustes', approach='le', kernel='normalized_angle',
       n_gradients=2, random_state=0)

#. The object has 3 important attributes: eigenvalues, gradients, and aligned gradients::

    >>> # The eigenvalues for x1
    >>> gm.lambdas_[0]
    array([0.76390278, 0.99411812])

    >>> # and x2
    >>> gm.lambdas_[1]
    array([0.77444778, 0.99058541])

    >>> # The gradients for x1
    >>> gm.gradients_[0].shape
    (100, 2)

    >>> # and the gradients after alignment
    >>> gm.aligned_[0].shape
    (100, 2)

#. To illustrate the effect of alignment, we can check the distance between the gradients::

    >>> import numpy as np

    >>> # Disparity between the original gradients
    >>> np.sum(np.square(gm.gradients_[0] - gm.gradients_[1]))
    0.07000481706312509

    >>> # disparity is decreased after alignment
    >>> np.sum(np.square(gm.aligned_[0] - gm.aligned_[1]))
    1.4615624326798128e-05

#. We can also change the embedding approach using 'dm' or an object of :class:`.DifusionMaps`::

    >>> # In this case we will pass an object
    >>> from brainspace.gradient import DiffusionMaps
    >>> dm = DiffusionMaps(alpha=1, diffusion_time=0)

    >>> # let's create a new gm object with the new embedding approach
    >>> gm2 = GradientMaps(n_gradients=2, approach=dm, kernel='normalized_angle',
    ...                    align='procrustes')

    >>> # and fit to the data
    >>> gm2.fit([x1, x2])
    GradientMaps(align='procrustes',
                 approach=DiffusionMaps(alpha=1, diffusion_time=0,
                                        n_components=2, random_state=None),
                 kernel='normalized_angle', n_gradients=2, random_state=None)

    >>> # the disparity between the gradients
    >>> np.sum(np.square(gm2.gradients_[0] - gm2.gradients_[1]))
    21.815792454516334

    >>> # and after alignment
    >>> np.sum(np.square(gm2.aligned_[0] - gm2.aligned_[1]))
    3.326408646218633e-05


#. If we try a different alignment method::

    >>> gm3 = GradientMaps(n_gradients=2, approach='le', kernel='normalized_angle',
    ...                    align='manifold')
    >>> gm3.fit([x1, x2])
    GradientMaps(align='manifold', approach='le', kernel='normalized_angle',
                 n_gradients=2, random_state=None)

    >>> # the disparity between the gradients
    >>> np.sum(np.square(gm3.gradients_[0] - gm3.gradients_[1]))
    0.019346449795655286

    >>> # with 'manifold', the embedding and alignment are performed simultaneously
    >>> np.sum(np.square(gm3.aligned_[0] - gm3.aligned_[1]))
    0.019346449795655286




VTK wrapping
-------------

All surface mesh functionality provided in BrainSpace is built on top of the
`Visualization Toolkit (VTK) <https://vtk.org/>`_. BrainSpace provides several
wrappers for most data objects and some filters in VTK. Here we present
a subset of this functionality.


Wrapping interface
^^^^^^^^^^^^^^^^^^

:class:`.BSVTKObjectWrapper` is the base class for all wrappers implemented in
BrainSpace. Wrapping a VTK object is done with the :func:`.wrap_vtk` function.
When wrapping a VTK object, if the corresponding wrapper does not exist, it
falls back to :class:`.BSVTKObjectWrapper`. The complete list of wrappers
implemented in the current version of BrainSpace is available in ??

The :class:`.BSVTKObjectWrapper` is a wrapper that extends the Python object
wrapper in VTK :class:`~vtkmodules.numpy_interface.dataset_adapter.VTKObjectWrapper`
to provide easier access to VTK setter and getter class methods. The wrapper,
since it is a subclass of :class:`~vtkmodules.numpy_interface.dataset_adapter.VTKObjectWrapper`,
holds a reference to the vtk object in the :attr:`.BSVTKObjectWrapper.VTKObject` attribute.
And further includes the following functionality:

#. :meth:`~.BSVTKObjectWrapper.setVTK` and :meth:`~.BSVTKObjectWrapper.getVTK` to invoke several setter/getter methods on the vtk object: ::

    >>> # Lets create a sphere with VTK
    >>> from vtkmodules.vtkCommonDataModelPython import vtkSphere
    >>> s = vtkSphere()
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

    >>> # and holds a reference to the vtk sphere
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
    >>> # them to the vtk object
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

    >>> ws2 = wrap_vtk(vtkSphere(), radius=10, center=(5, 5, 0))
    >>> ws2.getVTK('radius', 'center')
    {'radius': 10.0, 'center': (5.0, 5.0, 0.0)}

    >>> ws2.VTKObject.GetRadius()
    10.0
    >>> ws2.VTKObject.GetCenter()
    (5.0, 5.0, 0.0)


In VTK, among setter methods, we have state methods with the form **Set**\ Something\ **To**\ Value.
Using the previous functionality, these methods can be called as follows: ::

    >>> # Let's create a mapper
    >>> from vtkmodules.vtkRenderingCorePython import vtkPolyDataMapper
    >>> m = vtkPolyDataMapper()

    >>> # This class has several state methods to set the color mode
    >>> [m for m in dir(m) if m.startswith('SetColorModeTo')]
    ['SetColorModeToDefault',
     'SetColorModeToDirectScalars',
     'SetColorModeToMapScalars']

    >>> # The default value is
    >>> m.GetColorModeAsString()
    'Default'

    >>> # Now we are going to wrap the vtk object
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
current version of BrainSpace. We can also see this with vtk actor: ::

    >>> from vtkmodules.vtkRenderingCorePython import vtkActor
    >>> wa = wrap_vtk(vtkActor())
    >>> wa
    <brainspace.vtk_interface.wrappers.BSActor at 0x7f60cd749e80>

    >>> # When a wrapper exists, the vtk object can be created directly
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
forwarded to the vtk object of the actor, but if they don't exist, they are
forwarded then to the property. As of the current version, this is only implemented
for :class:`.BSActor`: ::

    >>> # To see the opacity using the vtk object
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
we don't need to learn about all the new API. If the user is familiar with VTK,
then using this approach is straightforward, we can invoke the setter and getter
methods by simply stripping the Get/Set prefixes.

.. * .. currentmodule:: brainspace.vtk_interface.wrappers

.. * .. autosummary::
.. *    :toctree: ../../generated/


.. *    BSVTKObjectWrapper
.. *    wrap_vtk
.. *    is_vtk
.. *    is_wrapper



Pipeline liaisons
^^^^^^^^^^^^^^^^^
VTK workflow is based on connecting (a source to) several filters (and to a sink).
This often makes the code very cumbersome. Let's see a dummy example: ::

    >>> import vtk

    >>> # Generate point cloud
    >>> point_source = vtk.vtkPointSource()
    >>> point_source.SetNumberOfPoints(25)

    >>> # Build convex hull from point cloud
    >>> delauny = vtk.vtkDelaunay2D()
    >>> delauny.SetInputConnection(point_source.GetOutputPort())
    >>> delauny.SetTolerance(0.01)

    >>> # Smooth convex hull
    >>> smooth_filter = vtk.vtkWindowedSincPolyDataFilter()
    >>> smooth_filter.SetInputConnection(delny.GetOutputPort())
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

First, note that we can simply provide the vtk class instead of the object
to :func:`.wrap_vtk`. Furthermore, the output object of the previous pipeline
is a polydata. This brings us to one of the most important wrappers in
BrainSpace, :class:`.BSPolyData`, a wrapper for vtk polydata objects.

.. * .. currentmodule:: brainspace.vtk_interface.pipeline

.. * .. autosummary::
.. *    :toctree: ../../generated/


.. *    serial_connect
.. *    get_output
.. *    to_data


Data object wrappers
^^^^^^^^^^^^^^^^^^^^

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
    >>> output2.get_cells2D().shape
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

    >>> # we don't have to specify the attributes
    >>> output2.get_array(name='Normals')
    (25, 3)

    >>> # raises exception if name is in more than one attribute (e.g., point
    >>> # and cell data)
    >>> dummy_cell_normals = np.zeros((output2.n_cells, 3))
    >>> output2.append_array(dummy_cell_normals, name='Normals', at='cell')

    >>> output2.get_array(name='Normals') # Raise exception!


Most properties and methods of :class:`.BSPolyData` are inherited
from :class:`.BSDataSet`. Check out their documentations for more information.

.. * .. currentmodule:: brainspace.vtk_interface.wrappers

.. * .. autosummary::
.. *    :toctree: ../../generated/


.. *    BSDataSet
.. *    BSAlgorithm
.. *    BSPolyData



Plotting
--------

BrainSpace offers two high-level plotting functions: :func:`.plot_surf` and
:func:`.plot_hemispheres`. These functions are based on the wrappers of the corresponding
vtk objects. We have already seen above the :class:`~.BSPolyDataMapper` and
:class:`~.BSActor` class wrappers. Here we will show how rendering is performed
using these wrappers. The base class for all plotters is :class:`BasePlotter`, with
subclasses :class:`Plotter` and :class:`GridPlotter`.
These classes forward unknown set/get requests to :class:`BSRenderWindow`, which
is a wrapper of :class:`~vtk.vtkRenderWindow`. Let's see a simple example: ::

    >>> ipth = '/media/hd105/___NEW/data_ABIDE/conte_10k_fixed.vtp'

    >>> from brainspace.mesh import mesh_io as mio
    >>> from brainspace.plotting.base import Plotter
    >>> # from brainspace.vtk_interface.wrappers import BSLookupTable

    >>> surf = mio.load_surface(ipth, return_data=True)

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

The only difference between :class:`Plotter` and  :class:`GridPlotter` is that
in the latter a renderer is restricted to a single entry, cannot span more than one
entry.