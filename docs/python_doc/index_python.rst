.. _python_package:

Python Package
======================
This page contains links to all related documents on BrainSpace python package.
The :ref:`tutorials<examples_index>` is a good starting point to get familiar
with the main functionality provided by BrainSpace: creation and alignment of
:ref:`gradients<pymod-gradient>`, and :ref:`null models<pymod-nullmodels>`.
With the tutorials you can also learn about the :ref:`data<pymod-datasets>`
that comes with the package, :ref:`plotting<pymod-plotting>` and other
functions. For more information, please refer to the API.

In the python package, surface functionality is built on top of the
`Visualization Toolkit (VTK) <https://vtk.org/>`_. BrainSpace
offers a :ref:`high-level interface<pymod-vtkinterface>` to work with VTK.
In VTK wrapping, we introduce the wrapping scheme used in BrainSpace and some
basic usage examples. Note, however, that this part is not a requirement to
start using BrainSpace. In the :ref:`Mesh<pymod-mesh>` module you can find most
of the functionality you need to work with surfaces. A surface in BrainSpace
is represented as :class:`.BSPolyData` object. Reading/writing of several
formats is supported through the :func:`.read_surface` and
:func:`.write_surface` functions.




.. toctree::
   :maxdepth: 1

   auto_examples/index
   api_doc/index_api
   vtk_wrap
