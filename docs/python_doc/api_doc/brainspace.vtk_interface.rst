VTK interface
=================================

All surface mesh functionality provided in BrainSpace is built on top of the
`Visualization Toolkit (VTK) <https://vtk.org/>`_. BrainSpace provides several
wrappers for most data objects and some filters in VTK. Here we present
a subset of this functionality.

- :ref:`Basic wrapping<Basic wrapping>`
- :ref:`VTK wrappers<VTK wrappers>`
- :ref:`VTK data object wrappers<VTK data object wrappers>`
- :ref:`Pipeline functionality<Pipeline functionality>`
- :ref:`Decorators<Decorators>`



Basic wrapping
-------------------------

.. currentmodule:: brainspace.vtk_interface.wrappers

.. autosummary::
   :toctree: ../../generated/

   wrap_vtk
   is_vtk
   is_wrapper


VTK wrappers
-------------------------

.. currentmodule:: brainspace.vtk_interface.wrappers

.. autosummary::
   :toctree: ../../generated/

   BSVTKObjectWrapper
   BSAlgorithm
   BSRenderer
   BSActor
   BSMapper
   BSPolyDataMapper
   BSLookupTable


VTK data object wrappers
-------------------------

.. currentmodule:: brainspace.vtk_interface.wrappers

.. autosummary::
   :toctree: ../../generated/

    BSDataObject
    BSCompositeDataSet
    BSDataSet
    BSPointSet
    BSPolyData
    BSUnstructuredGrid


Pipeline functionality
-------------------------

.. currentmodule:: brainspace.vtk_interface.pipeline

.. autosummary::
   :toctree: ../../generated/


   serial_connect
   get_output
   to_data


Decorators
-------------------------

.. currentmodule:: brainspace.vtk_interface.decorators

.. autosummary::
   :toctree: ../../generated/

    wrap_input
    wrap_output
    unwrap_input
    unwrap_output
    wrap_func
    unwrap_func
    append_vtk
