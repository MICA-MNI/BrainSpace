
.. _pymod-vtkinterface:

VTK interface
=================================

Surface mesh functionality provided in BrainSpace is built on top of the
`Visualization Toolkit (VTK) <https://vtk.org/>`_. BrainSpace provides several
wrappers for most data objects and some filters in VTK. Here we present
a subset of this functionality. Please also refer to
:ref:`pypage-vtk_wrapping` document for an introduction to the wrapping
interface.


- :ref:`pysec-vtkinterface-basic_wrapping`
- :ref:`pysec-vtkinterface-wrappers`
- :ref:`pysec-vtkinterface-object_wrappers`
- :ref:`pysec-vtkinterface-pipeline`
- :ref:`pysec-vtkinterface-decorators`



.. _pysec-vtkinterface-basic_wrapping:

Basic wrapping
-------------------------

.. currentmodule:: brainspace.vtk_interface.wrappers

.. autosummary::
   :toctree: ../../generated/

   wrap_vtk
   is_vtk
   is_wrapper


.. _pysec-vtkinterface-wrappers:

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


.. _pysec-vtkinterface-object_wrappers:

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


.. _pysec-vtkinterface-pipeline:

Pipeline functionality
-------------------------

.. currentmodule:: brainspace.vtk_interface.pipeline

.. autosummary::
   :toctree: ../../generated/


   serial_connect
   get_output
   to_data


.. _pysec-vtkinterface-decorators:

Decorators
-------------------------

.. currentmodule:: brainspace.vtk_interface.decorators

.. autosummary::
   :toctree: ../../generated/

    wrap_input
    wrap_output
    unwrap_input
    unwrap_output
    append_vtk
