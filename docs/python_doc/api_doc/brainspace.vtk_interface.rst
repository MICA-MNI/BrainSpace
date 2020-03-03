
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
- :ref:`pysec-vtkinterface-pipeline`
- :ref:`pysec-vtkinterface-wrappers`

    - :ref:`pysec-vtkinterface-object_wrappers`
    - :ref:`pysec-vtkinterface-algorithm`
    - :ref:`pysec-vtkinterface-mapper`
    - :ref:`pysec-vtkinterface-actor`
    - :ref:`pysec-vtkinterface-lut`
    - :ref:`pysec-vtkinterface-renderer`
    - :ref:`pysec-vtkinterface-property`
    - :ref:`pysec-vtkinterface-misc`

- :ref:`pysec-vtkinterface-decorators`



.. _pysec-vtkinterface-basic_wrapping:

Basic wrapping
-------------------------

.. currentmodule:: brainspace.vtk_interface.wrappers.base

.. autosummary::
   :toctree: ../../generated/

    wrap_vtk
    is_vtk
    is_wrapper
    BSVTKObjectWrapper


.. _pysec-vtkinterface-pipeline:

Pipeline functionality
-------------------------

.. currentmodule:: brainspace.vtk_interface.pipeline

.. autosummary::
   :toctree: ../../generated/

    serial_connect
    get_output
    to_data
    connect


.. _pysec-vtkinterface-wrappers:

VTK wrappers
-------------------------

.. _pysec-vtkinterface-object_wrappers:

Data objects
++++++++++++

.. currentmodule:: brainspace.vtk_interface.wrappers.data_object

.. autosummary::
   :toctree: ../../generated/

    BSDataObject
    BSTable
    BSCompositeDataSet
    BSDataSet
    BSPointSet
    BSPolyData
    BSUnstructuredGrid


.. _pysec-vtkinterface-algorithm:

Algorithms
++++++++++

.. currentmodule:: brainspace.vtk_interface.wrappers.algorithm

.. autosummary::
   :toctree: ../../generated/

    BSAlgorithm
    BSPolyDataAlgorithm
    BSWindowToImageFilter
    BSImageWriter
    BSBMPWriter
    BSJPEGWriter
    BSPNGWriter
    BSPostScriptWriter
    BSTIFFWriter


.. _pysec-vtkinterface-mapper:

Mappers
+++++++

.. currentmodule:: brainspace.vtk_interface.wrappers.algorithm

.. autosummary::
   :toctree: ../../generated/

    BSDataSetMapper
    BSPolyDataMapper
    BSLabeledContourMapper
    BSLabeledDataMapper
    BSLabelPlacementMapper
    BSPolyDataMapper2D
    BSTextMapper2D


.. _pysec-vtkinterface-actor:

Actors
++++++

.. currentmodule:: brainspace.vtk_interface.wrappers.actor

.. autosummary::
   :toctree: ../../generated/

    BSActor2D
    BSScalarBarActor
    BSTexturedActor2D
    BSTextActor
    BSActor


.. _pysec-vtkinterface-lut:

Lookup tables
+++++++++++++

.. currentmodule:: brainspace.vtk_interface.wrappers.lookup_table

.. autosummary::
   :toctree: ../../generated/


    BSScalarsToColors
    BSLookupTable
    BSLookupTableWithEnabling
    BSWindowLevelLookupTable
    BSColorTransferFunction
    BSDiscretizableColorTransferFunction


.. _pysec-vtkinterface-renderer:

Rendering
+++++++++

.. currentmodule:: brainspace.vtk_interface.wrappers.renderer

.. autosummary::
   :toctree: ../../generated/

    BSRenderer
    BSRenderWindow
    BSRenderWindowInteractor
    BSGenericRenderWindowInteractor
    BSInteractorStyle
    BSInteractorStyleJoystickCamera
    BSInteractorStyleJoystickActor
    BSInteractorStyleTerrain
    BSInteractorStyleRubberBandZoom
    BSInteractorStyleTrackballActor
    BSInteractorStyleTrackballCamera
    BSInteractorStyleImage
    BSInteractorStyleRubberBandPick
    BSInteractorStyleSwitchBase
    BSInteractorStyleSwitch
    BSCamera


.. _pysec-vtkinterface-property:

Properties
++++++++++

.. currentmodule:: brainspace.vtk_interface.wrappers.property

.. autosummary::
   :toctree: ../../generated/

   BSProperty
   BSProperty2D
   BSTextProperty


.. _pysec-vtkinterface-misc:

Miscellanea
+++++++++++

.. currentmodule:: brainspace.vtk_interface.wrappers.misc

.. autosummary::
   :toctree: ../../generated/

    BSCellArray
    BSGL2PSExporter
    BSCollection
    BSPropCollection
    BSActor2DCollection
    BSActorCollection
    BSProp3DCollection
    BSMapperCollection
    BSRendererCollection
    BSPolyDataCollection
    BSTextPropertyCollection
    BSCoordinate


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
