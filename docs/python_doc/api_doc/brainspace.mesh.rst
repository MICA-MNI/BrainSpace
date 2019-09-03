Mesh
=======================

BrainSpace provides basic functionality for working with surface meshes. This
functionality is built on top of the
`Visualization Toolkit (VTK) <https://vtk.org/>`_.

- :ref:`Read/Write functionality<Read/Write functionality>`
- :ref:`Surface creation<Surface creation>`
- :ref:`Elements<Elements>`
- :ref:`Connectivity<Connectivity>`
- :ref:`Operations on meshes<Operations on meshes>`
- :ref:`Operations on mesh data<Operations on mesh data>`
- :ref:`Mesh clustering<Mesh clustering>`


Read/Write functionality
-------------------------

.. currentmodule:: brainspace.mesh.mesh_io

.. autosummary::
   :toctree: ../../generated/


   load_surface
   save_surface


Surface creation
-------------------------

.. currentmodule:: brainspace.mesh.mesh_creation

.. autosummary::
   :toctree: ../../generated/

   build_polydata
   to_lines
   to_vertex


Elements
------------------------------

.. currentmodule:: brainspace.mesh.mesh_elements

.. autosummary::
   :toctree: ../../generated/


   get_cells
   get_points
   get_edges


Connectivity
------------------------------

.. currentmodule:: brainspace.mesh.mesh_elements

.. autosummary::
   :toctree: ../../generated/

   get_cell2point_connectivity
   get_point2cell_connectivity

   get_cell_neighbors

   get_immediate_adjacency
   get_ring_adjacency

   get_immediate_distance
   get_ring_distance


Operations on meshes
---------------------

.. currentmodule:: brainspace.mesh.mesh_operations

.. autosummary::
   :toctree: ../../generated/


   drop_cells
   mask_cells
   select_cells

   drop_points
   mask_points
   select_points


Operations on mesh data
-----------------------

.. currentmodule:: brainspace.mesh.array_operations

.. autosummary::
   :toctree: ../../generated/


   compute_cell_area
   compute_cell_center
   get_n_adjacent_cells

   map_celldata_to_pointdata
   map_pointdata_to_celldata
   compute_point_area
   get_connected_components
   get_labeling_border
   get_parcellation_centroids
   propagate_labeling






Mesh clustering
-----------------------

Clustering and sampling of surface vertices.


.. currentmodule:: brainspace.mesh.mesh_cluster

.. autosummary::
   :toctree: ../../generated/


   cluster_points
   sample_points_clustering
   sample_points_decimation


