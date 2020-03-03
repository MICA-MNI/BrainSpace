.. surface_to_graph_matlab:

surface_to_graph
==============================

Synopsis
---------

Converts a surface to a graph (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/surface_manipulation/surface_to_graph.m>`_).


Usage 
----------
::

    G = surface_to_graph(S,distance)
    G = surface_to_graph(S,distance,mask)
    G = surface_to_graph(S,distance,mask,removeDegreeZero)

- *G*: Output graph
- *S*: a surface. 
- *distance*: either 'geodesic' for returning a weighted graph, or 'mesh' for unweighted. 
- *mask*: a logical mask where True denotes vertices to remove (Default: []).
- *removeDegreeZero*: a logical, if true then vertices with degree 0 are removed from the output graph. 

Description
------------
Converts surfaces readable by
:ref:`convert_surface_matlab` into a graph. Masks can be used to remove portions
of the surfaces (e.g. midline)
