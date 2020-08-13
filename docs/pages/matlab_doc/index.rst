.. _matlab_package:

MATLAB Package
======================================

This page contains links to descriptions of all MATLAB code available in this
toolbox as well as the tutorials. The code is divided into five groups:  the
"main object" which performs the computations, "surface handling" functions for
read/writing and parcellating surfaces, "visualization" which plots data, "data
loaders" which loads sample data for our tutorials, and "support functions"
which are requisites for the other functions, but are not intended for usage by
the user.

Tutorials 
----------------
.. toctree::
   :glob:
   :maxdepth: 1

   examples/*

Main Object
--------------------
.. toctree::
   :maxdepth: 1

   main_functionality/gradientmaps

Surface Handling
------------------
.. toctree::
    :maxdepth: 1
    
    main_functionality/read_surface
    main_functionality/write_surface
    main_functionality/combine_surfaces
    main_functionality/split_surfaces
    main_functionality/full2parcel
    main_functionality/parcel2full

Null Hypothesis Testing 
-------------------------
.. toctree::
    :maxdepth: 1
    
    main_functionality/spin_permutations
    main_functionality/compute_mem
    main_functionality/moran_randomization

Visualization
---------------
.. toctree::
   :glob:
   :maxdepth: 1
   
   visualization/*

Data Loaders
---------------
.. toctree::
    :glob:
    :maxdepth: 1

    data_loaders/*
   
Support Functions
------------------
.. toctree::
   :glob:
   :maxdepth: 1

   support_functions/*
