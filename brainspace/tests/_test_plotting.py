""" Test datasest """


import pytest

import numpy as np
import PIL

import vtk

from brainspace.vtk_interface.pipeline import to_data
from brainspace.plotting.base import Plotter
from brainspace.plotting import plot_surf, plot_hemispheres


def test_plotter():
    s = to_data(vtk.vtkSphereSource())

    p = Plotter(offscreen=True)
    ren0 = p.AddRenderer(row=0, col=0)
    ac0 = ren0.AddActor()
    m0 = ac0.SetMapper(inputdata=s)

    p.show(embed_nb=False)
    img = p.screenshot()
    assert isinstance(img, PIL.Image.Image)

    p.close()


def test_plot_surf():
    s1 = to_data(vtk.vtkSphereSource())
    s2 = to_data(vtk.vtkSphereSource())

    surfs = {'s1': s1, 's2': s2}
    layout = np.array([['s1', 's2'], ['s2', 's2']])
    plot_surf(surfs, layout, offscreen=True)


def test_plot_hemispheres():
    s1 = to_data(vtk.vtkSphereSource())
    s2 = to_data(vtk.vtkSphereSource())

    plot_hemispheres(s1, s2, offscreen=True)
