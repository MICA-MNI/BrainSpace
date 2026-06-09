from .surface_report import write_surface_report
from .threemica_report import (available_threemica_surface_maps,
                               write_brainspace_threemica_report,
                               write_threemica_report)


__all__ = ['build_plotter',
           'plot_surf',
           'plot_hemispheres',
           'write_surface_report',
           'available_threemica_surface_maps',
           'write_brainspace_threemica_report',
           'write_threemica_report']


def __getattr__(name):
    if name in {'build_plotter', 'plot_surf', 'plot_hemispheres'}:
        from .surface_plotting import build_plotter, plot_surf, plot_hemispheres
        globals().update({
            'build_plotter': build_plotter,
            'plot_surf': plot_surf,
            'plot_hemispheres': plot_hemispheres,
        })
        return globals()[name]
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
