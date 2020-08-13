"""
Scrapper for sphinx-gallery to capture vtk figures.

"""

# This is shamelessly copied from PyVista

from .base import Plotter
from ..vtk_interface.wrappers import BSScalarBarActor


def _get_sg_image_scraper():
    return Scraper()


class Scraper(object):

    def __call__(self, block, block_vars, gallery_conf):
        """
        Called by sphinx-gallery to save the figures generated after running
        example code.
        """
        try:
            from sphinx_gallery.scrapers import figure_rst
        except ImportError:
            raise ImportError('You must install `sphinx_gallery`')
        image_names = list()
        image_path_iterator = block_vars["image_path_iterator"]
        for k, p in Plotter.DICT_PLOTTERS.items():
            fname = next(image_path_iterator)

            for _, lren in p.renderers.items():
                for r in lren:
                    for i in range(r.actors2D.n_items):
                        a = r.actors2D[i]
                        if not isinstance(a, BSScalarBarActor):
                            continue
                        a.labelTextProperty.fontsize = a.labelTextProperty.fontsize * 3

            p.screenshot(fname, scale=3)
            # p.screenshot(fname)
            image_names.append(fname)

        Plotter.close_all()  # close and clear all plotters
        return figure_rst(image_names, gallery_conf["src_dir"])
