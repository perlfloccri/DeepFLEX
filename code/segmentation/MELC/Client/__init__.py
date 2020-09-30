# --------------------------------------------------------
# Multi-Epitope-Ligand Cartography (MELC) phase-contrast image based segmentation pipeline
#
#
# Written by Filip Mivalt
# --------------------------------------------------------

"""
    A package containing fucntions and classes for data handling.
    The package performs many tasks employing different sub-packages.


    Sub-packages
    ----------
    Dataset:
        - Contains classes representing data such as MELC Run providing us with ready-to-use images

    Files:
        - Set of functions for fast and easy handling with folders. Contains functions for creating,
            removing folders and so on.

    Registration:
        - Set of functions for image registration of MELC data.
            Now implemented only translational pixel-wise registration.
            These functions are mainly used inside other classes and functions.


    """


