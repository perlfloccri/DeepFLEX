# --------------------------------------------------------
# Multi-Epitope-Ligand Cartography (MELC) phase-contrast image based segmentation pipeline
#
#
# Written by Filip Mivalt
# --------------------------------------------------------


class Pointer(object):
    """
            A class behaving as a pointer in other programming languages.
            Used for keeping of the changing class value, passed into sub(sub)classes.


            ...

            Attributes
            ----------
            PATH_DATA : str
                Path to the MELC Run raw data folder containing source and bleach folder with *.png files.
                Defined during initialization, by input parameter.

            Methods
            -------
            get_average_phase()
                Average phase image of registered phase contrast images from source (only fluorescence image corresponding phase
                contrast images, not bleaching) folder of original raw data.

            Example
            -------
            x_ptr = Pointer(5)
            print(x_ptr)
            print(x_ptr.get())


            y = x_ptr
            print(y)
            print(y.get())

            x_ptr.set(3)
            print(x.get())
            print(y.get())

            Output
            ------
            - PointerObject
            - 5

            -PointerObject
            - 5

            - 3
            - 3


        """
    def __init__(self, value=None):
        self._x = value

    def get(self):
        return self._x

    def set(self, value):
        self._x = value

    def add(self, value=1):
        self._x += value
