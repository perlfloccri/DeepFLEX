from feature import Feature
import numpy as np
from skimage.morphology import convex_hull_image as convhull


class Solidity(Feature):

    def __init__(self):
        self._name = 'Solidity'

    def get_name(self, m_type: int) -> str:
        loc = ['_nucleus', '_cell', '_membrane']
        return self._name + loc[m_type]

    def measure(self, img, m_type: int) -> float:
        mask = img.get_mask(m_type)

        mask_one = mask / mask.max()

        #filled area

        img_inv = (mask_one == 1)
        k = convhull(img_inv)
        k_coordinates = np.where(k)
        filled_area = k_coordinates[0].size

        #area
        a_coordinates = np.where(mask_one == 1)
        area = a_coordinates[0].size

        value = area/filled_area
        return value
