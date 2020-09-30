from feature import Feature
import numpy as np


class Size(Feature):

    def __init__(self):
        self._name = 'Size'

    def get_name(self, m_type: int) -> str:
        loc = ['_nucleus', '_cell', '_membrane']
        return self._name + loc[m_type]

    def measure(self, img, m_type: int) -> float:
        mask = img.get_mask(m_type)
        mask_coordinates = np.where(mask != 0)
        value = mask_coordinates[0].size
        value = float(value)
        return value

