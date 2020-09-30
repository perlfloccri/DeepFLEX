from feature import Feature
import numpy as np


class MeanIntensity(Feature):

    def __init__(self):
        self._name = 'Mean Intensity'

    def get_name(self, m_type: int) -> str:
        loc = ['_nucleus', '_cell', '_membrane']
        return self._name + loc[m_type]

    def measure(self, img, m_type: int) -> float:
        mask = img.get_mask(m_type)
        raw = img.get_raw(False)

        mask_coordinates = np.where(mask != 0)
        if raw[mask_coordinates].sum() == 0:
            value = 0
        else:
            value = raw[mask_coordinates].sum()/mask_coordinates[0].size
        return value

