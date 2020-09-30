from feature import Feature
import numpy as np


class TotalIntensity(Feature):

    def __init__(self):
        self._name = 'Total Intensity'

    def get_name(self, m_type: int) -> str:
        loc = ['_nucleus', '_cell', '_membrane']
        return self._name + loc[m_type]

    def measure(self, img, m_type: int) -> float:
        mask = img.get_mask(m_type)
        raw = img.get_raw(False)

        mask_coordinates = np.where(mask != 0)
        value = raw[mask_coordinates].sum()
        value = int(value)
        value = float(value)
        return value

