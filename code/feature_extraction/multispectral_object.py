import numpy as np
from multispectral_image import MSImage


class MSObject(MSImage):

    def __init__(self, images: np.ndarray, pat_id: str, idx_obj: int, label: int, idx_img: int, min_x: int=0, max_x: int=0, min_y: int=0, max_y: int=0):
        self._images = images
        self._idx_obj = idx_obj
        self._label = label
        self._idx_img = idx_img
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
        self._pat_id = pat_id

    def get_pat_id(self):
        return self._pat_id

    def get_idx_obj(self):
        return self._idx_obj

    def get_label(self):
        return self._label

    def set_label(self, label):
        self._label = label

    def get_min_x(self):
        return self._min_x

    def get_max_x(self):
        return self._max_x

    def get_min_y(self):
        return self._min_y

    def get_max_y(self):
        return self._max_y

    def get_mean_x(self):
        mean_x = self._min_x + (self._max_x - self._min_x)/2
        return mean_x

    def get_mean_y(self):
        mean_y = self._min_y + (self._max_y - self._min_y)/2
        return mean_y


