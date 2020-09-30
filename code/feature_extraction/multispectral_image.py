import numpy as np
from image import Image


class MSImage(object):

    def __init__(self, images: list, idx_img=int):
        self._images = images
        self._idx_img = idx_img

    def get_raw(self, vector: bool) -> np.ndarray:
        img_array = []

        if vector:
            for img in self._images:
                img_array.extend(img.get_raw(True))
        else:

            for img in self._images:
                img_array.append(img.get_raw(False))

        return np.asarray(img_array)

    def get_idx_img(self):
        return self._idx_img

    def get_features(self, vector: bool) -> np.ndarray:
        features = []

        if vector:
            for img in self._images:
                features.extend(img.get_feature_vector())
        else:
            for img in self._images:
                features.append(img.get_feature_vector())

        return np.asarray(features)

    def __getitem__(self, idx) -> Image:
        return self._images[idx]

    def __len__(self) -> int:
        return len(self._images)




