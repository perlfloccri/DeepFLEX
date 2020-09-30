from scipy import fftpack as fp
import numpy as np
import math
import matplotlib.pyplot as plt


def filterLowFrequencies(image=None,n=50):
    F1 = fp.fft2(image.astype(float))
    F2 = fp.fftshift(F1)

    (w, h) = image.shape

    # Gauss filtering of frequencies
    x, y = np.meshgrid(np.linspace(-math.ceil(w/2), math.ceil(w/2), w), np.linspace(-math.ceil(h/2), math.ceil(h/2), h))
    d = np.sqrt(x * x + y * y)
    sigma, mu = float(n), 0.0
    a = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    a = a.astype(np.complex128)

    F2 = F2 * a # Remove frequencies

    return fp.ifft2(fp.ifftshift(F2)).real


def visualize_frequencies(annotated_images):
    number = annotated_images.__len__()
    plt.figure(1)
    for index,image in enumerate(annotated_images):
        plt.subplot(2,number,index+1)
        F1 = fp.fft2(image.astype(float))
        F2 = fp.fftshift(F1)
        plt.imshow(image, cmap='gray');
        plt.axis('off')
        plt.subplot(2, number, index + number + 1)
        plt.imshow((20 * np.log10(0.1 + F2)).astype(int), cmap=plt.cm.gray)
        plt.axis('off')
