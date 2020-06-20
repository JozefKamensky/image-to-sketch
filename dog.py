from skimage.filters import gaussian, difference_of_gaussians
from skimage.io import imsave, imread
from skimage.transform import rescale
from skimage import util
import skimage.data as d
import math

k = 1.6
epsilon = 0.3
p = 18
fi = 0.6
image = d.camera()

sigmas = [1, 2, 3, 4, 5, 6]
for sigma in sigmas:
    low_sigma = sigma
    high_sigma = k * sigma
    im = (1 + p) * gaussian(image, low_sigma) - p * gaussian(image, high_sigma)
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            if im[i][j] >= epsilon:
                im[i][j] = 1
            else:
                im[i][j] = 1 + math.tanh(fi * (im[i][j] - epsilon))
    imsave('out/dog_' + str(sigma) + '.png', util.img_as_ubyte(im))
