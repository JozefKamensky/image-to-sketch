from skimage.filters import gaussian
from skimage.io import imsave
from skimage import util
import skimage.data as d
import math
import sys

k = 1.6
# threshold when pixel becomes white
epsilon = float(sys.argv[1])
# strength of edge sharpening effect
p = float(sys.argv[2])
# sharpness of black-white transition
fi = float(sys.argv[3])

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
    imsave('out/xdog_' + str(sigma) + '.png', util.img_as_ubyte(im))
