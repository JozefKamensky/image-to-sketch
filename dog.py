from skimage.filters import gaussian
from skimage.io import imsave
from skimage import util
import skimage.data as d

k = 1.6
epsilon = 0.05

image = d.camera()
sigmas = [1, 2, 3, 4, 5, 6]
for sigma in sigmas:
    low_sigma = sigma
    high_sigma = k * sigma
    im = gaussian(image, low_sigma) - gaussian(image, high_sigma)
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            if im[i][j] >= epsilon:
                im[i][j] = 0
            else:
                im[i][j] = 1
    imsave('out/dog_' + str(sigma) + '.png', util.img_as_ubyte(im))
