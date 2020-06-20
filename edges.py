from skimage import filters
from skimage.io import imsave, imread
from skimage import util
import sys

path = sys.argv[1]
image = imread(path, as_gray=True)
edge_roberts = filters.roberts(image)
edge_sobel = filters.sobel(image)
edge_scharr = filters.scharr(image)
edge_prewitt = filters.prewitt(image)

path = path[:path.rfind('.')]

imsave(path + '_roberts.png', util.invert(util.img_as_ubyte(edge_roberts)))
imsave(path + '_sobel.png', util.invert(util.img_as_ubyte(edge_sobel)))
imsave(path + '_scharr.png', util.invert(util.img_as_ubyte(edge_scharr)))
imsave(path + '_prewitt.png', util.invert(util.img_as_ubyte(edge_prewitt)))

