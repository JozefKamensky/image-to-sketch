from imageio import imsave
from skimage.color import rgb2gray
from skimage.data import astronaut
from utils import dog, xdog, fdog, xfdog

low_sigma = 1
high_sigma = 1.6
blur_sigma = 1
kernel_size = 5
epsilon = 0.4
fi = 0.01
p = 1

image = rgb2gray(astronaut())
imsave('./out/dog.png', dog(image, low_sigma, high_sigma))

image = rgb2gray(astronaut())
imsave('./out/xdog.png', xdog(image, low_sigma, high_sigma, epsilon, fi, p))

image = rgb2gray(astronaut())
imsave('./out/fdog.png', fdog(image, low_sigma, high_sigma, blur_sigma, kernel_size))

image = rgb2gray(astronaut())
imsave('./out/xfdog.png', xfdog(image, low_sigma, high_sigma, blur_sigma, kernel_size, epsilon, fi, p))