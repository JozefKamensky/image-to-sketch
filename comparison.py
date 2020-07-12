from imageio import imsave
from skimage.color import rgb2gray
from skimage.data import astronaut
from skimage.io import imread

from utils import dog, xdog, fdog, xfdog

low_sigmas = [1]
blur_sigmas = [0.1]
kernel_sizes = [7]
epsilons = [1]
fis = [1]
ps = [100]

counter = 0
out_f = open('params.txt', 'w')
for low_sigma in low_sigmas:
    high_sigma = 6.8 * low_sigma
    for blur_sigma in blur_sigmas:
        for kernel_size in kernel_sizes:
            for epsilon in epsilons:
                for fi in fis:
                    for p in ps:
                        counter += 1
                        out_f.write(str(counter) + ', ' + str(low_sigma) + ', ' + str(high_sigma) + ', ' + str(blur_sigma) + ', ' + str(kernel_size) + ', ' + str(epsilon) + ', ' + str(fi) + ', ' + str(p) + '\n')
                        # image = rgb2gray(imread('./in/image3.png'))
                        image = rgb2gray(astronaut())
                        imsave('./out/' + str(counter) + '.png', xfdog(image, low_sigma, high_sigma, blur_sigma, kernel_size, epsilon, fi, p))
out_f.close()
