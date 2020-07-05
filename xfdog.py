import skimage.data as d
from skimage.color import  rgb2gray
import numpy as np
import math
from skimage.draw import line
from skimage.io import imsave, imread
import copy
from skimage.filters import gaussian, difference_of_gaussians
import lic

# XFDoG - eXtended Flow-based Difference-of-Gaussians
# 7 steps:
# 1 [X] - input image to grayscale
# 2 [X] - calculate structure tensor
# 3 [X] - smooth (blur) structure tensor with Gaussian filter (sigma_c)
# 4 [X] - calculate flow field
# 5 [ ] - gradient aligned Difference-of-Gaussians (sigma_e, p)
# 6 [ ] - thresholding (epsilon, fi)
# 7 [ ] - line integral convolution (sigma_m)


def get_from_matrix(matrix, i, j):
    if i < 0 or i >= matrix.shape[0]:
        return np.zeros(3)
    if j < 0 or j >= matrix.shape[1]:
        return np.zeros(3)
    return matrix[i][j]


def calc_structure_tensor(im):
    t = np.ndarray((im.shape[0], im.shape[1], 3))
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            u = - get_from_matrix(im, i - 1, j - 1) \
                - get_from_matrix(im, i - 1, j) * 2 \
                - get_from_matrix(im, i - 1, j + 1) \
                + get_from_matrix(im, i + 1, j - 1) \
                + get_from_matrix(im, i + 1, j) * 2 \
                + get_from_matrix(im, i + 1, j + 1)
            v = - get_from_matrix(im, i - 1, j - 1) \
                - get_from_matrix(im, i, j - 1) * 2 \
                - get_from_matrix(im, i + 1, j - 1) \
                + get_from_matrix(im, i - 1, j + 1) \
                + get_from_matrix(im, i, j + 1) * 2 \
                + get_from_matrix(im, i + 1, j + 1)
            t[i][j] = np.array((np.dot(u, u), np.dot(v, v), np.dot(u, v)))
    return t


def smooth_structure_tensor(t, sigma):
    s_t = copy.deepcopy(t)
    return gaussian(s_t, sigma=sigma)


def calc_flow_field(s_t):
    f = np.ndarray((s_t.shape[0], s_t.shape[1], 3))
    for i in range(0, s_t.shape[0]):
        for j in range(0, s_t.shape[1]):
            x = s_t[i][j][0]
            y = s_t[i][j][1]
            z = s_t[i][j][2]
            lambda1 = 0.5 * (y + x + math.sqrt(y*y - 2*x*y + x*x + 4*z*z))
            d = np.array((x - lambda1, z))
            d_n = d/np.linalg.norm(d)
            len_d = math.sqrt(d[0]*d[0] + d[1]*d[1])
            t = (d_n[0], d_n[1], math.sqrt(lambda1)) if len_d > 0 else (0, 1, 0)
            f[i][j] = np.array(t)
    return f


def visualize_flow_field(im, f_f, res):
    im1 = copy.deepcopy(im)
    for i in range(0, im1.shape[0], res):
        for j in range(0, im1.shape[1], res):
            v = f_f[i][j]
            p2 = (int(i + v[0] * 5), int(j + v[1] * 5))
            if p2[0] >= im1.shape[0] or p2[0] < 0:
                continue
            if p2[1] >= im1.shape[1] or p2[1] < 0:
                continue
            rr, cc = line(i, j, p2[0], p2[1])
            im1[rr, cc] = 0.5
            # im1[rr, cc] = v[2]
    imsave('./out/flow_field.png', im1)


image = rgb2gray(imread('./in/image2.png'))
imsave('./out/input_image.png', image)

tensor = calc_structure_tensor(image)
smoothed_tensor = smooth_structure_tensor(tensor)
flow_field = calc_flow_field(smoothed_tensor)
visualize_flow_field(image, flow_field, 5)
lic_result = lic.lic(flow_field[:, :, 0], flow_field[:, :, 1], length=20)
imsave('./out/ETF.png', lic_result)
