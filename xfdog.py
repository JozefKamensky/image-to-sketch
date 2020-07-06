import skimage.data as d
from skimage.color import  rgb2gray
import numpy as np
import math
from skimage.draw import line
from skimage.io import imsave, imread
import copy
from skimage.filters import gaussian
import lic
from scipy.stats import norm
from  skimage.util import img_as_ubyte, invert

# XFDoG - eXtended Flow-based Difference-of-Gaussians
# 7 steps:
# 1 [X] - input image to grayscale
# 2 [X] - calculate structure tensor
# 3 [X] - smooth (blur) structure tensor with Gaussian filter (sigma_c)
# 4 [X] - calculate flow field
# 5 [X] - gradient aligned Difference-of-Gaussians (sigma_e, p)
# 6 [X] - thresholding (epsilon, fi)
# 7 [X] - line integral convolution (sigma_m)


def get_from_matrix(matrix, i, j, default_value):
    if i < 0 or i >= matrix.shape[0]:
        return default_value
    if j < 0 or j >= matrix.shape[1]:
        return default_value
    return matrix[i][j]


def get_from_matrix_interpolate(matrix, i, j):
    if i < 0 or i >= matrix.shape[0]:
        return 0
    if j < 0 or j >= matrix.shape[1]:
        return 0

    i_f = math.floor(i)
    i_c = math.ceil(i)
    j_f = math.floor(j)
    j_c = math.ceil(j)

    if i_f == int(i) or j_f == int(j):
        return get_from_matrix(matrix, int(i), int(j), 0)

    k1 = math.fabs((i_f - i) * (j_f - j))
    p1 = k1 * get_from_matrix(matrix, i_f, j_f, 0)
    k2 = math.fabs((i_f - i) * (j_c - j))
    p2 = k2 * get_from_matrix(matrix, i_f, j_c, 0)
    k3 = math.fabs((i_c - i) * (j_f - j))
    p3 = k3 * get_from_matrix(matrix, i_c, j_f, 0)
    k4 = math.fabs((i_c - i) * (j_c - j))
    p4 = k4 * get_from_matrix(matrix, i_c, j_c, 0)

    if k1 + k2 + k3 + k4 < 0.9:
        print(i, j)
    return p1 + p2 + p3 + p4


def calc_structure_tensor(im):
    t = np.ndarray((im.shape[0], im.shape[1], 3))
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            u = - get_from_matrix(im, i - 1, j - 1, np.zeros(3)) \
                - get_from_matrix(im, i - 1, j, np.zeros(3)) * 2 \
                - get_from_matrix(im, i - 1, j + 1, np.zeros(3)) \
                + get_from_matrix(im, i + 1, j - 1, np.zeros(3)) \
                + get_from_matrix(im, i + 1, j, np.zeros(3)) * 2 \
                + get_from_matrix(im, i + 1, j + 1, np.zeros(3))
            v = - get_from_matrix(im, i - 1, j - 1, np.zeros(3)) \
                - get_from_matrix(im, i, j - 1, np.zeros(3)) * 2 \
                - get_from_matrix(im, i + 1, j - 1, np.zeros(3)) \
                + get_from_matrix(im, i - 1, j + 1, np.zeros(3)) \
                + get_from_matrix(im, i, j + 1, np.zeros(3)) * 2 \
                + get_from_matrix(im, i + 1, j + 1, np.zeros(3))
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


def prepare_1d_gaussian_kernel(sigma, kernel_size):
    s = math.floor(kernel_size / 2)
    x_all = np.arange(-s, s + 1, 1)
    probs = norm.pdf(x_all, 0, sigma)
    p_sum = probs.sum()
    for i in range(0, probs.shape[0]):
        probs[i] = probs[i] / p_sum
    return probs


def get_orthogonal_normalized_vector(v):
    v_x = v[0]
    v_y = v[1]
    if v_x == 0:
        o_x = v_y
        o_y = 0
    elif v_y == 0:
        o_x = 0
        o_y = v_x
    else:
        o_x = v_y
        o_y = -v_x
    l = math.sqrt(o_x*o_x + o_y*o_y)
    return o_x / l, o_y / l


def gradient_aligned_1d_gaussian(im, f_f, sigma, kernel_size):
    im1 = copy.deepcopy(im)
    kernel = prepare_1d_gaussian_kernel(sigma, kernel_size)
    print(kernel)
    kernel_d = math.floor(kernel_size/2)
    for i in range(0, im1.shape[0]):
        for j in range(0, im1.shape[1]):
            o_v = get_orthogonal_normalized_vector((f_f[i][j][0], f_f[i][j][1]))
            res_val = 0
            for k in range(-kernel_d, kernel_d + 1):
                v = (i + k * o_v[0], j + k * o_v[1])
                val = get_from_matrix_interpolate(im1, v[0], v[1])
                res_val += kernel[k + kernel_d] * val
            im1[i][j] = res_val
    return im1


p = 1
epsilon = 0.03
fi = 2
# image = rgb2gray(imread('./in/image2.png'))
image = rgb2gray(d.astronaut())
imsave('./out/input_image.png', image)

tensor = calc_structure_tensor(image)
smoothed_tensor = smooth_structure_tensor(tensor, 1)
flow_field = calc_flow_field(smoothed_tensor)
visualize_flow_field(image, flow_field, 5)
dog1 = gradient_aligned_1d_gaussian(image, flow_field, 2, 9)
dog2 = gradient_aligned_1d_gaussian(image, flow_field, 3.2, 9)
dog_1d = dog1 - dog2
# g1 = gaussian(image, 1)
# g2 = gaussian(image, 1.6)
imsave('./out/g1.png', img_as_ubyte(dog1))
imsave('./out/g2.png', img_as_ubyte(dog2))
imsave('./out/dog.png', img_as_ubyte(dog_1d))
# imsave('./out/gauss1.png', g1)
# imsave('./out/gauss2.png', g2)
for i in range(0, dog_1d.shape[0]):
    for j in range(0, dog_1d.shape[1]):
        if dog_1d[i][j] >= epsilon:
            dog_1d[i][j] = 1
        else:
            dog_1d[i][j] = 1 + math.tanh(fi * (dog_1d[i][j] - epsilon))
dog_1d = img_as_ubyte(dog_1d)
imsave('./out/after_thresholding.png', dog_1d)
dog_1d = lic.lic(flow_field[:, :, 0], flow_field[:, :, 1], seed=dog_1d, length=20)
imsave('./out/xfdog.png', img_as_ubyte(dog_1d))

lic_result = lic.lic(flow_field[:, :, 0], flow_field[:, :, 1], length=20)
imsave('./out/ETF.png', img_as_ubyte(lic_result))
