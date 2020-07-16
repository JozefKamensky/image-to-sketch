import copy
import math

import lic
import numpy as np
from scipy.stats import norm
from skimage.filters import gaussian
from skimage.io import imsave
from skimage.util import invert


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


def get_flow_field(image, smooth):
    tensor = calc_structure_tensor(image)
    smoothed_tensor = smooth_structure_tensor(tensor, smooth)
    return calc_flow_field(smoothed_tensor)


def apply_simple_thresholding(image, epsilon):
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i][j] >= epsilon:
                image[i][j] = 1
            else:
                image[i][j] = 0
    return image


def apply_thresholding(image, epsilon, fi):
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i][j] >= epsilon:
                image[i][j] = 1
            else:
                image[i][j] = 1 + math.tanh(fi * (image[i][j] - epsilon))
    return image


def gradient_aligned_dog(image, flow_field, low_sigma, high_sigma, kernel_size, w1=1, w2=1):
    gag1 = gradient_aligned_1d_gaussian(image, flow_field, low_sigma, kernel_size)
    gag2 = gradient_aligned_1d_gaussian(image, flow_field, high_sigma, kernel_size)
    gadog = w1 * gag1 - w2 * gag2
    return gadog


def dog(image, low_sigma, high_sigma, epsilon):
    im = gaussian(image, low_sigma) - gaussian(image, high_sigma)
    return invert(apply_simple_thresholding(im, epsilon))


def fdog(image, low_sigma, high_sigma, blur_sigma, kernel_size, epsilon, length):
    flow_field = get_flow_field(image, blur_sigma)
    gadog = gradient_aligned_dog(image, flow_field, low_sigma, high_sigma, kernel_size)
    im = apply_simple_thresholding(gadog, epsilon)
    return lic.lic(flow_field[:, :, 0], flow_field[:, :, 1], seed=im, length=length)

def xdog(image, low_sigma, high_sigma, epsilon, fi, p):
    image = (1 + p) * gaussian(image, low_sigma) - p * gaussian(image, high_sigma)
    return apply_thresholding(image, epsilon, fi)


def xfdog(image, low_sigma, high_sigma, blur_sigma, kernel_size, epsilon, fi, p, length):
    flow_field = get_flow_field(image, blur_sigma)
    gadog = gradient_aligned_dog(image, flow_field, low_sigma, high_sigma, kernel_size, 1 + p, p)
    im = apply_thresholding(gadog, epsilon, fi)
    return lic.lic(flow_field[:, :, 0], flow_field[:, :, 1], seed=im, length=length)
