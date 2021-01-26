import copy
import math

import lic
import numpy as np
from scipy.stats import norm
from scipy import signal
from skimage.color import rgb2gray
from skimage.filters import gaussian, sobel
from skimage.io import imsave, imread
from skimage.util import invert
from skimage.draw import line

sobel_3_v = [[1, 0, -1],[2, 0, -2],[1, 0, -1]]
sobel_3_h = [[1, 2, 1],[0, 0, 0],[-1, -2, -1]]

sobel_7_v = [
    [-3/18, -2/13, -1/10,  0,  1/10, 2/13, 3/18],
    [-3/13,  -2/8,  -1/5,  0,   1/5,  2/8, 3/13],
    [-3/10,  -2/5,  -1/2,  0,   1/2,  2/5, 3/10],
    [ -3/9,  -2/4,  -1/1,  0,   1/1,  2/4,  3/9],
    [-3/10,  -2/5,  -1/2,  0,   1/2,  2/5, 3/10],
    [-3/13,  -2/8,  -1/5,  0,   1/5,  2/8, 3/13],
    [-3/18, -2/13, -1/10,  0,  1/10, 2/13, 3/18]
]

sobel_7_h = [
    [-3/18, -3/13, -3/10, -3/9, -3/10, -3/13, -3/18],
    [-2/13,  -2/8,  -2/5, -2/4,  -2/5,  -2/8, -2/13],
    [-1/10,  -1/5,  -1/2,   -1,  -1/2,  -1/5, -1/10],
    [    0,     0,     0,    0,     0,     0,     0],
    [ 1/10,   1/5,   1/2,    1,   1/2,   1/5,  1/10],
    [ 2/13,   2/8,   2/5,  2/4,   2/5,   2/8,  2/13],
    [ 3/18,  3/13,  3/10,  3/9,  3/10,  3/13,  3/18]
]

def wrap_to_range(val, min, max):
    if val < min:
        val = val + max
    if val >= max:
        val = val - max
    return val


def convolve_point(image, kernel, i, j):
    h, w = image.shape
    d = len(kernel)
    r = math.ceil(d/2)
    sum = 0
    for x in range(0, d):
        for y in range(0, d):
            u = wrap_to_range(i - r + x, 0, h)
            v = wrap_to_range(j - r + y, 0, w)
            sum = sum + image[u][v] * kernel[x][y]
    return sum


def gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def convolve_vector(image, kernel, i, j):
    h, w, ch = image.shape
    d = len(kernel)
    r = math.floor(d/2)
    vector = ()
    for channel in range(0, ch):
        sum = 0
        for x in range(0, d):
            for y in range(0, d):
                u = wrap_to_range(i - r + x, 0, h)
                v = wrap_to_range(j - r + y, 0, w)
                sum = sum + image[u][v][channel] * kernel[x][y]
        vector = vector + (sum,)
    return vector


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


# https://www.kyprianidis.com/p/tpcg2008/jkyprian-tpcg2008.pdf
def calc_structure_tensor(im):
    t = np.ndarray((im.shape[0], im.shape[1], 3))
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            u = convolve_point(im, sobel_7_h, i, j)
            v = convolve_point(im, sobel_7_v, i, j)
            E = np.dot(u, u)
            F = np.dot(u, v)
            G = np.dot(v, v)
            t[i][j] = np.array((E, F, G))
    return t


def smooth_structure_tensor(t, sigma):
    s_t = copy.deepcopy(t)
    kernel = gkern(5, sigma)
    for i in range(0, t.shape[0]):
        for j in range(0, t.shape[1]):
            s_t[i][j] = convolve_vector(t, kernel, i, j)
    return s_t

def smooth_structure_tensor_iteratively(t, sigma, n_of_iterations):
    s_t = copy.deepcopy(t)
    for i in range(0, n_of_iterations):
        s_t = smooth_structure_tensor(s_t, sigma)
    return s_t


def calc_flow_field(s_t):
    f = np.ndarray((s_t.shape[0], s_t.shape[1], 3))
    for i in range(0, s_t.shape[0]):
        for j in range(0, s_t.shape[1]):
            E = s_t[i][j][0]
            F = s_t[i][j][1]
            G = s_t[i][j][2]
            D = E*E - 2*E*G + G*G + 4*F*F
            lambda2 = 0.5 * (E + G - math.sqrt(D))
            d = (lambda2 - G, F)
            length = math.sqrt(d[0]*d[0] + d[1]*d[1])
            t = (d[0], d[1], math.sqrt(length)) if length > 0 else (0, 0.1, 0)
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
    flow_field = calc_flow_field(tensor)
    return smooth_structure_tensor(flow_field, smooth)


def visualize_flow_field(im, f_f):
    im1 = copy.deepcopy(im)
    resolution = 5
    for i in range(0, im1.shape[0], resolution):
        for j in range(0, im1.shape[1], resolution):
            v = f_f[i][j]
            p2 = (int(i + v[0] * 5), int(j + v[1] * 5))
            if p2[0] >= im.shape[0] or p2[0] < 0:
                continue
            if p2[1] >= im.shape[1] or p2[1] < 0:
                continue
            rr, cc = line(i, j, p2[0], p2[1])
            im1[rr, cc] = 255
    return im1


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


def load_image(path):
    return rgb2gray(imread(path))


def gradient_aligned_dog(image, flow_field, low_sigma, high_sigma, kernel_size, w1=1, w2=1):
    gag1 = gradient_aligned_1d_gaussian(image, flow_field, low_sigma, kernel_size)
    gag2 = gradient_aligned_1d_gaussian(image, flow_field, high_sigma, kernel_size)
    gadog = w1 * gag1 - w2 * gag2
    return gadog


def dog(path, low_sigma, high_sigma, epsilon):
    image = load_image(path)
    im = gaussian(image, low_sigma) - gaussian(image, high_sigma)
    return invert(apply_simple_thresholding(im, epsilon))


def fdog(path, low_sigma, high_sigma, blur_sigma, kernel_size, epsilon, length):
    image = load_image(path)
    flow_field = get_flow_field(image, blur_sigma)
    gadog = gradient_aligned_dog(image, flow_field, low_sigma, high_sigma, kernel_size)
    im = apply_simple_thresholding(gadog, epsilon)
    return lic.lic(flow_field[:, :, 0], flow_field[:, :, 1], seed=im, length=length)

def xdog(path, low_sigma, high_sigma, epsilon, fi, p):
    image = load_image(path)
    image = (1 + p) * gaussian(image, low_sigma) - p * gaussian(image, high_sigma)
    return apply_thresholding(image, epsilon, fi)


def xfdog(path, low_sigma, high_sigma, blur_sigma, kernel_size, epsilon, fi, p, length):
    image = load_image(path)
    flow_field = get_flow_field(image, blur_sigma)
    gadog = gradient_aligned_dog(image, flow_field, low_sigma, high_sigma, kernel_size, 1 + p, p)
    im = apply_thresholding(gadog, epsilon, fi)
    return lic.lic(flow_field[:, :, 0], flow_field[:, :, 1], seed=im, length=length)


def xfdog_debug(path_in, path_out, low_sigma, high_sigma, blur_sigma, kernel_size, epsilon, fi, p, length):
    image = load_image(path_in)
    tensor = calc_structure_tensor(image)
    flow_field = calc_flow_field(tensor)
    imsave(path_out + '0_flow_field.png', visualize_flow_field(image, flow_field))
    smoothed_flow_field = copy.deepcopy(flow_field)
    for i in range(1, 2):
        smoothed_flow_field = smooth_structure_tensor(smoothed_flow_field, blur_sigma)
        imsave(path_out + '1_smoothed_flow_field_' + str(i) + '.png', visualize_flow_field(image, smoothed_flow_field))
    lic_vis = lic.lic(smoothed_flow_field[:, :, 0], smoothed_flow_field[:, :, 1], length=length)
    imsave(path_out + '2_lic.png', lic_vis)
    gadog = gradient_aligned_dog(image, smoothed_flow_field, low_sigma, high_sigma, kernel_size, 1 + p, p)
    imsave(path_out + '3_gadog.png', gadog)
    im = apply_thresholding(gadog, epsilon, fi)
    imsave(path_out + '4_thresholding.png', im)
    res = lic.lic(smoothed_flow_field[:, :, 0], smoothed_flow_field[:, :, 1], seed=im, length=length)
    imsave(path_out + '5_result.png', res)