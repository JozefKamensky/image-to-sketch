import skimage.data as d
from skimage.color import rgb2lab, rgb2gray
import numpy as np
import math
from skimage.draw import line
from skimage.io import imsave


def euclidean(x, y):
    return math.sqrt(math.pow(x[0] - y[0], 2) + math.pow(x[1] - y[1], 2) + math.pow(x[2] - y[2], 2))


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


def smooth_structure_tensor(t):
    s_t = np.ndarray((t.shape[0], t.shape[1], 3))
    # horizontal smoothing
    for i in range(0, t.shape[0]):
        for j in range(0, t.shape[1]):
            s_t[i][j] = get_from_matrix(t, i, j - 2) / 16 \
                + get_from_matrix(t, i, j - 1) / 4 \
                + get_from_matrix(t, i, j) * 6 / 16 \
                + get_from_matrix(t, i, j + 1) / 4 \
                + get_from_matrix(t, i, j + 2) / 16
    # vertical smoothing
    for i in range(0, t.shape[0]):
        for j in range(0, t.shape[1]):
            s_t[i][j] = get_from_matrix(t, i - 2, j) / 16 \
                + get_from_matrix(t, i - 1, j) / 4 \
                + get_from_matrix(t, i, j) * 6 / 16 \
                + get_from_matrix(t, i + 1, j) / 4 \
                + get_from_matrix(t, i + 2, j) / 16
    return s_t


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


def visualize_flow_field(im, f_f):
    resolution = 5
    for i in range(0, im.shape[0], resolution):
        for j in range(0, im.shape[1], resolution):
            v = f_f[i][j]
            p2 = (int(i + v[0] * 10), int(j + v[1] * 10))
            if p2[0] >= im.shape[0] or p2[0] < 0:
                continue
            if p2[1] >= im.shape[1] or p2[1] < 0:
                continue
            rr, cc = line(i, j, p2[0], p2[1])
            im[rr, cc] = v[2]
    imsave('./out/flow_field.png', im)


def visualize_etf(shape, f_f):
    im = np.zeros(shape)
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            v = f_f[i][j]
            p2 = (int(i + v[0] * 5), int(j + v[1] * 5))
            if p2[0] >= im.shape[0] or p2[0] < 0:
                continue
            if p2[1] >= im.shape[1] or p2[1] < 0:
                continue
            rr, cc = line(i, j, p2[0], p2[1])
            im[rr, cc] = v[2]
    imsave('./out/edge_tangent_flow.png', im)


imsave('./out/input_image.png', d.astronaut())
image = rgb2gray(d.astronaut())
tensor = calc_structure_tensor(image)
smoothed_tensor = smooth_structure_tensor(tensor)
flow_field = calc_flow_field(smoothed_tensor)
visualize_flow_field(d.astronaut(), flow_field)
visualize_etf(image.shape, flow_field)
