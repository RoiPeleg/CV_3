import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[y,x]...], [[dU,dV]...] for each points
    """
    if win_size % 2 == 0:
        raise Exception("window size must be odd")
    Ix = cv2.Sobel(im2, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(im2, cv2.CV_64F, 0, 1, ksize=5)
    It = im2 - im1
    points = []
    d = []
    for i in range(win_size, im1.shape[0] - win_size + 1, step_size):
        for j in range(win_size, im1.shape[1] - win_size + 1, step_size):
            starti, startj, endi, endj = i - win_size // 2, j - win_size // 2, i + win_size // 2 + 1, j + win_size // 2 + 1
            b = -(It[starti:endi, startj:endj]).reshape(win_size ** 2, 1)
            A = np.asmatrix(np.concatenate((Ix[starti:endi, startj:endj].reshape(win_size ** 2, 1),
                                            Iy[starti:endi, startj:endj].reshape(win_size ** 2, 1)), axis=1))
            values, vec = np.linalg.eig(A.T * A)
            values.sort()
            values = values[::-1]
            if values[0] >= values[1] > 1 and values[0] / values[1] < 100:
                # v = (A.T * A).I * A.T * b
                v = np.array(np.dot(np.linalg.pinv(A), b))
                points.append(np.array([j, i]))
                d.append(v[::-1].copy())
    return np.array(points), np.array(d)


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    ker = cv2.getGaussianKernel(5, 1.1)
    ker = (ker * ker.T)
    ker = ker / ker.sum()
    pyrlst = [img - cv2.filter2D(img, -1, ker)]
    ls = [img]
    for i in range(1, levels):
        if img.ndim == 3:
            temp = cv2.filter2D(ls[i - 1][::2, ::2, :], -1, ker)
            ls.append(ls[i - 1][::2, ::2, :].copy())
            pyrlst.append(ls[i - 1][::2, ::2, :] - temp)
        else:
            temp = cv2.filter2D(ls[i - 1][::2, ::2], -1, ker)
            ls.append(ls[i - 1][::2, ::2].copy())
            pyrlst.append(ls[i - 1][::2, ::2] - temp)
    return pyrlst


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    i = -1
    gu = [lap_pyr[-1]*2]
    lap_pyr.reverse()
    ker = cv2.getGaussianKernel(5, 1.1)
    ker = ker * ker.T
    ker = (ker / ker.sum()) * 4
    for I in lap_pyr:
        if i != -1:
            gu.append(I + gaussExpand(gu[i], ker))
        i += 1
    print(len(gu), len(lap_pyr))
    return gu[-1]


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pyrlst = [img]
    for i in range(1, levels):
        temp = cv2.filter2D(pyrlst[i - 1], -1, cv2.getGaussianKernel(5, 0.3 * (4 * 0.5 - 1) + 0.8))
        pyrlst.append(temp[::2, ::2].copy())
    return pyrlst


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    if img.ndim != 3:
        r = np.zeros((img.shape[0] * 2, img.shape[1] * 2), dtype=float)
        r[::2, ::2] = img.copy()
    else:
        r = np.zeros(((img.shape[0] * 2), (img.shape[1] * 2), 3), dtype=float)
        r[::2, ::2, :] = img.copy()
    return cv2.filter2D(r, -1, gs_k)


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """
    ker = cv2.getGaussianKernel(5, 1.1)
    ker = ker * ker.T
    ker = (ker / ker.sum())*4
    im1_p = laplaceianReduce(img_1, levels)
    im2_p = laplaceianReduce(img_2, levels)
    mask_p = gaussianPyr(mask, levels)
    im1_p.reverse()
    im2_p.reverse()
    mask_p.reverse()
    merg = [(im1_p[0] * mask_p[0]) + (1 - mask_p[0]) * im2_p[0]]
    for i in range(1, levels):
        last = gaussExpand(merg[i-1], ker)
        if i == levels - 1:
            last = last[:-1, :-1, :].copy()
        merg.append(last + (im1_p[i] * mask_p[i]) + (1 - mask_p[i]) * im2_p[i])
    merg.reverse()
    t = np.zeros((merg[0].shape[0]+1, merg[0].shape[1]+1, 3))
    t[:merg[0].shape[0], :merg[0].shape[1], :] = merg[0].copy()
    merg[0] = t
    result = laplaceianExpand(merg)[:-1, :-1]
    return img_1 * mask + (1 - mask) * img_2, result
