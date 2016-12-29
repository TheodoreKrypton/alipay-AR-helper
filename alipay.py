from skimage import io
import numpy as np
from matplotlib import pyplot
from copy import deepcopy


def cut(img):
    return img[440:780, 150:490, :]


def yuv2rgb(pix):
    Y = pix[0]
    Cb = pix[1]
    Cr = pix[2]
    R = Y + 1.402*(Cr - 128)
    G = Y - 0.34414*(Cb - 128) - 0.71414*(Cr - 128)
    B = Y + 1.772*(Cb - 128)
    return np.array([R, G, B])


def rgb2yuv(pix):
    R = pix[0]
    G = pix[1]
    B = pix[2]
    Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
    Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128
    return np.array([Y, Cb, Cr])


def setPix(pix, r, g, b):
    pix[0] = r
    pix[1] = g
    pix[2] = b


def hist(img):
    rows = np.zeros(np.size(img, 0))
    cols = np.zeros(np.size(img, 1))
    # 17 25 44
    for row in range(0, np.size(img, 0)):
        r = img[row, :, 0]
        b = img[row, :, 2]
        rows[row] = sum((r <= 20) & (r >= 15) & (b > 40))

    for col in range(0, np.size(img, 1)):
        r = img[:, col, 0]
        b = img[:, col, 2]
        cols[col] = sum((r <= 20) & (r >= 15) & (b > 40))

    img2 = deepcopy(img)
    for i, r in enumerate(rows):
        for c in range(0, int(r)):
            img2[i][c][0] = 255
            img2[i][c][1] = 0
            img2[i][c][2] = 0

    for i, c in enumerate(cols):
        for r in range(0, int(c)):
            img2[r][i][0] = 0
            img2[r][i][1] = 255
            img2[r][i][2] = 0

    return img2


def main():
    imgorgn = cut(io.imread("C:\\users\\Admin\\desktop\\3.jpg"))
    img = deepcopy(imgorgn)

    selected = np.zeros(np.size(img, 0))

    for row in range(0, np.size(img, 0)):
        for col in range(0, np.size(img, 1)):
            pix = rgb2yuv(img[row][col])
            if pix[0] < 100:
                selected[row] += 1

    colnum = np.size(img, 1)
    selected = selected == colnum

    row = 0
    while row < len(selected):
        if selected[row]:
            r = row
            while r < len(selected) and selected[r]:
                r += 1
            img[row:r, :, :] = imgorgn[row-(r-row):row, :, :]
            row = r
        row += 1
    pyplot.subplot(121)
    io.imshow(imgorgn)
    pyplot.subplot(122)
    io.imshow(img)
    pyplot.show()


if __name__ == '__main__':
    main()
