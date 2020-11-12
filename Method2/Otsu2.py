# for reading and writing images
import cv2
import numpy as np

# convert the color image into a gray-scale image using the formula
def rgb2gray(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    gray_img = (0.299 * r + 0.587 * g + 0.114 * b).round()
    gray_img = gray_img.astype(np.uint8)  # unit8 range[0,255]

    return gray_img


def hist_img(image):
    gray_hist = {}  # key- grayscale fall in (0,255) value - #pixel
    for key in range(256):
        gray_hist[key] = gray_hist.get(key, 0)
    for i in range(0, image.shape[0]):  # height
        for j in range(0, image.shape[1]):  # width
            curpix_grayscale = image[i][j]
            gray_hist[curpix_grayscale] = gray_hist.get(curpix_grayscale, 0) + 1
    return gray_hist


def dict_mult(dict, i, j):
    res = 0
    for key in range(i, j):
        res += key * dict[key]
    return res


def dict_add(dict, i, j):
    res = 0
    for key in range(i, j):
        res += dict[key]
    return res


def otsu(hist, image):
    # compute the total number of pixels in the image
    pixels = image.shape[0] * image.shape[1]
    totalweight = 0
    t1, t2, t3 = 0, 0, 0
    maxval = 0

    for i in range(256):
        totalweight += i * hist[i]
    # average grayscale over
    mu = totalweight / pixels

    for i in range(0, 256):
        for j in range(i, 256):
            for k in range(j, 256):
                # compute the number of pixels in each region
                pa = dict_add(hist, 0, i+1)
                pb = dict_add(hist, i+1, j+1)
                pc = dict_add(hist, j+1, k+1)
                pd = dict_add(hist, k+1, 256)

                # compute the percentage that each region takes of the whole image
                wa = pa / pixels
                wb = pb / pixels
                wc = pc / pixels
                wd = pd / pixels

                # the average gray scale in each region and in the whole image
                mua = dict_mult(hist, 0, i+1) / float(pa) if pa > 0 else 0
                mub = dict_mult(hist, i+1, j+1) / float(pb) if pb > 0 else 0
                muc = dict_mult(hist, j+1, k+1) / float(pc) if pc > 0 else 0
                mud = dict_mult(hist, k+1, 256) / float(pd) if pd > 0 else 0

                v = wa * (mua - mu) ** 2 + wb * (mub - mu) ** 2 + wc * (muc - mu) ** 2 + wd * (mud - mu) ** 2

                if v > maxval:
                    t1, t2, t3 = i, j, k
                    maxval = v

    # print the three thresholds
    print('For image: {}'.format(name))
    print(f'--- First threshold: {t1} \n')
    print(f'--- Second threshold: {t2} \n')
    print(f'--- Third threshold: {t3} \n')
    # since we have three thresholds, we separate image into 3 regions
    img = image.copy()
    img[img <= t1] = 0
    img[(t1 < img) & (img <= t2)] = 85
    img[(t2 < img) & (img <= t3)] = 170
    img[t3 < img] = 255
    return img


for name in ['data13', 'fruits2b', 'tiger1-24bits']:
    # read image from the certain file
    img = cv2.imread(r'{}.bmp'.format(name)).astype(np.float32)

    # convert to grayscale
    grayscale = rgb2gray(img)

    # create 256 bins histogram
    hist = hist_img(grayscale)

    # apply the otsu thresholding
    img = otsu(hist, grayscale)

    # write each result image to a new bmp file and show the results, using openCV
    cv2.imwrite('{}_out.bmp'.format(name), img)

