import cv2
# only used in reading images
import numpy as np

def grayscale(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    gray_img = (0.299 * r + 0.587 * g + 0.114 * b).round()
    return gray_img


def hist(image):
    gray_hist = {}
    for key in range(256):
        gray_hist[key] = gray_hist.get(key, 0)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            curpix_grayscale = image[i][j]
            gray_hist[curpix_grayscale] = gray_hist.get(curpix_grayscale, 0) + 1
    return gray_hist

def otsu(hist, img_array):
    height = img_array.shape[0]
    width = img_array.shape[1]

    total_pixel = height * width

    min_variance = float(inf)
    t1 = t2 =t3 = 0
    for thresold1 in range(0, 253):
        for thresold2 in range(thresold1 + 1, 254):
            for thresold3 in range(thresold2 + 1, 255):
                # number of pixels in each region
                pa = pb = pc = pd = 0
                # mu in each region which is the total_weight / # of pixels
                mu1, mu2, mu3, mu4 = 0, 0, 0, 0
                
                for a in range(0, thresold1 + 1):
                    pa += hist[a]
                    # calculate the total weight here
                    mu1 += a * hist[a]
                    
                for a in range(thresold1 + 1, thresold2 + 1):
                    pb += hist[a]
                    mu2 += a * hist[a]
                    
                for a in range(thresold2 + 1, thresold3 + 1):
                    pc += hist[a]
                    mu3 += a * hist[a]
                for a in range(thresold3 + 1, 256):
                    pd += hist[a]
                    mu4 += a * hist[a]
                
                if pa > 0:
                # divided by the pixel numbers in current region
                    mu1 /= pa
                if pb > 0:
                    mu2 /= pb
                if pc > 0:
                    mu3 /= pc
                if pd > 0:
                    mu4 /= pd
                    
                # mu_square = sum((Xi - mu) ** 2) / N for each region
                mua_square, mub_square, muc_square, mud_square = 0, 0, 0, 0
                for i in range(0, thresold1 + 1):
                    if pa > 0:
                        mua_square += (i - mu1) ** 2 * hist[i] / pa
                for i in range(thresold1 + 1, thresold2 + 1):
                    if pb > 0:
                        mub_square += (i - mu2) ** 2 * hist[i] / pb
                for i in range(thresold2 + 1, thresold3 + 1):
                    if pc > 0:
                        muc_square += (i - mu3) ** 2 * hist[i] / pc
                for i in range(thresold3 + 1, 256):
                    if pd > 0:
                        mud_square += (i - mu4) ** 2 * hist[i] / pd
                    
                w1, w2, w3, w4 = pa / total_pixel, pb / total_pixel, pc / total_pixel, pd / total_pixel
                # V = sum(w * mu_square)
                tmp_var = w1 * mua_square + w2 * mub_square + w3 * muc_square + w4 * mud_square
                # find the minimum variance
                if tmp_var < min_variance:
                    t1, t2, t3 = thresold1, thresold2, thresold3
                    min_variance = tmp_var

    img = img_array.copy()
    # print the three thresholds
    print(f'First threshold: {t1} \n')
    print(f'Second threshold: {t2} \n')
    print(f'Third threshold: {t3} \n')
    # generate different gray-scale according to three thresholds
    img[img_array <= t1] = 0
    img[(t1 < img_array) & (img_array <= t2)] = 85
    img[(t2 < img_array) & (img_array <= t3)] = 170
    img[t3 < img_array] = 255
    return img
    
name = '{}'.format('tiger1')
img = cv2.imread(r'{}.bmp'.format(name)).astype(np.float32)

# convert to grayscale
grayscale = grayscale(img)

# create histogram
hist = hist(grayscale)

# apply the otsu thresholding
img = otsu(hist, grayscale)

# write each result image to a new bmp file and show the results, using openCV
cv2.imwrite('{}_output.bmp'.format(name),img)

