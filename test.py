# import numpy as np
# import cv2
#
#
def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def MSR(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)

    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img,dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R,None,0,255,cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

#
# if __name__ == '__main__':
#     #img = r'C:\yu_project\OpencvPython\Resources\P55.png'
#     img = r'C:\yu_project\OpencvPython\hxq_30.png'
#     scales = [15,101,301]  # [3,5,9]  #看不出效果有什么差别
#     src_img = cv2.imread(img)
#     b_gray, g_gray, r_gray = cv2.split(src_img)
#     b_gray = MSR(b_gray, scales)
#     g_gray = MSR(g_gray, scales)
#     r_gray = MSR(r_gray, scales)
#     result = cv2.merge([b_gray, g_gray, r_gray])
#
#     cv2.imshow('img',src_img)
#     cv2.imshow('MSR_result',result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     #img = r'C:\yu_project\OpencvPython\Resources\P55.png'
#     # img = r'C:\yu_project\OpencvPython\hxq_30.png'
#     # size = 3
#     # src_img = cv2.imread(img)
#     # b_gray, g_gray, r_gray = cv2.split(src_img)
#     # b_gray = SSR(b_gray, size)
#     # g_gray = SSR(g_gray, size)
#     # r_gray = SSR(r_gray, size)
#     # result = cv2.merge([b_gray, g_gray, r_gray])
#     #
#     # cv2.imshow('img', src_img)
#     # cv2.imshow('result', result)
#     # cv2.waitKey(0)
import cv2
import numpy as np

def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst

if __name__ == '__main__':
    #file = r'C:\yu_project\OpencvPython\Resources\1.bmp'
    file = r'C:\yu_project\OpencvPython\Resources\P55.png'
    blockSize = 16
    src_img = cv2.imread(file)
    img1 = src_img.copy()
    if img1.shape[1] > 800:
        img1 = cv2.resize(img1, (800, int(800 * img1.shape[0] / img1.shape[1])))
    if img1.shape[0] > 800:
        img1 = cv2.resize(img1, (int(800 * img1.shape[1] / img1.shape[0]), 800))
    print(img1.shape)
    # img1 = img1[200:400,300:700]
    size = 5
    b_gray, g_gray, r_gray = cv2.split(img1)
    b_gray = SSR(b_gray, size)
    g_gray = SSR(g_gray, size)
    r_gray = SSR(r_gray, size)
    light_prv = cv2.merge([b_gray, g_gray, r_gray])
    blur = cv2.GaussianBlur(light_prv,(3,3),0)
    #blur = cv2.GaussianBlur(img1,(3,3),0)
    # dst = unevenLightCompensate(light_prv, blockSize)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    #res = cv2.adaptiveThreshold(res1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 5)
    retval = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, 1))
    open = cv2.morphologyEx(th, cv2.MORPH_OPEN, retval)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(open, connectivity=8)
    print(stats)
    mser = cv2.MSER_create()
    regions, boxes = mser.detectRegions(open)
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 0), 2)
    imgstack = stackImages(0.4,[[src_img,blur,gray],[th,open,img1]])
    cv2.imshow('result', imgstack)
    cv2.waitKey(0)

