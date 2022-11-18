import numpy as np
import cv2
import os
import math
save_path = r'D:\nextvpu\yanye\code\2_analyse\save_pic\\'
#打开文件
path = r"D:\nextvpu\yanye\code\2_analyse\dataset\\"
dirs = os.listdir(path)
#print(dirs)#输出所有子文件和文件夹
#数据增强后的图片保存路径
for file in dirs:
    pic_dir = os.path.join(path, file) #images中子文件夹的路径
    print(pic_dir)
# for info in os.listdir(pic_dir):
    # info1 = os.path.join(pic_dir, info)  # 将路径与文件名结合起来就是每个文件的完整路径
    img = cv2.imread(pic_dir)
    # cv2.waitKey(1000)
    #镜像
    # 水平镜像
    h_flip = cv2.flip(img, 1)#1 左右翻转
    # cv2.imshow("Flipped Horizontally", h_flip)
    cv2.imwrite(save_path + '_h_flip.jpg', h_flip)
    # 垂直镜像
    v_flip = cv2.flip(img, 0) #0 上下翻转
    # cv2.imshow("Flipped Vertically", v_flip)
    cv2.imwrite(save_path + '_v_flip.jpg', v_flip)
    # 水平垂直镜像
    hv_flip = cv2.flip(img, -1)
    # cv2.imshow("Flipped Horizontally & Vertically", hv_flip)
    cv2.imwrite(save_path +'hv_flip.jpg', hv_flip)
    #平移
    # 平移矩阵[[1,0,-100],[0,1,-12]]
    #M = np.array([[1, 0, -100], [0, 1, -12]], dtype=np.float32)
    #img_change = cv2.warpAffine(img, M, (300, 300))
    #cv2.imshow("translation", img_change)
    #cv2.imwrite(save_path+str(info)+'img_change.jpg', img_change)
    #旋转
    # 90度旋转
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst_90 = cv2.warpAffine(img, M, (cols, rows))
    # cv2.imshow("dst_90", dst_90)
    cv2.imwrite(save_path +'dst_90.jpg', dst_90)
    # 70度旋转
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 70, 1)
    dst_70 = cv2.warpAffine(img, M, (cols, rows))
    # cv2.imshow("dst_70", dst_70)
    cv2.imwrite(save_path+ 'dst_70.jpg', dst_70)
    # 60度旋转
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 60, 1)
    dst_60 = cv2.warpAffine(img, M, (cols, rows))
    # cv2.imshow("dst_60", dst_60)
    cv2.imwrite(save_path +'dst_60.jpg', dst_60)
    # 50度旋转
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 50, 1)
    dst_50 = cv2.warpAffine(img, M, (cols, rows))
    # cv2.imshow("dst_50", dst_50)
    cv2.imwrite(save_path +'dst_50.jpg', dst_50)
    # 45度旋转
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    dst_45 = cv2.warpAffine(img, M, (cols, rows))
    # cv2.imshow("dst_45", dst_45)
    cv2.imwrite(save_path +'dst_45.jpg', dst_45)
    # 40度旋转
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 40, 1)
    dst_40 = cv2.warpAffine(img, M, (cols, rows))
    # cv2.imshow("dst_40", dst_40)
    cv2.imwrite(save_path +'dst_40.jpg', dst_40)
    # 30度旋转
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
    dst_30 = cv2.warpAffine(img, M, (cols, rows))
    # cv2.imshow("dst_30", dst_30)
    cv2.imwrite(save_path +'dst_30.jpg', dst_30)
    # 20度旋转
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 20, 1)
    dst_20 = cv2.warpAffine(img, M, (cols, rows))
    # cv2.imshow("dst_20", dst_20)
    cv2.imwrite(save_path +'dst_20.jpg', dst_20)
    # 缩放
    height, width = img.shape[:2]
    res = cv2.resize(img, (2 * width, 2 * height))
    # cv2.imshow("large", res)
    cv2.imwrite(save_path +'res.jpg', res)
    # 图像平移——仿射变换
    # 对图像进行变换（三点得到一个变换矩阵）
    # 我们知道三点确定一个平面，我们也可以通过确定三个点的关系来得到转换矩阵
    # 然后再通过warpAffine来进行变换
    point1 = np.float32([[50, 50], [300, 50], [50, 200]])
    point2 = np.float32([[10, 100], [300, 50], [100, 250]])
    M = cv2.getAffineTransform(point1, point2)
    dst1 = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))
    # cv2.imshow("affine transformation", dst1)
    # cv2.imshow("affine transformation", dst1)
    cv2.imwrite(save_path +'dst1.jpg', dst1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



