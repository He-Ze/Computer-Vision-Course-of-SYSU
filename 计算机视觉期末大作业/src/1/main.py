import cv2
import numpy as np
import imageio

# 计算梯度能量
def energy(img):
    gaussian_blur = cv2.GaussianBlur(img, (3, 3), 0, 0)
    gray = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT,)
    y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT,)
    return cv2.add(np.absolute(x), np.absolute(y))


# 计算水平方向能量图
def cal_energy_horizon(energy,img_gt):
    (height, width) = energy.shape[:2]
    energy_map = np.zeros((height, width))
    for j in range(1, width):
        for i in range(height):
            top = (energy_map[i - 1, j - 1] if i - 1 >= 0 else 1e6)
            middle = energy_map[i, j - 1]
            bottom = (energy_map[i + 1, j - 1] if i + 1 < height else 1e6)
            energy_map[i, j] = energy[i, j] + min(top, middle, bottom)
    return energy_map


# 计算垂直方向能量图
def cal_energy_vertical(energy,img_gt):
    (height, width) = energy.shape[:2]
    energy_map = np.zeros((height, width))
    for i in range(1, height):
        for j in range(width):
            left = (energy_map[i - 1, j - 1] if j - 1 >= 0 else 1e6)
            middle = energy_map[i - 1, j]
            right = (energy_map[i - 1, j + 1] if j + 1 < width else 1e6)
            energy_map[i, j] = energy[i, j] + min(left, middle, right)
    return energy_map


# 寻找水平方向最小seam线
def findseam_horizon(img,energy_map):
    height, width = energy_map.shape[0], energy_map.shape[1]
    before, seam = 0, []
    for i in range(width - 1, -1, -1):
        col = energy_map[:, i]
        if i == width - 1:
            before = np.argmin(col)
        else:
            top = (col[before - 1] if before - 1 >= 0 else 1e6)
            middle = col[before]
            bottom = (col[before + 1] if before + 1 < height else 1e6)
            before += np.argmin([top, middle, bottom]) - 1
        seam.append([i, before])
    return seam


# 寻找垂直方向最小seam线
def findseam_vertical(img,energy_map):
    height, width = energy_map.shape[0], energy_map.shape[1]
    before, seam = 0, []
    for i in range(height - 1, -1, -1):
        row = energy_map[i, :]
        if i == height - 1:
            before = np.argmin(row)
        else:
            left = (row[before - 1] if before - 1 >= 0 else 1e6)
            middle = row[before]
            right = (row[before + 1] if before + 1 < width else 1e6)
            before += np.argmin([left, middle, right]) - 1
        seam.append([before, i])
    return seam


# 移除水平方向最小seam线
def removeseam_horizon(img, seam):
    (height, width, depth) = img.shape
    removed = np.zeros((height - 1, width, depth), np.uint8)
    for (x, y) in reversed(seam):
        removed[0:y, x] = img[0:y, x]
        removed[y:height - 1, x] = img[y + 1:height, x]
    return removed


# 移除垂直方向最小seam线
def removeseam_vertical(img, seam):
    (height, width, depth) = img.shape
    removed = np.zeros((height, width - 1, depth), np.uint8)
    for (x, y) in reversed(seam):
        removed[y, 0:x] = img[y, 0:x]
        removed[y, x:width - 1] = img[y, x + 1:width]
    return removed


# 实时显示过程并将这一帧存入list方便之后生成GIF
def plot(img, seam,image_list):
    cv2.polylines(img, np.int32([np.asarray(seam)]), False, (0, 255, 0))
    image_list.append(img)
    cv2.imshow('seam', img)
    cv2.waitKey(1)


if __name__ == "__main__":
    #读取哪张图片在下面更改即可
    img_number="752"
    # 读取相应编号原图和前景图
    img = cv2.imread("../../data/imgs/"+img_number+".png")
    img_gt = cv2.imread("../../data/gt/"+img_number+".png")
    # 原图像长宽
    img_height, img_width = img.shape[0], img.shape[1]
    # 缩放比例
    ratio = 0.55
    # 缩放后的长宽
    width = int(ratio * img_width)
    height = int(img_height * ratio)
    
    cv2.namedWindow('seam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('seam', 500, 500)
    # 先在当前目录生成原图和前景图
    cv2.imwrite(img_number+"_origin.png", img)
    cv2.imwrite(img_number+"_gt.png", img_gt)
    # 计算在水平方向和垂直方向各需要删除多少像素
    x = img_width - width if img_width > width else 0
    y = img_height - height if img_height > height else 0
    
    # 用于存放生成GIF的各帧图片
    image_list=[]
    # 水平方向
    for i in range(y):
        energy_map = cal_energy_horizon(energy(img),img_gt)
        seam = findseam_horizon(img,energy_map)
        plot(img, seam,image_list)
        img = removeseam_horizon(img, seam)
    # 垂直方向
    for i in range(x):
        energy_map = cal_energy_vertical(energy(img),img_gt)
        seam = findseam_vertical(img,energy_map)
        plot(img, seam,image_list)
        img = removeseam_vertical(img, seam)
    
    # 生成结果图
    cv2.imwrite(img_number+'_result.png', img)
    cv2.imshow('seam', img)
    # 生成GIF
    imageio.mimsave(img_number+'.gif', image_list, 'GIF', duration=0.1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()