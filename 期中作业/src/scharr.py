#!/usr/bin/env python3

import cv2 as cv
import numpy as np

def ManyImgs(scale, imgarray):
	rows = len(imgarray)         # 元组或者列表的长度
	cols = len(imgarray[0])      # 如果imgarray是列表，返回列表里第一幅图像的通道数，如果是元组，返回元组里包含的第一个列表的长度
	# print("rows=", rows, "cols=", cols)
	
	# 判断imgarray[0]的类型是否是list
	# 是list，表明imgarray是一个元组，需要垂直显示
	rowsAvailable = isinstance(imgarray[0], list)
	
	# 第一张图片的宽高
	width = imgarray[0][0].shape[1]
	height = imgarray[0][0].shape[0]
	# print("width=", width, "height=", height)
	
	# 如果传入的是一个元组
	if rowsAvailable:
		for x in range(0, rows):
			for y in range(0, cols):
				# 遍历元组，如果是第一幅图像，不做变换
				if imgarray[x][y].shape[:2] == imgarray[0][0].shape[:2]:
					imgarray[x][y] = cv.resize(imgarray[x][y], (0, 0), None, scale, scale)
				# 将其他矩阵变换为与第一幅图像相同大小，缩放比例为scale
				else:
					imgarray[x][y] = cv.resize(imgarray[x][y], (imgarray[0][0].shape[1], imgarray[0][0].shape[0]), None, scale, scale)
				# 如果图像是灰度图，将其转换成彩色显示
				if  len(imgarray[x][y].shape) == 2:
					imgarray[x][y] = cv.cvtColor(imgarray[x][y], cv.COLOR_GRAY2BGR)
					
		# 创建一个空白画布，与第一张图片大小相同
		imgBlank = np.zeros((height, width, 3), np.uint8)
		hor = [imgBlank] * rows   # 与第一张图片大小相同，与元组包含列表数相同的水平空白图像
		for x in range(0, rows):
			# 将元组里第x个列表水平排列
			hor[x] = np.hstack(imgarray[x])
		ver = np.vstack(hor)   # 将不同列表垂直拼接
	# 如果传入的是一个列表
	else:
		# 变换操作，与前面相同
		for x in range(0, rows):
			if imgarray[x].shape[:2] == imgarray[0].shape[:2]:
				imgarray[x] = cv.resize(imgarray[x], (0, 0), None, scale, scale)
			else:
				imgarray[x] = cv.resize(imgarray[x], (imgarray[0].shape[1], imgarray[0].shape[0]), None, scale, scale)
			if len(imgarray[x].shape) == 2:
				imgarray[x] = cv.cvtColor(imgarray[x], cv2.COLOR_GRAY2BGR)
		# 将列表水平排列
		hor = np.hstack(imgarray)
		ver = hor
	return ver

#Scharr算子(Sobel算子的增强版，效果更突出)

src = cv.imread('test1.JPG')
grad_x = cv.Scharr(src, cv.CV_32F, 1, 0)   #对x求一阶导
grad_y = cv.Scharr(src, cv.CV_32F, 0, 1)   #对y求一阶导
gradx = cv.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回原来的uint8形式
grady = cv.convertScaleAbs(grad_y)
gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)

src1 = cv.imread('test2.JPG')
grad_x1 = cv.Scharr(src1, cv.CV_32F, 1, 0)   #对x求一阶导
grad_y1 = cv.Scharr(src1, cv.CV_32F, 0, 1)   #对y求一阶导
gradx1 = cv.convertScaleAbs(grad_x1)  #用convertScaleAbs()函数将其转回原来的uint8形式
grady1 = cv.convertScaleAbs(grad_y1)
gradxy1 = cv.addWeighted(gradx1, 0.5, grady1, 0.5, 0)

result = ManyImgs(1, ([gradxy], [gradxy1]))
cv.imshow("result",result)

cv.waitKey(0)
cv.destroyAllWindows()