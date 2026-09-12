import cv2

# 1.读取原始彩色图片
img = cv2.imread('images/lake.jpg')

if img is None:
    print("读取图片失败，请检查图片路径是否正确")
    exit()

# 2.彩色图片转换为灰度图，减少计算量
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3.进行滤波降噪，采用双边滤波
gray_img = cv2.bilateralFilter(gray_img, 9, 75, 75)

# 4.图片二值化处理，采用自适应阈值处理
gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# 5-1.形态学优化，定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# 5-2.形态学优化，进行开运算
gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)

# 6.边缘检测，采用Canny边缘检测
gray_img = cv2.Canny(gray_img, 50, 150)

# 7.轮廓查找。只查找外层轮廓，并只存储关键轮廓点
contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 8.复制原图作为绘制画布。保留原图不被修改
draw_img = img.copy()

# 9.筛选轮廓，只绘制有效轮廓
for contour in contours:
    # 计算轮廓周长
    perimeter = cv2.arcLength(contour, False)
    if perimeter > 200:
        # 10.绘制有效轮廓，颜色使用红色，线宽为2像素
        cv2.drawContours(draw_img, [contour], -1, (0, 0, 255), 2)

# 11.展示处理结果图片
cv2.imshow('processed image', draw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()