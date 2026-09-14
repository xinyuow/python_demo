import os
import cv2
import numpy as np


def object_recognition(object_img, result_img_path):
    """
    基于轮廓的物体识别、计数与标记
    :param object_img: 物体图片
    :param result_img_path: 物体识别结果保存路径
    :return:
    """
    img_copy = np.asarray(object_img).copy()

    # 高斯滤波降噪，消除画面细小噪点
    img_copy = cv2.GaussianBlur(img_copy, (3, 3), 0)

    # 转换为灰度图
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # 反向二值化，物体像素变黑，背景像素变白
    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 借助形态学开运算，消除画面细碎噪点，孤立小点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # 提取最外层外部轮廓，物理物体内部小孔嵌套轮廓
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 有效物体数量
    object_count = 0
    # 最大物体轮廓面积
    max_area = 0
    # 最大物体轮廓
    max_contour = None

    # 遍历全部轮廓，逐一对物体进行处理
    for cnt in contours:
        # 计算当前轮廓包围面积
        area = cv2.contourArea(cnt)
        # 过滤极小噪点面积
        if area < 100:
            continue
        object_count += 1
        # 获取轮廓外接矩形左上角坐标、宽度、高度
        x, y, w, h = cv2.boundingRect(cnt)
        # 绘制蓝色的物体矩形框
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 计算物体中心坐标
        center_x = (x + w // 2)
        center_y = (y + h // 2)
        # 黄色实心圆点标记物体中心
        cv2.circle(img_copy, (center_x, center_y), 5, (0, 255, 255), -1)
        # 打印物体信息
        print(f"第{object_count}个物体，面积：{area}，中心点：({center_x}, {center_y})")
        # 更新最大物体轮廓面积
        if area > max_area:
            max_area = area
            max_contour = cnt

    # 高亮标记最大物体轮廓，使用文字+红色粗框
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(img_copy, "Max object", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    print(f"共检测到{object_count}个有效物体，最大物体轮廓面积：{max_area}")
    # 保存结果
    write_result = cv2.imwrite(result_img_path, img_copy)
    if not write_result:
        raise IOError(f"保存物体识别结果图片失败：{result_img_path}")
    print("物体识别完成，文件已保存！")
    return True, img_copy


# 定义图片所在路径
image_path = './images/'
# 物体图片名称
object_img_filename = 'objects.png'
# 物体识别结果图片名称
object_recognition_result_img_filename = 'object_recognition_result.png'
# 拼接图片路径
object_img_path = os.path.join(image_path, object_img_filename)
object_recognition_result_img_path = os.path.join(image_path, object_recognition_result_img_filename)

# 读取物体图片
objects = cv2.imread(object_img_path)
if objects is None:
    raise FileNotFoundError(f"读取物体图片失败：{object_img_path}， 请检查图片路径是否正确")

# 调用物体轮廓识别
result_flag, img = object_recognition(objects, object_recognition_result_img_path)
if result_flag:
    if img is not None:
        # 展示图片
        cv2.imshow('object recognition', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()