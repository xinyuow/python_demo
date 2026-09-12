import os
import cv2
import numpy as np

def id_photo_background_blue_to_white(id_blue_photo_path, id_white_photo_path):
    """
    证件照：蓝底转白底
    :param id_blue_photo_path: 蓝底证件照路径
    :param id_white_photo_path: 白底证件照路径
    :return: bool 是否成功
    """
    # 1.读取蓝底证件照
    img = cv2.imread(id_blue_photo_path)
    if img is None:
        raise FileNotFoundError(f"读取蓝底证件照失败：{id_blue_photo_path}， 请检查图片路径是否正确")

    # 2.复制原图作为绘制画布。保留原图不被修改
    img_copy = np.asarray(img).copy()

    # 3.双边降噪，消除背景细碎噪点
    blur_img = cv2.bilateralFilter(img_copy, 9, 75, 75)

    # 4.将BGR格式转换为HSV格式，因为色彩空间提取必须使用HSV格式的图片
    hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)

    # 5.设置蓝底标准 HSV 上下界阈值
    lower_blue = np.array([90, 43, 46])
    upper_blue = np.array([130, 255, 255])

    # 6.生成背景掩码：蓝色区域为白色，其余区域为黑色
    mask_bg = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # 7.将掩码对应的背景区域，统一填充为纯白色
    img_copy[mask_bg > 0] = [255, 255, 255]

    # 8.保存处理结果图片
    write_result = cv2.imwrite(id_white_photo_path, img_copy)
    if not write_result:
        raise IOError(f"保存白底证件照失败：{id_white_photo_path}")
    print("证件照：蓝底转白底成功，文件已保存！")
    return True


# 定义图片所在路径
image_path = './images/'
# 蓝底证件照文件名称
id_blue_filename = 'id_blue.jpg'
# 白底证件照文件名称
id_white_filename = 'id_white.jpg'
# 蓝底证件照路径
id_blue_path = os.path.join(image_path, id_blue_filename)
# 白底证件照路径
id_white_path = os.path.join(image_path, id_white_filename)

# 调用函数
convert_flag = id_photo_background_blue_to_white(id_blue_path, id_white_path)
if convert_flag:
    white_img = cv2.imread(id_white_path)
    if white_img is not None:
        # 展示白底证件照图片
        cv2.imshow('id white img', white_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
