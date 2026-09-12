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
    # 读取蓝底证件照
    img = cv2.imread(id_blue_photo_path)
    if img is None:
        raise FileNotFoundError(f"读取蓝底证件照失败：{id_blue_photo_path}， 请检查图片路径是否正确")

    # 复制原图作为绘制画布。保留原图不被修改
    img_copy = np.asarray(img).copy()

    # 双边降噪，消除背景细碎噪点
    blur_img = cv2.bilateralFilter(img_copy, 9, 75, 75)

    # 将BGR格式转换为HSV格式，因为色彩空间提取必须使用HSV格式的图片
    hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)

    # 设置蓝底标准 HSV 上下界阈值
    lower_blue = np.array([90, 43, 46])
    upper_blue = np.array([130, 255, 255])

    # 生成背景掩码：蓝色区域为白色，其余区域为黑色
    mask_bg = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # 将掩码对应的背景区域，统一填充为纯白色
    img_copy[mask_bg > 0] = [255, 255, 255]

    # 保存处理结果图片
    write_result = cv2.imwrite(id_white_photo_path, img_copy)
    if not write_result:
        raise IOError(f"保存白底证件照失败：{id_white_photo_path}")
    print("证件照：蓝底转白底成功，文件已保存！")
    return True

def id_photo_background_red_to_white(id_red_photo_path, id_white_photo_path):
    """
    证件照：红底转白底
    :param id_red_photo_path: 红底证件照路径
    :param id_white_photo_path: 白底证件照路径
    :return: bool 是否成功
    """
    # 读取红底证件照
    img = cv2.imread(id_red_photo_path)
    if img is None:
        raise FileNotFoundError(f"读取红底证件照失败：{id_red_photo_path}， 请检查图片路径是否正确")
    assert img is not None

    # 复制原图作为绘制画布。保留原图不被修改
    img_copy = np.asarray(img).copy()

    # 双边降噪，消除背景细碎噪点。实际测试下来，双边降噪比高斯降噪效果更好
    blur_img = cv2.bilateralFilter(img_copy, 9, 75, 75)

    # 将BGR格式转换为HSV格式，因为色彩空间提取必须使用HSV格式的图片
    hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)

    # 设置两段红色区域的HSV值范围
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 生成两段背景掩码：红色区域为白色，其余区域为黑色
    mask_bg1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask_bg2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    # 合并两段红色背景，按位或
    mask = cv2.bitwise_or(mask_bg1, mask_bg2)

    # 形态学优化。生成的背景掩码可能会有噪点或孔洞，所以要进行形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 闭运算：填充背景区域内部细小孔洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 开运算：清除背景区域零散红色噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 目前的mask是背景区域已经从红色转换为白色，人像区域为黑色
    # 但是在黑色区域可能会有白色噪点，所以要进行轮廓检测，只保留最大连通区域（即白色背景区域）

    # 查找所有最外层轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 找到面积最大的外部轮廓，即最大的白色区域
        largest_contour = max(contours, key=cv2.contourArea)
        # 创建一个全黑的数组，和mask大小相同
        mask_clean = np.zeros_like(mask)
        # 在这个黑色数组下，绘制刚得到的最大白色区域轮廓。这个轮廓就是最终要保留的白色背景区域
        cv2.drawContours(mask_clean, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        # 替换mask。现在的mask是背景区域是白色，人像区域是黑色
        mask = mask_clean

    # 目前mask是背景区域是白色，人像区域是黑色

    # 掩码按位取反：人像区域变白色，背景区域变黑色
    mask_inv = cv2.bitwise_not(mask)
    # 提取出人像区域。(mask_inv中，人像区域为255，允许通过；背景区域为0，禁止通过)
    mask_person = cv2.bitwise_and(img_copy, img_copy, mask=mask_inv)

    # 构建一个纯白色背景图层
    white_bg = np.full_like(img, 255)
    # 提取出背景区域。(mask中，背景区域为255，允许通过；人像区域为0，禁止通过)
    background = cv2.bitwise_and(white_bg, white_bg, mask=mask)

    # 人像和背景合并
    img_copy = cv2.add(mask_person, background)

    # 保存处理结果图片
    write_result = cv2.imwrite(id_white_photo_path, img_copy)
    if not write_result:
        raise IOError(f"保存白底证件照失败：{id_white_photo_path}")
    print("证件照：红底转白底成功，文件已保存！")
    return True


# 定义图片所在路径
image_path = './images/'
# 蓝底证件照文件名称
id_blue_filename = 'id_blue.jpg'
# 蓝底转白底证件照文件名称
id_white_filename = 'id_blue_to_white.jpg'
# 蓝底证件照路径
id_blue_path = os.path.join(image_path, id_blue_filename)
# 蓝底转白底证件照路径
id_blue_to_white_path = os.path.join(image_path, id_white_filename)

# 调用转换函数
blue_to_white_flag = id_photo_background_blue_to_white(id_blue_path, id_blue_to_white_path)
if blue_to_white_flag:
    blue_to_white_flag = cv2.imread(id_blue_to_white_path)
    if blue_to_white_flag is not None:
        # 展示白底证件照图片
        cv2.imshow('id blue to white img', blue_to_white_flag)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# 红底证件照文件名称
id_red_filename = 'id_red.jpeg'
# 红底转白底证件照文件名称
id_red_white_filename = 'id_red_to_white.jpg'
# 红底证件照路径
id_red_path = os.path.join(image_path, id_red_filename)
# 红底转白底证件照路径
id_red_to_white_path = os.path.join(image_path, id_red_white_filename)

# 调用转换函数
red_to_white_flag = id_photo_background_red_to_white(id_red_path, id_red_to_white_path)
if red_to_white_flag:
    red_to_white_flag = cv2.imread(id_red_to_white_path)
    if red_to_white_flag is not None:
        # 展示白底证件照图片
        cv2.imshow('id red to white img', red_to_white_flag)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
