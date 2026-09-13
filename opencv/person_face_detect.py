import os
import cv2
import numpy as np


def face_detect(crowd_img, crowd_face_detect_path):
    """
    人脸检测 + 绘制矩形框
    :param crowd_img: 人像图片
    :param crowd_face_detect_path: 人脸检测结果保存路径
    :return:
    """
    img_copy = np.asarray(crowd_img).copy()

    # 转换为灰度图，因为人脸检测仅支持灰度图
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # 加载OpenCV内置人脸(正面)分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # 执行人脸多尺度检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=25)

    # 遍历所有人脸，并绘制红色矩形框
    for (x, y, w, h) in faces:
        # 绘制矩形框。五个参数：原图，矩形框左上角坐标，矩形框右下角坐标，矩形框颜色，矩形框线宽
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 打印单张人脸坐标信息
        print(f"人脸坐标：({x}, {y})、宽度：{w}、高度：{h}")

    # 统计人脸数量
    print(f"共检测到{len(faces)}张人脸")
    # 保存结果
    write_result = cv2.imwrite(crowd_face_detect_path, img_copy)
    if not write_result:
        raise IOError(f"保存人脸检测结果图片失败：{crowd_face_detect_path}")
    print("人脸检测完成，文件已保存！")
    return True, img_copy


def face_and_eye_detect(crowd_img, crowd_face_detect_path):
    """
    人脸、双眼同时检测 + 绘制矩形框
    :param crowd_img: 人像图片
    :param crowd_face_detect_path: 人脸检测结果保存路径
    :return:
    """
    img_copy = np.asarray(crowd_img).copy()

    # 转换为灰度图，因为人脸检测仅支持灰度图
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # 加载OpenCV内置人脸和双眼(正面)分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # 执行人脸多尺度检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=25)

    # 遍历所有人脸
    for (x, y, w, h) in faces:
        # 绘制矩形框。五个参数：原图，矩形框左上角坐标，矩形框右下角坐标，矩形框颜色，矩形框线宽
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 从人脸区域提取眼睛区域
        eyes = eye_cascade.detectMultiScale(gray[y:y + h, x:x + w], scaleFactor=1.1, minNeighbors=5)
        # 绘制眼睛矩形框
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img_copy, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
        # 打印单张人脸坐标信息
        print(f"人脸坐标：({x}, {y})、宽度：{w}、高度：{h}")

    # 统计人脸数量
    print(f"共检测到{len(faces)}张人脸 + 双眼")
    # 保存结果
    write_result = cv2.imwrite(crowd_face_detect_path, img_copy)
    if not write_result:
        raise IOError(f"保存人脸+双眼检测结果图片失败：{crowd_face_detect_path}")
    print("人脸检测完成，文件已保存！")
    return True, img_copy


# 定义图片所在路径
image_path = './images/'
# 人像图片名称
person_img_filename = 'crowd.jpg'
# 人脸识别结果图片名称
person_face_detect_img_filename = 'crowd_face_detect.jpg'
# 人脸+双眼检测结果图片名称
person_face_and_eye_detect_img_filename = 'crowd_face_and_eye_detect.jpg'
# 拼接图片路径
person_img_path = os.path.join(image_path, person_img_filename)
person_face_detect_img_path = os.path.join(image_path, person_face_detect_img_filename)
person_face_and_eye_detect_img_path = os.path.join(image_path, person_face_and_eye_detect_img_filename)

# 读取人像图片
crowd = cv2.imread(person_img_path)
if crowd is None:
    raise FileNotFoundError(f"读取人像图片失败：{person_img_path}， 请检查图片路径是否正确")

# 调用人脸检测
result_flag, img = face_detect(crowd, person_face_detect_img_path)
if result_flag:
    if img is not None:
        # 展示图片
        cv2.imshow('face detect', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 调用人脸+双眼检测
result_flag, img = face_and_eye_detect(crowd, person_face_and_eye_detect_img_path)
if result_flag:
    if img is not None:
        # 展示图片
        cv2.imshow('face and eye detect', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()