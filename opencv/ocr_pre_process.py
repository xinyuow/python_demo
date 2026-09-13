import os
import cv2
import numpy as np


def ocr_pre_process(img_path, ocr_img_path):
    """
    图片OCR预处理函数
    :param img_path: 文字图片路径
    :param ocr_img_path: OCR预处理图片路径
    :return: bool 是否成功
    """
    # 读取文字图片
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"读取文档图片失败：{img}， 请检查图片路径是否正确")
    img_copy = np.asarray(img).copy()

    # 高斯滤波降噪，去除画面颗粒噪点
    img_copy = cv2.GaussianBlur(img_copy, (3, 3), 0)

    # 转换为灰度图
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # 图像锐化处理，强化文字边缘
    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]
    )
    img_copy = cv2.filter2D(img_copy, -1, sharpen_kernel)

    # 全局二值化处理，将灰度图转换为二值图
    _, binary_img = cv2.threshold(img_copy, 127, 255, cv2.THRESH_BINARY_INV)

    # 保存结果
    write_result = cv2.imwrite(ocr_img_path, binary_img)
    if not write_result:
        raise IOError(f"保存OCR预处理图片失败：{ocr_img_path}")
    print("图片OCR预处理完成，文件已保存！")
    return True


# 定义图片所在路径
image_path = './images/'
# 文字图片名称
text_img_filename = 'text.jpg'
# OCR预处理图片名称
ocr_text_img_filename = 'ocr_text.jpg'
# 拼接图片路径
text_img_path = os.path.join(image_path, text_img_filename)
ocr_text_img_path = os.path.join(image_path, ocr_text_img_filename)

# 调用OCR预处理函数
result_flag = ocr_pre_process(text_img_path, ocr_text_img_path)
if result_flag:
    ocr_text_img = cv2.imread(ocr_text_img_path)
    if ocr_text_img is not None:
        # 展示图片
        cv2.imshow('ocr text img', ocr_text_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
