import cv2
import numpy as np
import pyautogui
from PIL import Image

def capture_desktop():
    """捕获整个桌面截图，返回OpenCV BGR格式图像"""
    pil_img = pyautogui.screenshot()
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_cv

def cv2_to_pil(img_cv):
    """BGR OpenCV图像转PIL图像"""
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def resize_cv_image(pil_img, w, h):
    """缩放PIL图像，使用LANCZOS高质量缩放"""
    return pil_img.resize((w, h), Image.Resampling.LANCZOS)
