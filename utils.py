# utils.py
import cv2
import numpy as np
import pyautogui
from PIL import Image


def capture_desktop():
    """截取全屏，返回 BGR cv 图像"""
    pil_img = pyautogui.screenshot()
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_cv


def cv2_to_pil(img_cv):
    """BGR cv → PIL RGB"""
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def resize_pil_image(pil_img, target_w, target_h):
    """PIL 图像高质量缩放"""
    return pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def safe_moments(mask):
    """安全计算图像 moments，防止 m00=0 除零；返回 None 代表无有效像素"""
    m = cv2.moments(mask)
    if abs(m["m00"]) < 1e-6:
        return None
    cy = m["m01"] / m["m00"]
    cx = m["m10"] / m["m00"]
    return {"cx": cx, "cy": cy, "moments": m}


def preprocess_mask(mask, blur_ksize=(3, 3)):
    """mask 降噪，形态学闭运算消除细碎噪点"""
    blur = cv2.GaussianBlur(mask, blur_ksize, 0)
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    return clean


def split_region_by_ratio(img, top_ratio, bottom_ratio):
    """按高度比例切分图片区域，做边界 clip 保护"""
    h, w = img.shape[:2]
    y1 = int(h * top_ratio)
    y2 = int(h * bottom_ratio)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, y1 + 1, h)
    return img[y1:y2, :]


def crop_width_ratio(img, width_ratio):
    """按宽度比例裁剪左侧区域，规避右侧盘口面板干扰；返回裁剪后的 BGR 图"""
    if width_ratio is None or width_ratio >= 1.0:
        return img
    h, w = img.shape[:2]
    x2 = int(w * width_ratio)
    x2 = np.clip(x2, 1, w)
    return img[:, :x2]
