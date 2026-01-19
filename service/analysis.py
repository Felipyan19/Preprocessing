import cv2
import numpy as np


def analyze_image(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255])),
    )
    red_pct = np.sum(mask_red > 0) / mask_red.size * 100

    mask_blue = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
    blue_pct = np.sum(mask_blue > 0) / mask_blue.size * 100

    recommendations = []

    if img.shape[0] < 500 or img.shape[1] < 500:
        recommendations.append('upscale')
    if red_pct > 10 or blue_pct > 10:
        recommendations.append('remove_color_bg')
    if np.mean(gray) < 100:
        recommendations.append('enhance_contrast')

    return {
        'size': {'w': img.shape[1], 'h': img.shape[0]},
        'brightness': float(np.mean(gray)),
        'red_percent': round(red_pct, 1),
        'blue_percent': round(blue_pct, 1),
        'recommendations': recommendations,
        'suggested_preset': 'table_ocr_aggressive' if len(recommendations) >= 2 else 'table_ocr',
    }
