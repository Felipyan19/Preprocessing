import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ============================================
# OPERACIONES DE PREPROCESAMIENTO
# ============================================

def enhance_contrast_clahe(img: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """CLAHE - muy efectivo para texto blanco sobre fondos de color."""
    if len(img.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def remove_color_background(img: np.ndarray) -> np.ndarray:
    """Elimina fondos de color preservando texto claro."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rojo
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Azul
    mask_blue = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))

    # Verde
    mask_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

    # Amarillo/Naranja
    mask_yellow = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([35, 255, 255]))

    combined = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_blue, cv2.bitwise_or(mask_green, mask_yellow)))
    result = img.copy()
    result[combined > 0] = [255, 255, 255]
    return result


def extract_text_from_colored_bg(img: np.ndarray) -> np.ndarray:
    """
    Extrae texto claro (blanco) de fondos de color.
    Ideal para tablas nutricionales con fondo rojo/azul y texto blanco.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = cv2.bitwise_not(binary)
    return result


def extract_text_adaptive(img: np.ndarray) -> np.ndarray:
    """Extraccion adaptativa de texto - funciona con texto claro u oscuro."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )

    h, w = binary.shape
    border_mean = np.mean(
        [
            np.mean(binary[0:10, :]),
            np.mean(binary[-10:, :]),
            np.mean(binary[:, 0:10]),
            np.mean(binary[:, -10:]),
        ]
    )

    if border_mean < 127:
        binary = cv2.bitwise_not(binary)

    return binary


def adaptive_binarize(img: np.ndarray, method: str = 'otsu') -> np.ndarray:
    """Binarizacion adaptativa."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    if method == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive_gaussian':
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
    else:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def deskew_image(img: np.ndarray) -> np.ndarray:
    """Corrige rotacion/inclinacion."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) < 0.5:
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos, sin = np.abs(matrix[0, 0]), np.abs(matrix[0, 1])
    new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    logger.info("Rotando %.2f grados", angle)
    return cv2.warpAffine(
        img,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def denoise_image(img: np.ndarray, method: str = 'bilateral') -> np.ndarray:
    """Reduce ruido."""
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (5, 5), 0)
    if method == 'bilateral':
        return cv2.bilateralFilter(img, 9, 75, 75)
    if method == 'nlm':
        if len(img.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    return img


def upscale_image(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Agranda imagen pequena."""
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def sharpen_image(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Aumenta nitidez."""
    kernel = np.array([[-1, -1, -1], [-1, 9 + strength, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


def invert_if_dark(img: np.ndarray) -> np.ndarray:
    """Invierte si el fondo es oscuro."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    if np.mean(gray) < 127:
        logger.info("Invirtiendo imagen (fondo oscuro)")
        return cv2.bitwise_not(img)
    return img


# ============================================
# PIPELINE PRINCIPAL
# ============================================

def preprocess_for_table_ocr(img: np.ndarray, options: dict | None = None) -> np.ndarray:
    """Pipeline optimizado para tablas nutricionales."""
    if options is None:
        options = {}

    result = img.copy()
    applied = []

    if options.get('rotate_180', False):
        result = cv2.rotate(result, cv2.ROTATE_180)
        applied.append('rotate_180')

    h, w = result.shape[:2]
    min_size = options.get('min_size', 800)
    max_scale = options.get('max_scale', 3.0)
    if options.get('upscale', True) and (h < min_size or w < min_size):
        scale = min(max(min_size / min(h, w), 1.5), max_scale)
        result = upscale_image(result, scale)
        applied.append(f'upscale_{scale:.1f}x')

    if options.get('enhance_contrast', True):
        result = enhance_contrast_clahe(result, options.get('clip_limit', 3.0))
        applied.append('clahe')

    if options.get('extract_white_text', False):
        result = extract_text_from_colored_bg(result)
        applied.append('extract_white_text')
    elif options.get('extract_text_adaptive', False):
        result = extract_text_adaptive(result)
        applied.append('extract_text_adaptive')
    elif options.get('remove_color_bg', True):
        result = remove_color_background(result)
        applied.append('remove_color_bg')

    if options.get('deskew', True):
        result = deskew_image(result)
        applied.append('deskew')

    if options.get('denoise', True):
        denoise_method = options.get('denoise_method', 'bilateral')
        if len(result.shape) == 2:
            result = cv2.medianBlur(result, 3)
        else:
            result = denoise_image(result, denoise_method)
        applied.append('denoise')

    if options.get('sharpen', False) and len(result.shape) == 3:
        result = sharpen_image(result, options.get('sharpen_strength', 1.0))
        applied.append('sharpen')

    if options.get('binarize', False) and len(result.shape) == 3:
        result = adaptive_binarize(result, options.get('binarize_method', 'otsu'))
        applied.append('binarize')

    if options.get('auto_invert', True) and len(result.shape) == 3:
        result = invert_if_dark(result)

    logger.info("Aplicado: %s", applied)
    return result
