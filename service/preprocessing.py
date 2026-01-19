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
# ANÁLISIS DINÁMICO DE COLOR PARA TABLAS
# ============================================

def detect_table_regions(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detecta regiones rectangulares que contengan tablas.
    Retorna lista de bounding boxes: [(x, y, w, h), ...]
    Si no detecta nada, retorna la imagen completa.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape

    # Detectar bordes
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detectar líneas horizontales y verticales (características de tablas)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 30, 1))
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 30))

    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_horizontal)
    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_vertical)

    # Combinar líneas
    table_structure = cv2.add(horizontal_lines, vertical_lines)

    # Dilatar para conectar regiones
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(table_structure, kernel, iterations=2)

    # Encontrar contornos
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    min_area = (w * h) * 0.05  # Al menos 5% del área total

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch

        # Filtrar regiones muy pequeñas o que ocupan casi toda la imagen
        if area > min_area and area < (w * h) * 0.95:
            # Expandir ligeramente la región para capturar todo el contenido
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            cw = min(w - x, cw + 2 * margin)
            ch = min(h - y, ch + 2 * margin)
            regions.append((x, y, cw, ch))

    # Si no se detectaron regiones, usar toda la imagen
    if not regions:
        logger.info("No se detectaron regiones de tabla, usando imagen completa")
        return [(0, 0, w, h)]

    logger.info("Detectadas %d regiones de tabla", len(regions))
    return regions


def analyze_text_and_background_colors(img: np.ndarray, region: tuple[int, int, int, int] | None = None) -> dict:
    """
    Analiza colores dominantes del texto y fondo en una región.

    Args:
        img: Imagen BGR
        region: (x, y, w, h) o None para toda la imagen

    Returns:
        dict con:
            - text_color: (R, G, B)
            - text_luminosity: 0.0-1.0
            - bg_color: (R, G, B)
            - bg_luminosity: 0.0-1.0
            - contrast: 0.0-1.0
    """
    # Extraer región de interés
    if region:
        x, y, w, h = region
        roi = img[y:y+h, x:x+w].copy()
    else:
        roi = img.copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Detectar bordes para identificar texto
    edges = cv2.Canny(gray, 50, 150)

    # Dilatar bordes para capturar área de texto
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_mask = cv2.dilate(edges, kernel, iterations=2)

    # Invertir para obtener máscara de fondo
    bg_mask = cv2.bitwise_not(text_mask)

    # Calcular color promedio del texto
    text_pixels = roi[text_mask > 0]
    if len(text_pixels) > 0:
        text_color = np.mean(text_pixels, axis=0).astype(int)
        text_gray = np.mean(gray[text_mask > 0])
    else:
        # Fallback: usar píxeles más oscuros o más claros
        threshold = np.median(gray)
        if np.mean(gray) > 127:
            # Imagen clara, texto probablemente oscuro
            text_color = np.mean(roi[gray < threshold], axis=0).astype(int) if np.any(gray < threshold) else np.array([0, 0, 0])
            text_gray = np.mean(gray[gray < threshold]) if np.any(gray < threshold) else 0
        else:
            # Imagen oscura, texto probablemente claro
            text_color = np.mean(roi[gray > threshold], axis=0).astype(int) if np.any(gray > threshold) else np.array([255, 255, 255])
            text_gray = np.mean(gray[gray > threshold]) if np.any(gray > threshold) else 255

    # Calcular color promedio del fondo
    bg_pixels = roi[bg_mask > 0]
    if len(bg_pixels) > 50:  # Necesitamos suficientes píxeles
        bg_color = np.mean(bg_pixels, axis=0).astype(int)
        bg_gray = np.mean(gray[bg_mask > 0])
    else:
        # Fallback: usar color promedio global
        bg_color = np.mean(roi.reshape(-1, 3), axis=0).astype(int)
        bg_gray = np.mean(gray)

    # Calcular luminosidad (0.0 = negro, 1.0 = blanco)
    text_luminosity = float(text_gray) / 255.0
    bg_luminosity = float(bg_gray) / 255.0

    # Calcular contraste
    contrast = abs(text_luminosity - bg_luminosity)

    analysis = {
        'text_color': tuple(int(c) for c in text_color[::-1]),  # BGR -> RGB
        'text_luminosity': text_luminosity,
        'bg_color': tuple(int(c) for c in bg_color[::-1]),  # BGR -> RGB
        'bg_luminosity': bg_luminosity,
        'contrast': contrast,
    }

    logger.info(
        "Análisis de color - Texto: RGB%s (L=%.2f), Fondo: RGB%s (L=%.2f), Contraste: %.2f",
        analysis['text_color'],
        text_luminosity,
        analysis['bg_color'],
        bg_luminosity,
        contrast,
    )

    return analysis


def decide_conversion_strategy(analysis: dict) -> str:
    """
    Decide la estrategia de conversión basada en el análisis de colores.

    Estrategias:
        - 'white_on_black': Texto claro sobre fondo oscuro
        - 'black_on_white': Texto oscuro sobre fondo claro
        - 'enhance_contrast': Bajo contraste, aumentar primero
        - 'extract_luminosity': Fondo de color saturado
        - 'invert_colors': Invertir todo
    """
    text_lum = analysis['text_luminosity']
    bg_lum = analysis['bg_luminosity']
    contrast = analysis['contrast']

    # Bajo contraste: necesita mejora
    if contrast < 0.3:
        logger.info("Estrategia: enhance_contrast (contraste bajo: %.2f)", contrast)
        return 'enhance_contrast'

    # Detectar si el fondo es de color saturado
    bg_r, bg_g, bg_b = analysis['bg_color']
    bg_saturation = (max(bg_r, bg_g, bg_b) - min(bg_r, bg_g, bg_b)) / 255.0

    if bg_saturation > 0.4 and contrast > 0.3:
        logger.info("Estrategia: extract_luminosity (fondo saturado: %.2f)", bg_saturation)
        return 'extract_luminosity'

    # Texto claro sobre fondo oscuro
    if text_lum > 0.6 and bg_lum < 0.5:
        logger.info("Estrategia: white_on_black (texto claro sobre fondo oscuro)")
        return 'white_on_black'

    # Texto oscuro sobre fondo claro
    if text_lum < 0.5 and bg_lum > 0.6:
        logger.info("Estrategia: black_on_white (texto oscuro sobre fondo claro)")
        return 'black_on_white'

    # Invertido: texto oscuro sobre fondo claro pero al revés
    if text_lum < 0.5 and bg_lum < 0.5:
        logger.info("Estrategia: invert_colors (ambos oscuros, necesita inversión)")
        return 'invert_colors'

    # Por defecto: texto oscuro sobre fondo claro (estándar OCR)
    logger.info("Estrategia: black_on_white (por defecto)")
    return 'black_on_white'


def apply_smart_conversion(img: np.ndarray, strategy: str, analysis: dict) -> np.ndarray:
    """
    Aplica la estrategia de conversión decidida.
    """
    result = img.copy()

    if strategy == 'enhance_contrast':
        # Aumentar contraste agresivamente
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Binarización adaptativa
        result = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )
        logger.info("Aplicado: enhance_contrast con CLAHE + adaptive threshold")

    elif strategy == 'extract_luminosity':
        # Extraer canal de luminosidad (ignora color)
        if len(result.shape) == 3:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l_channel, _, _ = cv2.split(lab)
            # Aplicar CLAHE al canal L
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l_channel)
            # Binarización
            _, result = cv2.threshold(enhanced_l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Decidir si invertir basado en luminosidad del texto
            if analysis['text_luminosity'] > 0.6:
                result = cv2.bitwise_not(result)
            logger.info("Aplicado: extract_luminosity (canal L de LAB)")
        else:
            result = adaptive_binarize(result)

    elif strategy == 'white_on_black':
        # Convertir a escala de grises
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
        # Invertir para que texto oscuro sobre fondo claro
        inverted = cv2.bitwise_not(gray)
        # Binarización Otsu
        _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Volver a invertir para texto claro sobre fondo oscuro -> texto oscuro sobre fondo claro
        result = cv2.bitwise_not(binary)
        logger.info("Aplicado: white_on_black (invertir + Otsu + invertir)")

    elif strategy == 'black_on_white':
        # Texto oscuro sobre fondo claro - solo binarizar
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info("Aplicado: black_on_white (Otsu directo)")

    elif strategy == 'invert_colors':
        # Invertir todo
        result = cv2.bitwise_not(result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info("Aplicado: invert_colors")

    return result


# ============================================
# PIPELINE PRINCIPAL
# ============================================

def preprocess_for_table_ocr(img: np.ndarray, options: dict | None = None) -> tuple[np.ndarray, dict]:
    """
    Pipeline optimizado para tablas nutricionales.

    Returns:
        tuple: (imagen_procesada, metadata) donde metadata contiene:
            - applied_operations: lista de operaciones aplicadas
            - smart_analysis_used: bool
            - strategy: estrategia usada (si smart_analysis_used=True)
            - color_analysis: análisis de colores (si smart_analysis_used=True)
    """
    if options is None:
        options = {}

    result = img.copy()
    applied = []
    metadata = {
        'applied_operations': [],
        'smart_analysis_used': False,
    }

    # NUEVO: Análisis inteligente de color para tablas
    if options.get('smart_table_analysis', True):
        logger.info("=== ANÁLISIS INTELIGENTE DE TABLAS ACTIVADO ===")
        metadata['smart_analysis_used'] = True

        # Rotación inicial si es necesario
        if options.get('rotate_180', False):
            result = cv2.rotate(result, cv2.ROTATE_180)
            applied.append('rotate_180')

        # Upscale inicial si es necesario
        h, w = result.shape[:2]
        min_size = options.get('min_size', 800)
        max_scale = options.get('max_scale', 3.0)
        if options.get('upscale', True) and (h < min_size or w < min_size):
            scale = min(max(min_size / min(h, w), 1.5), max_scale)
            result = upscale_image(result, scale)
            applied.append(f'upscale_{scale:.1f}x')

        # Detectar regiones de tabla
        regions = detect_table_regions(result)
        metadata['detected_regions'] = len(regions)

        # Analizar la región más grande (asumiendo que es la tabla principal)
        largest_region = max(regions, key=lambda r: r[2] * r[3])

        # Analizar colores de texto y fondo
        analysis = analyze_text_and_background_colors(result, largest_region)

        # Decidir estrategia de conversión (o usar la forzada)
        force_strategy = options.get('force_strategy')
        if force_strategy:
            strategy = force_strategy
            metadata['strategy_forced'] = True
            logger.info("Estrategia FORZADA por usuario: %s", strategy)
        else:
            strategy = decide_conversion_strategy(analysis)
            metadata['strategy_forced'] = False

        # Guardar en metadata
        metadata['strategy'] = strategy
        metadata['color_analysis'] = {
            'text_color': analysis['text_color'],
            'text_luminosity': round(analysis['text_luminosity'], 2),
            'bg_color': analysis['bg_color'],
            'bg_luminosity': round(analysis['bg_luminosity'], 2),
            'contrast': round(analysis['contrast'], 2),
        }

        # Aplicar conversión inteligente
        result = apply_smart_conversion(result, strategy, analysis)
        applied.append(f'smart_conversion_{strategy}')

        # Denoise ligero después de la conversión
        if len(result.shape) == 2:
            result = cv2.medianBlur(result, 3)
            applied.append('median_blur')

        metadata['applied_operations'] = applied
        logger.info("Pipeline inteligente completado: %s", applied)
        return result, metadata

    # FALLBACK: Pipeline clásico (si smart_table_analysis está desactivado)
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

    metadata['applied_operations'] = applied
    logger.info("Aplicado: %s", applied)
    return result, metadata
