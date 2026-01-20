import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ============================================
# OPERACIONES DE PREPROCESAMIENTO
# ============================================

def enhance_contrast_clahe(img: np.ndarray, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE - muy efectivo para texto blanco sobre fondos de color.
    
    Args:
        clip_limit: Límite de contraste (1.0-5.0). Más bajo = más suave.
        tile_grid_size: Tamaño de la cuadrícula (default 8x8)
    """
    if len(img.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
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


def extract_white_text_from_red_bg_advanced(img: np.ndarray) -> np.ndarray:
    """
    Pipeline optimizado específicamente para texto BLANCO sobre fondo ROJO borroso.
    Combina extracción de luminosidad LAB + eliminación de rojo HSV.
    
    Este es el método más efectivo para tablas nutricionales rojas.
    """
    # Paso 1: Extraer canal L (luminosidad) del espacio LAB
    # Esto ignora el color y se enfoca en la intensidad de luz
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Paso 2: Aplicar CLAHE agresivo al canal L para aumentar contraste
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    
    # Paso 3: Eliminar fondo rojo usando HSV como máscara adicional
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Rangos optimizados para rojos saturados (tablas nutricionales)
    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Paso 4: Aplicar la máscara al canal L mejorado
    # Donde hay rojo (fondo), ponemos blanco
    l_enhanced[red_mask > 0] = 255
    
    # Paso 5: Invertir (texto blanco -> texto negro para OCR)
    inverted = cv2.bitwise_not(l_enhanced)
    
    # Paso 6: Binarización Otsu
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Paso 7: Limpieza con morfología (eliminar ruido pequeño)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Paso 8: Volver a invertir para tener texto negro sobre fondo blanco
    result = cv2.bitwise_not(cleaned)
    
    logger.info("Aplicado pipeline avanzado: LAB + HSV + CLAHE + morfología")
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


def adaptive_binarize(img: np.ndarray, method: str = 'otsu', block_size: int = 11, C: int = 2) -> np.ndarray:
    """
    Binarización adaptativa.
    
    Args:
        method: 'otsu', 'adaptive_gaussian', 'adaptive_mean'
        block_size: Tamaño del bloque para adaptive (debe ser impar). Más grande = más suave.
        C: Constante que se resta del promedio. Más alto = más conservador.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    if method == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive_gaussian':
        # Asegurar que block_size es impar
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C,
        )
    elif method == 'adaptive_mean':
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C,
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


def denoise_image(img: np.ndarray, method: str = 'bilateral', d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Reduce ruido.
    
    Args:
        method: 'bilateral', 'gaussian', 'nlm'
        d: Diámetro del filtro bilateral (5-9 típico)
        sigma_color: Filtro en espacio de color (más bajo = más selectivo)
        sigma_space: Filtro en espacio de coordenadas
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (5, 5), 0)
    if method == 'bilateral':
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    if method == 'nlm':
        if len(img.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    return img


def upscale_image(img: np.ndarray, scale: float = 2.0, method: str = 'cubic') -> np.ndarray:
    """
    Agranda imagen pequeña.
    
    Args:
        img: Imagen a escalar
        scale: Factor de escalado
        method: 'cubic' (default), 'lanczos4' (mejor calidad para texto pequeño), 'linear'
    """
    if method == 'lanczos4':
        # Lanczos4 preserva mejor los detalles finos (texto pequeño)
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    elif method == 'linear':
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def sharpen_image(img: np.ndarray, strength: float = 1.0, method: str = 'kernel') -> np.ndarray:
    """
    Aumenta nitidez.
    
    Args:
        img: Imagen a procesar
        strength: Intensidad del sharpening
        method: 'kernel' (rápido), 'unsharp' (mejor calidad para texto pequeño)
    """
    if method == 'unsharp':
        # Unsharp mask - mejor para texto pequeño
        gaussian = cv2.GaussianBlur(img, (0, 0), 1.0)
        sharpened = cv2.addWeighted(img, 1.0 + strength, gaussian, -strength, 0)
        return sharpened
    else:
        # Kernel tradicional
        kernel = np.array([[-1, -1, -1], [-1, 9 + strength, -1], [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)


def deblur_image(img: np.ndarray, method: str = 'unsharp', strength: float = 1.0) -> np.ndarray:
    """
    Reduce borrosidad de la imagen.
    Especialmente efectivo para tablas con texto borroso.
    
    Args:
        img: Imagen a procesar
        method: 'unsharp', 'laplacian', 'aggressive' (para texto muy pequeño)
        strength: Factor de intensidad (0.5-2.0)
    """
    if method == 'aggressive':
        # Deblurring agresivo para texto muy pequeño y pegado
        # Paso 1: Unsharp mask fuerte
        gaussian = cv2.GaussianBlur(img, (0, 0), 3.0)
        unsharp = cv2.addWeighted(img, 1.5 * strength, gaussian, -0.5 * strength, 0)
        
        # Paso 2: High-pass filter para resaltar bordes
        if len(unsharp.shape) == 3:
            gray = cv2.cvtColor(unsharp, cv2.COLOR_BGR2GRAY)
        else:
            gray = unsharp
        
        # Aplicar filtro high-pass
        kernel_size = 3
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]) * (0.3 * strength)
        kernel[1, 1] = 1 + kernel[1, 1]
        
        if len(img.shape) == 3:
            sharpened = cv2.filter2D(unsharp, -1, kernel)
        else:
            sharpened = cv2.filter2D(gray, -1, kernel)
        
        return sharpened
        
    elif method == 'unsharp':
        # Unsharp masking - muy efectivo para borrosidad
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        unsharp = cv2.addWeighted(img, 1.0 + strength, gaussian, -strength, 0)
        return unsharp
    elif method == 'laplacian':
        # Sharpening con Laplacian
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        if len(img.shape) == 3:
            sharpened = cv2.addWeighted(img, 1.0 + 0.5 * strength, cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR), -0.5 * strength, 0)
        else:
            sharpened = cv2.addWeighted(img, 1.0 + 0.5 * strength, laplacian, -0.5 * strength, 0)
        return sharpened
    return img


def invert_if_dark(img: np.ndarray) -> np.ndarray:
    """Invierte si el fondo es oscuro."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    if np.mean(gray) < 127:
        logger.info("Invirtiendo imagen (fondo oscuro)")
        return cv2.bitwise_not(img)
    return img


def apply_morphology(img: np.ndarray, mode: str = 'open', kernel_size: tuple = (2, 2), iterations: int = 1) -> np.ndarray:
    """
    Aplica operaciones morfológicas para limpiar la imagen binarizada.
    
    Args:
        mode: 'open' (elimina ruido pequeño), 'close' (rellena huecos), 'erode', 'dilate'
        kernel_size: Tamaño del kernel (ancho, alto)
        iterations: Número de veces que se aplica la operación
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    if mode == 'open':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif mode == 'close':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif mode == 'erode':
        return cv2.erode(img, kernel, iterations=iterations)
    elif mode == 'dilate':
        return cv2.dilate(img, kernel, iterations=iterations)
    else:
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
        - 'red_background_advanced': Fondo rojo específicamente (MEJOR para tablas nutricionales)
        - 'invert_colors': Invertir todo
    """
    text_lum = analysis['text_luminosity']
    bg_lum = analysis['bg_luminosity']
    contrast = analysis['contrast']

    # Detectar si el fondo es de color saturado
    bg_r, bg_g, bg_b = analysis['bg_color']
    bg_saturation = (max(bg_r, bg_g, bg_b) - min(bg_r, bg_g, bg_b)) / 255.0
    
    # NUEVO: Detectar específicamente fondo ROJO (tablas nutricionales)
    # Rojo: R alto, G y B bajos
    is_red_background = (bg_r > 150 and bg_g < 100 and bg_b < 100) or \
                       (bg_r > bg_g + 80 and bg_r > bg_b + 80)
    
    # Detectar texto blanco: luminosidad alta
    is_white_text = text_lum > 0.7
    
    # ESTRATEGIA PRIORITARIA: Fondo rojo + texto blanco
    if is_red_background and is_white_text:
        logger.info("Estrategia: red_background_advanced (fondo rojo detectado R=%d, texto blanco L=%.2f)", 
                   bg_r, text_lum)
        return 'red_background_advanced'

    # Bajo contraste: necesita mejora
    if contrast < 0.3:
        logger.info("Estrategia: enhance_contrast (contraste bajo: %.2f)", contrast)
        return 'enhance_contrast'

    # Fondo de color saturado genérico
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

    if strategy == 'red_background_advanced':
        # NUEVA ESTRATEGIA: Pipeline optimizado para fondo rojo + texto blanco borroso
        result = extract_white_text_from_red_bg_advanced(result)
        logger.info("Aplicado: red_background_advanced (pipeline LAB+HSV optimizado)")

    elif strategy == 'enhance_contrast':
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
        upscale_method = options.get('upscale_method', 'cubic')
        if options.get('upscale', True) and (h < min_size or w < min_size):
            scale = min(max(min_size / min(h, w), 1.5), max_scale)
            result = upscale_image(result, scale, upscale_method)
            applied.append(f'upscale_{scale:.1f}x_{upscale_method}')

        # NUEVO: Deblurring ANTES de conversión (clave para texto pequeño)
        if options.get('deblur', False):
            deblur_method = options.get('deblur_method', 'unsharp')
            deblur_strength = options.get('deblur_strength', 1.0)
            result = deblur_image(result, deblur_method, deblur_strength)
            applied.append(f'deblur_{deblur_method}_{deblur_strength}')
            logger.info("Aplicado deblurring con método: %s, strength: %.1f", deblur_method, deblur_strength)

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

        # Sharpening DESPUÉS de conversión (opcional, para texto pequeño)
        if options.get('sharpen', False):
            sharpen_strength = options.get('sharpen_strength', 1.0)
            sharpen_method = options.get('sharpen_method', 'unsharp')
            if len(result.shape) == 2:
                # Imagen ya binarizada, aplicar sharpening suave
                result = sharpen_image(result, sharpen_strength, sharpen_method)
                applied.append(f'sharpen_{sharpen_method}_{sharpen_strength}')
                logger.info("Aplicado sharpening post-conversión: %s, strength: %.1f", sharpen_method, sharpen_strength)

        # Denoise ligero después de la conversión (solo si no es texto muy pequeño)
        if not options.get('preserve_fine_details', False):
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

    # IMPORTANTE: Convertir a escala de grises ANTES de los filtros (si está activado)
    # Esto evita que CLAHE blanquee la imagen a color
    if options.get('convert_to_grayscale', False) and len(result.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        applied.append('convert_to_grayscale')
        logger.info("Convertido a escala de grises ANTES de filtros")

    if options.get('enhance_contrast', True):
        tile_grid = tuple(options.get('clahe_tile_grid_size', [8, 8]))
        result = enhance_contrast_clahe(result, options.get('clip_limit', 3.0), tile_grid)
        applied.append(f'clahe_{options.get("clip_limit", 3.0)}')

    # Solo procesar color si NO se convirtió a escala de grises
    if not options.get('convert_to_grayscale', False):
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

    # Deblur (antes de denoise para mejor resultado)
    if options.get('deblur', False):
        deblur_method = options.get('deblur_method', 'unsharp')
        deblur_strength = options.get('deblur_strength', 1.0)
        result = deblur_image(result, deblur_method, deblur_strength)
        applied.append(f'deblur_{deblur_method}_{deblur_strength}')
        logger.info("Aplicado deblurring: %s, strength: %.1f", deblur_method, deblur_strength)

    if options.get('denoise', True):
        denoise_method = options.get('denoise_method', 'bilateral')
        if len(result.shape) == 2:
            result = cv2.medianBlur(result, 3)
        else:
            d = options.get('bilateral_d', 9)
            sigma_color = options.get('bilateral_sigma_color', 75)
            sigma_space = options.get('bilateral_sigma_space', 75)
            result = denoise_image(result, denoise_method, d, sigma_color, sigma_space)
        applied.append(f'denoise_{denoise_method}')

    if options.get('sharpen', False) and len(result.shape) == 3:
        result = sharpen_image(result, options.get('sharpen_strength', 1.0))
        applied.append('sharpen')

    if options.get('binarize', False) and len(result.shape) == 3:
        block_size = options.get('adaptive_block_size', 11)
        C = options.get('adaptive_C', 2)
        result = adaptive_binarize(result, options.get('binarize_method', 'otsu'), block_size, C)
        applied.append(f'binarize_{options.get("binarize_method", "otsu")}')

    # Morfología post-procesamiento
    if options.get('post_morphology', False) and len(result.shape) == 2:
        morph_mode = options.get('morphology_mode', 'open')
        kernel_size = tuple(options.get('morphology_kernel', [2, 2]))
        iterations = options.get('morphology_iterations', 1)
        result = apply_morphology(result, morph_mode, kernel_size, iterations)
        applied.append(f'morph_{morph_mode}')

    if options.get('auto_invert', True) and len(result.shape) == 3:
        result = invert_if_dark(result)

    # Nota: convert_to_grayscale se movió al INICIO del pipeline (línea ~768)
    # para que los filtros procesen la imagen gris desde el principio

    metadata['applied_operations'] = applied
    logger.info("Aplicado: %s", applied)
    return result, metadata
