# ============================================
# MICROSERVICIO DE PREPROCESAMIENTO DE IMAGENES
# Para mejorar OCR en tablas con fondos de color
# ============================================

from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import requests
import logging
import fitz  # PyMuPDF para PDFs
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def pdf_to_image(pdf_bytes: bytes, dpi: int = 300) -> np.ndarray:
    """Convierte primera página de PDF a imagen."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]  # Primera página
    # Renderizar a alta resolución
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    # Convertir a numpy array
    img_data = pix.tobytes("png")
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    doc.close()
    logger.info("PDF convertido a imagen: %dx%d", img.shape[1], img.shape[0])
    return img

def is_pdf(content: bytes) -> bool:
    """Detecta si el contenido es un PDF."""
    return content[:4] == b'%PDF'

def download_image_from_url(url: str) -> np.ndarray:
    """Descarga imagen o PDF desde URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content = response.content

    # Detectar si es PDF
    if is_pdf(content) or url.lower().endswith('.pdf'):
        logger.info("Detectado PDF, convirtiendo a imagen...")
        return pdf_to_image(content)

    img_array = np.frombuffer(content, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decodifica imagen o PDF base64."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_bytes = base64.b64decode(base64_string)

    # Detectar si es PDF
    if is_pdf(img_bytes):
        logger.info("Detectado PDF en base64, convirtiendo a imagen...")
        return pdf_to_image(img_bytes)

    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_image_to_base64(img: np.ndarray, fmt: str = 'png') -> str:
    """Codifica imagen a base64."""
    _, buffer = cv2.imencode(f'.{fmt}', img)
    return base64.b64encode(buffer).decode('utf-8')

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
    """Elimina fondos rojos, azules, verdes."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rojo
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Azul
    mask_blue = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))

    # Verde
    mask_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

    combined = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_blue, mask_green))
    result = img.copy()
    result[combined > 0] = [255, 255, 255]
    return result

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

    # 1. Upscale si es pequena
    h, w = result.shape[:2]
    if options.get('upscale', True) and (h < 800 or w < 800):
        scale = min(max(800 / min(h, w), 1.5), 3.0)
        result = upscale_image(result, scale)
        applied.append(f'upscale_{scale:.1f}x')

    # 2. Mejorar contraste (critico para fondos de color)
    if options.get('enhance_contrast', True):
        result = enhance_contrast_clahe(result, options.get('clip_limit', 3.0))
        applied.append('clahe')

    # 3. Eliminar fondos de color
    if options.get('remove_color_bg', True):
        result = remove_color_background(result)
        applied.append('remove_color_bg')

    # 4. Corregir rotacion
    if options.get('deskew', True):
        result = deskew_image(result)
        applied.append('deskew')

    # 5. Reducir ruido
    if options.get('denoise', True):
        result = denoise_image(result, options.get('denoise_method', 'bilateral'))
        applied.append('denoise')

    # 6. Nitidez (opcional)
    if options.get('sharpen', False):
        result = sharpen_image(result, options.get('sharpen_strength', 1.0))
        applied.append('sharpen')

    # 7. Binarizar (opcional)
    if options.get('binarize', False):
        result = adaptive_binarize(result, options.get('binarize_method', 'otsu'))
        applied.append('binarize')

    # 8. Invertir si es necesario
    if options.get('auto_invert', True):
        result = invert_if_dark(result)

    logger.info("Aplicado: %s", applied)
    return result

# ============================================
# ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/preprocess', methods=['POST'])
def preprocess():
    """
    POST /preprocess
    Body: {
        "image_url": "https://...",     # o
        "image_base64": "base64...",
        "preset": "table_ocr",          # table_ocr, table_ocr_aggressive, minimal
        "options": {...}                # opcional
    }
    """
    try:
        data = request.json or {}

        # Obtener imagen
        if 'image_url' in data:
            img = download_image_from_url(data['image_url'])
        elif 'image_base64' in data:
            img = decode_base64_image(data['image_base64'])
        else:
            return jsonify({'error': 'Falta image_url o image_base64'}), 400

        if img is None:
            return jsonify({'error': 'No se pudo decodificar imagen'}), 400

        # Presets
        preset = data.get('preset', 'table_ocr')
        options = data.get('options', {})

        if not options:
            if preset == 'table_ocr':
                options = {
                    'upscale': True,
                    'enhance_contrast': True,
                    'remove_color_bg': True,
                    'deskew': True,
                    'denoise': True,
                    'sharpen': False,
                    'binarize': False,
                    'auto_invert': True,
                    'clip_limit': 3.0,
                }
            elif preset == 'table_ocr_aggressive':
                options = {
                    'upscale': True,
                    'enhance_contrast': True,
                    'remove_color_bg': True,
                    'deskew': True,
                    'denoise': True,
                    'denoise_method': 'nlm',
                    'sharpen': True,
                    'sharpen_strength': 1.5,
                    'binarize': True,
                    'binarize_method': 'adaptive_gaussian',
                    'auto_invert': True,
                    'clip_limit': 4.0,
                }
            elif preset == 'minimal':
                options = {
                    'upscale': False,
                    'enhance_contrast': True,
                    'remove_color_bg': False,
                    'deskew': False,
                    'denoise': False,
                    'auto_invert': True,
                }

        # Procesar
        processed = preprocess_for_table_ocr(img, options)
        result_b64 = encode_image_to_base64(processed)

        return jsonify(
            {
                'success': True,
                'processed_image': result_b64,
                'original_size': {'w': img.shape[1], 'h': img.shape[0]},
                'processed_size': {'w': processed.shape[1], 'h': processed.shape[0]},
            }
        )

    except Exception as exc:
        logger.error("Error: %s", exc)
        return jsonify({'error': str(exc)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analiza imagen y sugiere operaciones."""
    try:
        data = request.json or {}

        if 'image_url' in data:
            img = download_image_from_url(data['image_url'])
        elif 'image_base64' in data:
            img = decode_base64_image(data['image_base64'])
        else:
            return jsonify({'error': 'Falta imagen'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Detectar rojo
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255])),
        )
        red_pct = np.sum(mask_red > 0) / mask_red.size * 100

        # Detectar azul
        mask_blue = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
        blue_pct = np.sum(mask_blue > 0) / mask_blue.size * 100

        recommendations = []

        if img.shape[0] < 500 or img.shape[1] < 500:
            recommendations.append('upscale')
        if red_pct > 10 or blue_pct > 10:
            recommendations.append('remove_color_bg')
        if np.mean(gray) < 100:
            recommendations.append('enhance_contrast')

        return jsonify(
            {
                'size': {'w': img.shape[1], 'h': img.shape[0]},
                'brightness': float(np.mean(gray)),
                'red_percent': round(red_pct, 1),
                'blue_percent': round(blue_pct, 1),
                'recommendations': recommendations,
                'suggested_preset': 'table_ocr_aggressive'
                if len(recommendations) >= 2
                else 'table_ocr',
            }
        )

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

if __name__ == '__main__':
    import os

    port = int(os.environ.get('PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('DEBUG', 'false') == 'true',
    )
