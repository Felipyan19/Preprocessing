import logging

from flask import Blueprint, jsonify, request

from .analysis import analyze_image
from .image_io import (
    decode_base64_image,
    decode_base64_pdf,
    download_image_from_url,
    download_pdf_from_url,
    encode_image_to_base64,
)
from .pdf_extract import extract_pdf_fe
from .preprocessing import preprocess_for_table_ocr

logger = logging.getLogger(__name__)

api = Blueprint('api', __name__)


@api.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@api.route('/preprocess', methods=['POST'])
def preprocess():
    """
    POST /preprocess
    Body: {
        "image_url": "https://...",     # o
        "image_base64": "base64...",
        "preset": "table_ocr",          # Presets disponibles:
                                        #   - table_ocr (default)
                                        #   - table_ocr_aggressive
                                        #   - white_text_on_color
                                        #   - red_table_blurry (NUEVO: para fondo rojo + texto blanco borroso)
                                        #   - smart_auto (NUEVO: análisis inteligente automático)
                                        #   - minimal
        
        # Opciones directas (sin anidación):
        "smart_table_analysis": true,
        "force_strategy": "red_background_advanced",  # Estrategias: white_on_black, black_on_white,
                                                      # enhance_contrast, extract_luminosity, 
                                                      # red_background_advanced (NUEVO), invert_colors
        "deblur": true,
        "deblur_method": "unsharp",
        "upscale": true,
        "min_size": 1000,
        "max_scale": 3.0,
        "rotate_180": false,
        "enhance_contrast": true,
        "clip_limit": 3.0,
        "remove_color_bg": true,
        "extract_white_text": false,
        "extract_text_adaptive": false,
        "deskew": true,
        "denoise": true,
        "denoise_method": "bilateral",
        "sharpen": false,
        "sharpen_strength": 1.0,
        "binarize": false,
        "binarize_method": "otsu",
        "auto_invert": true
    }
    """
    try:
        data = request.json or {}

        if 'image_url' in data:
            img = download_image_from_url(data['image_url'])
        elif 'image_base64' in data:
            img = decode_base64_image(data['image_base64'])
        else:
            return jsonify({'error': 'Falta image_url o image_base64'}), 400

        if img is None:
            return jsonify({'error': 'No se pudo decodificar imagen'}), 400

        preset = data.get('preset', 'table_ocr')
        
        # Claves especiales que no son opciones de procesamiento
        special_keys = {'image_url', 'image_base64', 'pdf_url', 'pdf_base64', 'preset'}
        
        # Extraer todas las opciones directamente del nivel raíz (excluyendo claves especiales)
        options = {k: v for k, v in data.items() if k not in special_keys}

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
            elif preset == 'white_text_on_color':
                options = {
                    'rotate_180': False,
                    'upscale': True,
                    'min_size': 1200,
                    'max_scale': 4.0,
                    'enhance_contrast': True,
                    'clip_limit': 5.0,
                    'extract_white_text': True,
                    'remove_color_bg': False,
                    'deskew': True,
                    'denoise': True,
                    'sharpen': False,
                    'binarize': False,
                    'auto_invert': False,
                }
            elif preset == 'red_table_blurry':
                # NUEVO: Preset óptimo para tablas con fondo rojo y texto blanco borroso
                options = {
                    'smart_table_analysis': True,
                    'force_strategy': 'red_background_advanced',
                    'upscale': True,
                    'min_size': 1000,
                    'max_scale': 3.0,
                    'deblur': True,
                    'deblur_method': 'unsharp',
                }
            elif preset == 'smart_auto':
                # Análisis inteligente automático (sin forzar estrategia)
                options = {
                    'smart_table_analysis': True,
                    'upscale': True,
                    'min_size': 1000,
                    'max_scale': 3.0,
                    'deblur': True,
                    'deblur_method': 'unsharp',
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

        processed, metadata = preprocess_for_table_ocr(img, options)
        result_b64 = encode_image_to_base64(processed)

        return jsonify(
            {
                'success': True,
                'processed_image': result_b64,
                'original_size': {'w': img.shape[1], 'h': img.shape[0]},
                'processed_size': {'w': processed.shape[1], 'h': processed.shape[0]},
                'preprocessing_metadata': metadata,
            }
        )

    except Exception as exc:
        logger.error("Error: %s", exc)
        return jsonify({'error': str(exc)}), 500


@api.route('/analyze', methods=['POST'])
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

        result = analyze_image(img)
        return jsonify(result)

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@api.route('/extract-pdf-fe', methods=['POST'])
def extract_pdf_fe_endpoint():
    """
    POST /extract-pdf-fe
    Body: {
        "pdf_url": "https://...",     # o
        "pdf_base64": "base64..."
    }
    """
    try:
        data = request.json or {}
        if 'pdf_url' in data:
            pdf_bytes = download_pdf_from_url(data['pdf_url'])
        elif 'pdf_base64' in data:
            pdf_bytes = decode_base64_pdf(data['pdf_base64'])
        else:
            return jsonify({'error': 'Falta pdf_url o pdf_base64'}), 400

        result = extract_pdf_fe(pdf_bytes)
        return jsonify(result)

    except ValueError as exc:
        logger.warning("PDF sin texto embebido: %s", exc)
        return jsonify({'error': str(exc)}), 422
    except Exception as exc:
        logger.error("Error: %s", exc)
        return jsonify({'error': str(exc)}), 500
