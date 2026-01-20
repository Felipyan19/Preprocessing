import logging
from io import BytesIO

import cv2
from flask import Blueprint, jsonify, request, send_file

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
                                        #   - red_table_blurry (para fondo rojo + texto blanco borroso)
                                        #   - smart_auto (análisis inteligente automático)
                                        #   - small_text_sharp (para detección de estructura)
                                        #   - ocr_preserve_details (preserva detalles finos)
                                        #   - ocr_ultra_fine (CLAHE+bilateral+adaptive+morfología)
                                        #   - gemini_vision (NUEVO⭐: para Gemini/GPT-Vision/Claude)
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
            elif preset == 'small_text_sharp':
                # Para DETECCIÓN DE ESTRUCTURA de tabla (bordes, líneas)
                # ADVERTENCIA: Puede engrosar trazos y perder detalles finos para OCR
                options = {
                    'smart_table_analysis': True,
                    'upscale': True,
                    'min_size': 2000,
                    'max_scale': 5.0,
                    'upscale_method': 'lanczos4',
                    'deblur': True,
                    'deblur_method': 'aggressive',
                    'deblur_strength': 1.0,
                    'sharpen': True,
                    'sharpen_strength': 0.5,
                    'sharpen_method': 'unsharp',
                    'preserve_fine_details': True,
                }
            elif preset == 'ocr_preserve_details':
                # NUEVO: Para OCR de texto/números - PRESERVA detalles finos
                # Optimizado para mantener comas, puntos, símbolos (<, %, *, etc.)
                # Menos procesamiento = menos distorsión
                options = {
                    'smart_table_analysis': True,
                    'upscale': True,
                    'min_size': 1800,
                    'max_scale': 4.0,
                    'upscale_method': 'lanczos4',
                    'deblur': True,
                    'deblur_method': 'unsharp',  # Menos agresivo que aggressive
                    'deblur_strength': 0.6,  # Muy suave
                    'sharpen': False,  # Sin sharpening extra para no engrosar
                    'preserve_fine_details': True,
                }
            elif preset == 'ocr_ultra_fine':
                # NUEVO: Ultra-optimizado para OCR con parámetros granulares
                # Basado en feedback: CLAHE + bilateral + adaptive threshold + morfología
                # Ideal para tablas nutricionales con texto muy pequeño
                options = {
                    'smart_table_analysis': False,  # Control manual total
                    'upscale': True,
                    'min_size': 2400,
                    'max_scale': 3.0,
                    'upscale_method': 'lanczos4',
                    'denoise': True,
                    'denoise_method': 'bilateral',
                    'bilateral_d': 5,
                    'bilateral_sigma_color': 50,
                    'bilateral_sigma_space': 50,
                    'enhance_contrast': True,
                    'clip_limit': 1.8,
                    'clahe_tile_grid_size': [8, 8],
                    'remove_color_bg': True,
                    'deblur': True,
                    'deblur_method': 'unsharp',
                    'deblur_strength': 0.35,
                    'sharpen': False,
                    'binarize': True,
                    'binarize_method': 'adaptive_gaussian',
                    'adaptive_block_size': 51,
                    'adaptive_C': 9,
                    'post_morphology': True,
                    'morphology_mode': 'open',
                    'morphology_kernel': [2, 2],
                    'morphology_iterations': 1,
                    'preserve_fine_details': True,
                    'deskew': False,
                    'auto_invert': False,
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

    except ValueError as exc:
        logger.warning("Error de validación: %s", exc)
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        logger.error("Error interno: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': f'Error interno del servidor: {str(exc)}'}), 500


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

    except ValueError as exc:
        logger.warning("Error de validación en analyze: %s", exc)
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        logger.error("Error interno en analyze: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': f'Error interno del servidor: {str(exc)}'}), 500


@api.route('/preprocess-image', methods=['POST'])
def preprocess_image():
    """
    POST /preprocess-image
    
    Devuelve la imagen procesada directamente como archivo (no JSON).
    
    Body: {
        "image_url": "https://...",     # o
        "image_base64": "base64...",
        "preset": "table_ocr",          # Presets: table_ocr, table_ocr_aggressive, white_text_on_color,
                                        # red_table_blurry, smart_auto, minimal
        "format": "png",                # Formato de salida: "png" (default), "jpg", "jpeg", "webp"
        
        # Todas las opciones de preprocesamiento (ver /preprocess)
        "force_strategy": "white_on_black",
        "upscale": true,
        "deblur": true,
        # ... etc
    }
    
    Respuesta: archivo de imagen (image/png, image/jpeg, etc.)
    Headers opcionales en la respuesta:
        - X-Original-Width: ancho original
        - X-Original-Height: alto original
        - X-Processed-Width: ancho procesado
        - X-Processed-Height: alto procesado
        - X-Applied-Operations: operaciones aplicadas
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
        output_format = data.get('format', 'png').lower()
        
        # Validar formato
        if output_format not in ['png', 'jpg', 'jpeg', 'webp']:
            return jsonify({'error': f'Formato no soportado: {output_format}. Use: png, jpg, jpeg, webp'}), 400
        
        # Claves especiales que no son opciones de procesamiento
        special_keys = {'image_url', 'image_base64', 'pdf_url', 'pdf_base64', 'preset', 'format'}
        
        # Extraer todas las opciones directamente del nivel raíz
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
                options = {
                    'smart_table_analysis': True,
                    'upscale': True,
                    'min_size': 1000,
                    'max_scale': 3.0,
                    'deblur': True,
                    'deblur_method': 'unsharp',
                }
            elif preset == 'small_text_sharp':
                # Para DETECCIÓN DE ESTRUCTURA de tabla (bordes, líneas)
                # ADVERTENCIA: Puede engrosar trazos y perder detalles finos para OCR
                options = {
                    'smart_table_analysis': True,
                    'upscale': True,
                    'min_size': 2000,
                    'max_scale': 5.0,
                    'upscale_method': 'lanczos4',
                    'deblur': True,
                    'deblur_method': 'aggressive',
                    'deblur_strength': 1.0,
                    'sharpen': True,
                    'sharpen_strength': 0.5,
                    'sharpen_method': 'unsharp',
                    'preserve_fine_details': True,
                }
            elif preset == 'ocr_preserve_details':
                # NUEVO: Para OCR de texto/números - PRESERVA detalles finos
                # Optimizado para mantener comas, puntos, símbolos (<, %, *, etc.)
                # Menos procesamiento = menos distorsión
                options = {
                    'smart_table_analysis': True,
                    'upscale': True,
                    'min_size': 1800,
                    'max_scale': 4.0,
                    'upscale_method': 'lanczos4',
                    'deblur': True,
                    'deblur_method': 'unsharp',  # Menos agresivo que aggressive
                    'deblur_strength': 0.6,  # Muy suave
                    'sharpen': False,  # Sin sharpening extra para no engrosar
                    'preserve_fine_details': True,
                }
            elif preset == 'ocr_ultra_fine':
                # NUEVO: Ultra-optimizado con parámetros granulares
                options = {
                    'smart_table_analysis': False,
                    'upscale': True,
                    'min_size': 2400,
                    'max_scale': 3.0,
                    'upscale_method': 'lanczos4',
                    'denoise': True,
                    'denoise_method': 'bilateral',
                    'bilateral_d': 5,
                    'bilateral_sigma_color': 50,
                    'bilateral_sigma_space': 50,
                    'enhance_contrast': True,
                    'clip_limit': 1.8,
                    'clahe_tile_grid_size': [8, 8],
                    'remove_color_bg': True,
                    'deblur': True,
                    'deblur_method': 'unsharp',
                    'deblur_strength': 0.35,
                    'sharpen': False,
                    'binarize': True,
                    'binarize_method': 'adaptive_gaussian',
                    'adaptive_block_size': 51,
                    'adaptive_C': 9,
                    'post_morphology': True,
                    'morphology_mode': 'open',
                    'morphology_kernel': [2, 2],
                    'morphology_iterations': 1,
                    'preserve_fine_details': True,
                    'deskew': False,
                    'auto_invert': False,
                }
            elif preset == 'gemini_vision':
                # Base: Escala de grises + filtros MUY suaves para nitidez
                # Optimizado para Gemini 2.5 Pro (mantiene naturalidad)
                options = {
                    'smart_table_analysis': False,  # Sin análisis automático
                    'upscale': True,
                    'min_size': 2000,
                    'max_scale': 3.0,
                    'upscale_method': 'lanczos4',
                    # Convertir a escala de grises (rojo → gris)
                    'convert_to_grayscale': True,
                    # CLAHE MUY suave (solo mejorar contraste de números finos)
                    'enhance_contrast': True,
                    'clip_limit': 1.5,  # Suave
                    'clahe_tile_grid_size': [8, 8],
                    # Deblur ultra-suave (ayuda con 7,2 vs 7,1)
                    'deblur': True,
                    'deblur_method': 'unsharp',
                    'deblur_strength': 0.3,  # Ultra-suave
                    # Resto DESACTIVADO
                    'denoise': False,
                    'remove_color_bg': False,
                    'sharpen': False,
                    'binarize': False,
                    'post_morphology': False,
                    'deskew': False,
                    'auto_invert': False,
                    'extract_white_text': False,
                    'extract_text_adaptive': False,
                    'preserve_fine_details': True,
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

        # Procesar imagen
        processed, metadata = preprocess_for_table_ocr(img, options)
        
        # Convertir a bytes según el formato solicitado
        if output_format == 'png':
            success, buffer = cv2.imencode('.png', processed)
            mimetype = 'image/png'
            extension = 'png'
        elif output_format in ['jpg', 'jpeg']:
            # Si la imagen es en escala de grises, convertir a BGR para JPEG
            if len(processed.shape) == 2:
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            else:
                processed_bgr = processed
            success, buffer = cv2.imencode('.jpg', processed_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            mimetype = 'image/jpeg'
            extension = 'jpg'
        elif output_format == 'webp':
            success, buffer = cv2.imencode('.webp', processed, [cv2.IMWRITE_WEBP_QUALITY, 95])
            mimetype = 'image/webp'
            extension = 'webp'
        
        if not success:
            return jsonify({'error': 'Error al codificar imagen'}), 500
        
        # Crear BytesIO buffer
        img_io = BytesIO(buffer.tobytes())
        img_io.seek(0)
        
        # Preparar headers con metadata
        response = send_file(
            img_io,
            mimetype=mimetype,
            as_attachment=False,
            download_name=f'processed.{extension}'
        )
        
        # Agregar headers con información adicional
        response.headers['X-Original-Width'] = str(img.shape[1])
        response.headers['X-Original-Height'] = str(img.shape[0])
        response.headers['X-Processed-Width'] = str(processed.shape[1])
        response.headers['X-Processed-Height'] = str(processed.shape[0])
        response.headers['X-Applied-Operations'] = ','.join(metadata.get('applied_operations', []))
        
        if metadata.get('smart_analysis_used'):
            response.headers['X-Smart-Analysis'] = 'true'
            response.headers['X-Strategy'] = metadata.get('strategy', 'unknown')
        
        logger.info(
            "Imagen procesada y devuelta - Original: %dx%d, Procesado: %dx%d, Formato: %s",
            img.shape[1], img.shape[0],
            processed.shape[1], processed.shape[0],
            output_format
        )
        
        return response

    except ValueError as exc:
        logger.warning("Error de validación en preprocess-image: %s", exc)
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        logger.error("Error interno en preprocess-image: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': f'Error interno del servidor: {str(exc)}'}), 500


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
        # PDF sin texto embebido retorna 422 (Unprocessable Entity)
        logger.warning("Error de validación en extract-pdf-fe: %s", exc)
        return jsonify({'success': False, 'error': str(exc)}), 422
    except Exception as exc:
        logger.error("Error interno en extract-pdf-fe: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': f'Error interno del servidor: {str(exc)}'}), 500
