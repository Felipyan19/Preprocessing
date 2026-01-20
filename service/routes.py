import logging
import os
from io import BytesIO

import cv2
import numpy as np
from flask import Blueprint, jsonify, request, send_file
from PIL import Image

from .image_io import (
    decode_base64_image,
    download_image_from_url,
    encode_image_to_base64,
)
from .preprocessing import preprocess_for_table_ocr
from .temp_files import cleanup_expired_files, get_temp_file, save_temp_file

logger = logging.getLogger(__name__)

api = Blueprint('api', __name__)


@api.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@api.route('/download/<file_id>', methods=['GET'])
def download_temp_file(file_id: str):
    """
    GET /download/<file_id>
    
    Descarga un archivo temporal procesado.
    Los archivos expiran automáticamente después de un tiempo configurado.
    """
    try:
        logger.info("Solicitud de descarga recibida para file_id: %s", file_id)
        
        # Limpiar archivos expirados periódicamente
        cleanup_expired_files()
        
        # Obtener archivo temporal
        file_path = get_temp_file(file_id)
        
        if file_path is None:
            logger.warning("Archivo temporal no encontrado o expirado: %s", file_id)
            return jsonify({'error': 'Archivo no encontrado o expirado'}), 404
        
        if not file_path.exists():
            logger.error("Ruta de archivo no existe: %s (absoluta: %s)", file_path, file_path.absolute())
            return jsonify({'error': f'Archivo no encontrado: {file_path}'}), 404
        
        logger.info("Archivo encontrado: %s (absoluta: %s)", file_path, file_path.absolute())
        
        # Determinar mimetype según extensión
        extension = file_path.suffix.lower()
        mimetypes = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.pdf': 'application/pdf',
        }
        mimetype = mimetypes.get(extension, 'application/octet-stream')
        
        # Enviar archivo
        return send_file(
            str(file_path),
            mimetype=mimetype,
            as_attachment=True,
            download_name=file_path.name
        )
    
    except Exception as exc:
        logger.error("Error al descargar archivo temporal %s: %s", file_id, exc, exc_info=True)
        return jsonify({'error': f'Error al descargar archivo: {str(exc)}'}), 500


@api.route('/preprocess', methods=['POST'])
def preprocess():
    """
    POST /preprocess
    Aplica rotación automática para dejar la tabla derecha.
    
    Body: {
        "image_url": "https://...",     # o
        "image_base64": "base64...",
        "preset": "default",            # Presets disponibles:
                                        #   - default (rotación automática con deskew)
                                        #   - rotation_only (solo rotación, sin deskew)
        
        # Opciones de rotación automática:
        "enable_deskew": true,          # Aplicar corrección de inclinación pequeña (default: true)
        "deskew_max_abs_deg": 8.0,      # Máximo ángulo de deskew en grados (default: 8.0)
        "min_improvement": 0.0         # Mejora mínima requerida para aplicar rotación (default: 0.0 - máximo agresivo)
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

        preset = data.get('preset', 'default')

        # Claves especiales que no son opciones de procesamiento
        special_keys = {'image_url', 'image_base64', 'preset'}

        # Extraer opciones personalizadas del usuario (excluyendo claves especiales)
        user_options = {k: v for k, v in data.items() if k not in special_keys}

        # Cargar preset primero
        options = {}
        if preset == 'default':
            # Rotación automática con deskew habilitado
            options = {
                'enable_deskew': True,
                'deskew_max_abs_deg': 8.0,
                'min_improvement': 0.0,
            }
        elif preset == 'rotation_only':
            # Solo rotación automática, sin deskew
            options = {
                'enable_deskew': False,
                'deskew_max_abs_deg': 8.0,
                'min_improvement': 0.0,
            }
        else:
            # Si el preset no existe, usar valores por defecto
            options = {
                'enable_deskew': True,
                'deskew_max_abs_deg': 8.0,
                'min_improvement': 0.0,
            }

        # Sobrescribir opciones del preset con las opciones personalizadas del usuario
        options.update(user_options)

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


@api.route('/preprocess-image', methods=['POST'])
def preprocess_image():
    """
    POST /preprocess-image
    
    Procesa la imagen y devuelve una URL temporal para descargarla.
    Aplica rotación automática para dejar la tabla derecha.
    
    Body: {
        "image_url": "https://...",     # o
        "image_base64": "base64...",
        "preset": "default",            # Presets: default, rotation_only
        "format": "png",                # Formato de salida: "png" (default), "jpg", "jpeg", "webp"
        "dpi": 300,                     # DPI de la imagen de salida (default: 300)
        "return_url": true,             # Si true, devuelve URL temporal; si false, devuelve archivo directo (default: true)
        "expiry_seconds": 3600,        # Tiempo de expiración de la URL en segundos (default: 3600 = 1 hora)
        
        # Opciones de rotación automática:
        "enable_deskew": true,          # Aplicar corrección de inclinación pequeña (default: true)
        "deskew_max_abs_deg": 8.0,      # Máximo ángulo de deskew en grados (default: 8.0)
        "min_improvement": 0.0,         # Mejora mínima requerida para aplicar rotación (default: 0.0 - máximo agresivo)
        
        # Opciones para mejorar detección de tablas (Azure Vision):
        "enhance_table_structure": true, # Mejorar bordes y contraste de tablas (default: false) ⭐ RECOMENDADO PARA AZURE VISION
        "enhance_edges": true,          # Mejorar bordes de tabla (default: true, solo si enhance_table_structure=true)
        "enhance_contrast": true,        # Mejorar contraste (default: true, solo si enhance_table_structure=true)
        "aggressive_enhancement": true,  # Mejora agresiva de bordes (default: false) - Hace bordes más gruesos y visibles
        "detect_rotated_tables": true,  # Detectar y rotar regiones de tabla rotadas 90° dentro de la imagen (default: false) ⭐ IMPORTANTE
        "draw_table_borders": true,     # Detectar estructura de tabla y dibujar bordes explícitos (default: false) ⭐⭐⭐ MUY RECOMENDADO PARA AZURE VISION
        "grayscale": false,             # Convertir a escala de grises (default: false) - Mejora detección de tablas
        "upscale": false,               # Aumentar resolución de imagen (default: false)
        "upscale_factor": 2.0,          # Factor de escala (ej: 2.0 = 2x) (opcional)
        "upscale_width": 6000,          # Ancho objetivo en píxeles (opcional)
        "upscale_height": 8000           # Alto objetivo en píxeles (opcional)
    }
    
    Respuesta (si return_url=true):
    {
        "success": true,
        "download_url": "http://localhost:5050/download/<file_id>",
        "file_id": "<file_id>",
        "expires_in_seconds": 3600,
        "metadata": {
            "original_size": {"w": 1000, "h": 800},
            "processed_size": {"w": 1000, "h": 800},
            "dpi": 300,
            "format": "png",
            "applied_operations": ["rotate_0"]
        }
    }
    
    Respuesta (si return_url=false):
    - Archivo de imagen directamente (comportamiento anterior)
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

        preset = data.get('preset', 'default')
        output_format = data.get('format', 'png').lower()

        # Validar formato
        if output_format not in ['png', 'jpg', 'jpeg', 'webp']:
            return jsonify({'error': f'Formato no soportado: {output_format}. Use: png, jpg, jpeg, webp'}), 400

        # Claves especiales que no son opciones de procesamiento
        special_keys = {'image_url', 'image_base64', 'preset', 'format', 'dpi', 'return_url', 'expiry_seconds'}

        # Extraer opciones personalizadas del usuario
        user_options = {k: v for k, v in data.items() if k not in special_keys}

        # Cargar preset primero
        options = {}
        if preset == 'default':
            # Rotación automática con deskew habilitado
            options = {
                'enable_deskew': True,
                'deskew_max_abs_deg': 8.0,
                'min_improvement': 0.0,
            }
        elif preset == 'rotation_only':
            # Solo rotación automática, sin deskew
            options = {
                'enable_deskew': False,
                'deskew_max_abs_deg': 8.0,
                'min_improvement': 0.0,
            }
        else:
            # Si el preset no existe, usar valores por defecto
            options = {
                'enable_deskew': True,
                'deskew_max_abs_deg': 8.0,
                'min_improvement': 0.0,
            }

        # Sobrescribir opciones del preset con las opciones personalizadas del usuario
        options.update(user_options)
        
        # Log de opciones para debugging
        logger.info("Opciones de preprocesamiento: %s", options)

        # Procesar imagen
        processed, metadata = preprocess_for_table_ocr(img, options)
        
        # Obtener DPI del request (por defecto 300 DPI)
        dpi = int(data.get('dpi', 300))
        
        # Verificar si se debe devolver URL o archivo directo
        return_url = data.get('return_url', True)
        
        # Convertir imagen de OpenCV (BGR) a PIL (RGB)
        if len(processed.shape) == 2:
            # Escala de grises
            pil_img = Image.fromarray(processed, mode='L')
        elif len(processed.shape) == 3:
            # Color BGR -> RGB
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(processed_rgb, mode='RGB')
        else:
            return jsonify({'error': 'Formato de imagen no soportado'}), 500
        
        # Establecer DPI en la imagen
        pil_img.info['dpi'] = (dpi, dpi)
        
        # Guardar imagen en BytesIO con DPI
        img_io = BytesIO()
        
        if output_format == 'png':
            pil_img.save(img_io, format='PNG', dpi=(dpi, dpi))
            mimetype = 'image/png'
            extension = 'png'
        elif output_format in ['jpg', 'jpeg']:
            # Convertir a RGB si es escala de grises para JPEG
            if pil_img.mode == 'L':
                pil_img = pil_img.convert('RGB')
            pil_img.save(img_io, format='JPEG', dpi=(dpi, dpi), quality=95)
            mimetype = 'image/jpeg'
            extension = 'jpg'
        elif output_format == 'webp':
            pil_img.save(img_io, format='WEBP', dpi=(dpi, dpi), quality=95)
            mimetype = 'image/webp'
            extension = 'webp'
        
        img_io.seek(0)
        file_data = img_io.getvalue()
        
        # Si return_url es True, guardar archivo temporal y devolver URL
        if return_url:
            # Obtener tiempo de expiración
            expiry_seconds = data.get('expiry_seconds')
            if expiry_seconds is not None:
                expiry_seconds = int(expiry_seconds)
            
            # Guardar archivo temporal
            file_id, file_path = save_temp_file(file_data, extension, expiry_seconds)
            
            # Construir URL de descarga
            # Usar request.host_url para obtener la URL base del servidor
            base_url = request.host_url.rstrip('/')
            download_url = f"{base_url}/download/{file_id}"
            
            logger.info(
                "Imagen procesada y guardada temporalmente - Original: %dx%d, Procesado: %dx%d, Formato: %s, DPI: %d, File ID: %s",
                img.shape[1], img.shape[0],
                processed.shape[1], processed.shape[0],
                output_format, dpi, file_id
            )
            
            # Devolver JSON con URL temporal
            return jsonify({
                'success': True,
                'download_url': download_url,
                'file_id': file_id,
                'expires_in_seconds': expiry_seconds if expiry_seconds is not None else 3600,
                'metadata': {
                    'original_size': {'w': img.shape[1], 'h': img.shape[0]},
                    'processed_size': {'w': processed.shape[1], 'h': processed.shape[0]},
                    'dpi': dpi,
                    'format': output_format,
                    'applied_operations': metadata.get('applied_operations', []),
                    'rotation_metadata': metadata.get('rotation_metadata', {})
                }
            })
        
        # Comportamiento anterior: devolver archivo directo
        img_io.seek(0)
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
        response.headers['X-DPI'] = str(dpi)
        response.headers['X-Applied-Operations'] = ','.join(metadata.get('applied_operations', []))
        
        # Agregar información de rotación si está disponible
        rotation_meta = metadata.get('rotation_metadata', {})
        if rotation_meta.get('auto_rotation_applied'):
            response.headers['X-Rotation-Degrees'] = str(rotation_meta.get('rotation_degrees', 0))
        if rotation_meta.get('deskew_applied'):
            response.headers['X-Deskew-Angle'] = str(rotation_meta.get('deskew_angle_deg', 0.0))
        
        logger.info(
            "Imagen procesada y devuelta - Original: %dx%d, Procesado: %dx%d, Formato: %s, DPI: %d",
            img.shape[1], img.shape[0],
            processed.shape[1], processed.shape[0],
            output_format, dpi
        )
        
        return response

    except ValueError as exc:
        logger.warning("Error de validación en preprocess-image: %s", exc)
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        logger.error("Error interno en preprocess-image: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': f'Error interno del servidor: {str(exc)}'}), 500
