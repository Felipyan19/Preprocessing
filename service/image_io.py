import base64
import logging

import cv2
import fitz
import numpy as np
import requests

logger = logging.getLogger(__name__)


def pdf_to_image(pdf_bytes: bytes, dpi: int = 300) -> np.ndarray:
    """Convierte primera pagina de PDF a imagen."""
    doc = None
    try:
        logger.info("Abriendo PDF, tamaño: %d bytes", len(pdf_bytes))
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if doc.page_count == 0:
            raise ValueError("El PDF no tiene páginas")
        
        logger.info("PDF abierto, páginas: %d", doc.page_count)
        page = doc[0]
        
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Error al decodificar la imagen del PDF")
        
        logger.info("PDF convertido a imagen: %dx%d", img.shape[1], img.shape[0])
        return img
        
    except Exception as e:
        logger.error("Error al convertir PDF a imagen: %s", str(e))
        raise ValueError(f"Error al procesar PDF: {str(e)}")
    finally:
        if doc:
            doc.close()


def is_pdf(content: bytes) -> bool:
    """Detecta si el contenido es un PDF."""
    return content[:4] == b'%PDF'


def download_image_from_url(url: str) -> np.ndarray:
    """Descarga imagen o PDF desde URL."""
    try:
        logger.info("Descargando desde URL: %s", url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.content
        
        logger.info("Descarga exitosa, tamaño: %d bytes", len(content))

        if is_pdf(content) or url.lower().endswith('.pdf'):
            logger.info("Detectado PDF, convirtiendo a imagen...")
            img = pdf_to_image(content)
            if img is None:
                raise ValueError("Error al convertir PDF a imagen")
            logger.info("PDF convertido exitosamente")
            return img

        logger.info("Decodificando imagen...")
        img_array = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen. Verifica que la URL contenga una imagen válida (jpg, png, etc.)")
        
        logger.info("Imagen decodificada exitosamente: %dx%d", img.shape[1], img.shape[0])
        return img
        
    except requests.exceptions.Timeout:
        logger.error("Timeout al descargar desde URL: %s", url)
        raise ValueError(f"Timeout al descargar la imagen (>30s). URL: {url}")
    except requests.exceptions.RequestException as e:
        logger.error("Error al descargar URL: %s - %s", url, str(e))
        raise ValueError(f"Error al descargar la imagen: {str(e)}")
    except Exception as e:
        logger.error("Error inesperado al procesar URL: %s - %s", url, str(e))
        raise


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decodifica imagen o PDF base64."""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        logger.info("Decodificando base64...")
        img_bytes = base64.b64decode(base64_string)
        logger.info("Base64 decodificado, tamaño: %d bytes", len(img_bytes))

        if is_pdf(img_bytes):
            logger.info("Detectado PDF en base64, convirtiendo a imagen...")
            img = pdf_to_image(img_bytes)
            if img is None:
                raise ValueError("Error al convertir PDF a imagen")
            logger.info("PDF convertido exitosamente")
            return img

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen base64. Verifica que sea una imagen válida (jpg, png, etc.)")
        
        logger.info("Imagen decodificada exitosamente: %dx%d", img.shape[1], img.shape[0])
        return img
        
    except base64.binascii.Error as e:
        logger.error("Error al decodificar base64: %s", str(e))
        raise ValueError(f"Base64 inválido: {str(e)}")
    except Exception as e:
        logger.error("Error inesperado al decodificar base64: %s", str(e))
        raise


def encode_image_to_base64(img: np.ndarray, fmt: str = 'png') -> str:
    """Codifica imagen a base64."""
    _, buffer = cv2.imencode(f'.{fmt}', img)
    return base64.b64encode(buffer).decode('utf-8')


def download_pdf_from_url(url: str) -> bytes:
    """Descarga PDF desde URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content = response.content
    if not is_pdf(content) and not url.lower().endswith('.pdf'):
        raise ValueError('El recurso no parece ser un PDF')
    return content


def decode_base64_pdf(base64_string: str) -> bytes:
    """Decodifica PDF base64."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    pdf_bytes = base64.b64decode(base64_string)
    if not is_pdf(pdf_bytes):
        raise ValueError('El contenido base64 no parece ser un PDF')
    return pdf_bytes
