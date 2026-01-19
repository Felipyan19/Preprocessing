import base64
import logging

import cv2
import fitz
import numpy as np
import requests

logger = logging.getLogger(__name__)


def pdf_to_image(pdf_bytes: bytes, dpi: int = 300) -> np.ndarray:
    """Convierte primera pagina de PDF a imagen."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
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

    if is_pdf(img_bytes):
        logger.info("Detectado PDF en base64, convirtiendo a imagen...")
        return pdf_to_image(img_bytes)

    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


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
