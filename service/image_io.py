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


def extract_images_with_headers(pdf_bytes: bytes, header_distance_threshold: float = 50.0) -> list:
    """
    Extrae todas las imágenes de un PDF junto con sus headers (texto arriba de la imagen).
    
    Args:
        pdf_bytes: Contenido del PDF en bytes
        header_distance_threshold: Distancia máxima en puntos para considerar texto como header (default: 50)
    
    Returns:
        Lista de diccionarios con:
        - 'image': np.ndarray (imagen extraída)
        - 'header': str (texto del header)
        - 'page': int (número de página, 0-indexed)
        - 'bbox': dict (bounding box: x0, y0, x1, y1)
        - 'width': int (ancho de la imagen)
        - 'height': int (alto de la imagen)
    """
    doc = None
    images_with_headers = []
    
    try:
        logger.info("Extrayendo imágenes del PDF, tamaño: %d bytes", len(pdf_bytes))
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if doc.page_count == 0:
            raise ValueError("El PDF no tiene páginas")
        
        logger.info("PDF abierto, páginas: %d", doc.page_count)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            logger.info("Procesando página %d/%d", page_num + 1, doc.page_count)
            
            # Obtener todas las imágenes de la página
            image_list = page.get_images(full=True)
            
            # Obtener texto de la página con sus posiciones
            text_dict = page.get_text("dict")
            
            # Extraer bloques de texto con sus posiciones
            text_blocks = []
            if "blocks" in text_dict:
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                for span in line["spans"]:
                                    if "text" in span and "bbox" in span:
                                        text_blocks.append({
                                            "text": span["text"].strip(),
                                            "bbox": span["bbox"],  # [x0, y0, x1, y1]
                                            "y0": span["bbox"][1],  # Coordenada Y superior
                                            "y1": span["bbox"][3],  # Coordenada Y inferior
                                        })
            
            # Procesar cada imagen
            for img_index, img_item in enumerate(image_list):
                try:
                    xref = img_item[0]
                    
                    # Obtener la imagen
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Obtener posición de la imagen en la página
                    image_rects = page.get_image_rects(xref)
                    
                    if not image_rects:
                        logger.warning("No se encontró posición para imagen %d en página %d", img_index, page_num)
                        continue
                    
                    # Usar el primer rectángulo (puede haber múltiples instancias)
                    img_rect = image_rects[0]
                    img_x0, img_y0, img_x1, img_y1 = img_rect
                    
                    # Buscar texto que esté arriba de la imagen (header)
                    header_texts = []
                    for text_block in text_blocks:
                        text_y0 = text_block["y0"]
                        text_y1 = text_block["y1"]
                        text_x0 = text_block["bbox"][0]
                        text_x1 = text_block["bbox"][2]
                        
                        # Verificar si el texto está arriba de la imagen
                        # y dentro de un rango horizontal razonable
                        is_above = text_y1 <= img_y0  # El texto termina antes de que empiece la imagen
                        horizontal_overlap = not (text_x1 < img_x0 or text_x0 > img_x1)
                        distance = img_y0 - text_y1 if is_above else float('inf')
                        
                        # Considerar como header si está arriba, se superpone horizontalmente
                        # y está dentro del umbral de distancia
                        if is_above and horizontal_overlap and distance <= header_distance_threshold:
                            header_texts.append({
                                "text": text_block["text"],
                                "distance": distance,
                                "y0": text_y0,
                            })
                    
                    # Ordenar headers por distancia (más cercanos primero) y luego por posición Y
                    header_texts.sort(key=lambda x: (x["distance"], x["y0"]))
                    
                    # Combinar headers cercanos en un solo texto
                    header = ""
                    if header_texts:
                        # Tomar los headers más cercanos (dentro de 20 puntos)
                        close_headers = [h for h in header_texts if h["distance"] <= 20]
                        if close_headers:
                            # Ordenar por posición Y (de arriba hacia abajo)
                            close_headers.sort(key=lambda x: x["y0"])
                            header = " ".join([h["text"] for h in close_headers if h["text"]])
                        else:
                            # Si no hay headers muy cercanos, tomar el más cercano
                            header = header_texts[0]["text"]
                    
                    # Decodificar imagen
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        logger.warning("No se pudo decodificar imagen %d en página %d", img_index, page_num)
                        continue
                    
                    # Agregar información de la imagen
                    images_with_headers.append({
                        "image": img,
                        "header": header,
                        "page": page_num,
                        "bbox": {
                            "x0": float(img_x0),
                            "y0": float(img_y0),
                            "x1": float(img_x1),
                            "y1": float(img_y1),
                        },
                        "width": img.shape[1],
                        "height": img.shape[0],
                        "format": image_ext,
                    })
                    
                    logger.info(
                        "Imagen extraída: página %d, tamaño %dx%d, header: '%s'",
                        page_num + 1, img.shape[1], img.shape[0], header[:50] if header else "(sin header)"
                    )
                    
                except Exception as e:
                    logger.error("Error al procesar imagen %d en página %d: %s", img_index, page_num, str(e))
                    continue
        
        logger.info("Extracción completada: %d imágenes encontradas", len(images_with_headers))
        return images_with_headers
        
    except Exception as e:
        logger.error("Error al extraer imágenes del PDF: %s", str(e))
        raise ValueError(f"Error al procesar PDF: {str(e)}")
    finally:
        if doc:
            doc.close()
