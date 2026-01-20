# @service/temp_files.py
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Directorio temporal para archivos
# Usar ruta absoluta basada en el directorio del módulo o variable de entorno
if 'TEMP_FILES_DIR' in os.environ:
    TEMP_DIR = Path(os.environ['TEMP_FILES_DIR']).resolve()
else:
    # Crear directorio temporal en el directorio raíz del proyecto (al mismo nivel que service/)
    # Obtener el directorio del módulo actual (service/)
    module_dir = Path(__file__).parent.resolve()
    # Subir un nivel para llegar al directorio raíz del proyecto
    project_root = module_dir.parent
    TEMP_DIR = (project_root / 'temp_files').resolve()

# Crear directorio si no existe (usar resolve() para asegurar ruta absoluta)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
logger.info("Directorio temporal configurado: %s (absoluto: %s)", TEMP_DIR, TEMP_DIR.absolute())

# Tiempo de expiración por defecto (en segundos): 1 hora
DEFAULT_EXPIRY_SECONDS = int(os.environ.get('TEMP_FILE_EXPIRY', 3600))


def save_temp_file(file_data: bytes, extension: str, expiry_seconds: Optional[int] = None) -> Tuple[str, str]:
    """
    Guarda un archivo temporal y retorna el ID y la ruta completa.
    
    Args:
        file_data: Datos del archivo en bytes
        extension: Extensión del archivo (sin punto, ej: 'png', 'pdf')
        expiry_seconds: Tiempo de expiración en segundos (None = usar default)
    
    Returns:
        (file_id, file_path) - ID único y ruta completa del archivo
    """
    # Asegurar que el directorio existe
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generar ID único
    file_id = str(uuid.uuid4())
    
    # Nombre del archivo
    filename = f"{file_id}.{extension}"
    file_path = TEMP_DIR / filename
    
    # Guardar archivo
    try:
        with open(file_path, 'wb') as f:
            f.write(file_data)
        logger.info("Archivo temporal guardado exitosamente: %s (tamaño: %d bytes)", file_path.absolute(), len(file_data))
    except Exception as e:
        logger.error("Error al guardar archivo temporal %s: %s", file_path.absolute(), e)
        raise
    
    # Guardar timestamp de creación para limpieza
    timestamp_file = TEMP_DIR / f"{file_id}.timestamp"
    try:
        with open(timestamp_file, 'w') as f:
            expiry = expiry_seconds if expiry_seconds is not None else DEFAULT_EXPIRY_SECONDS
            expiry_time = time.time() + expiry
            f.write(str(expiry_time))
        logger.debug("Timestamp guardado: %s (expira en %d segundos)", timestamp_file.absolute(), expiry)
    except Exception as e:
        logger.error("Error al guardar timestamp %s: %s", timestamp_file.absolute(), e)
        # Intentar eliminar el archivo si falla el timestamp
        try:
            file_path.unlink()
        except Exception:
            pass
        raise
    
    return file_id, str(file_path.absolute())


def get_temp_file(file_id: str) -> Optional[Path]:
    """
    Obtiene la ruta del archivo temporal si existe y no ha expirado.
    
    Args:
        file_id: ID del archivo temporal
    
    Returns:
        Path del archivo si existe y es válido, None en caso contrario
    """
    # Asegurar que el directorio existe
    if not TEMP_DIR.exists():
        logger.warning("Directorio temporal no existe: %s", TEMP_DIR.absolute())
        return None
    
    # Buscar archivo con cualquier extensión
    pattern = f"{file_id}.*"
    matches = list(TEMP_DIR.glob(pattern))
    logger.debug("Buscando archivo %s en %s, encontrados: %d archivos", file_id, TEMP_DIR.absolute(), len(matches))
    
    # Filtrar el archivo .timestamp
    file_matches = [m for m in matches if m.suffix != '.timestamp']
    
    if not file_matches:
        logger.warning("Archivo temporal no encontrado: %s en directorio %s", file_id, TEMP_DIR.absolute())
        # Listar archivos disponibles para debug
        if TEMP_DIR.exists():
            all_files = list(TEMP_DIR.glob("*"))
            logger.debug("Archivos en directorio temporal: %s", [f.name for f in all_files[:10]])
        return None
    
    file_path = file_matches[0]
    timestamp_file = TEMP_DIR / f"{file_id}.timestamp"
    
    # Verificar si existe el timestamp
    if not timestamp_file.exists():
        logger.warning("Timestamp no encontrado para archivo: %s", file_id)
        # Eliminar archivo huérfano
        try:
            file_path.unlink()
        except Exception:
            pass
        return None
    
    # Verificar expiración
    try:
        with open(timestamp_file, 'r') as f:
            expiry_time = float(f.read().strip())
        
        if time.time() > expiry_time:
            logger.info("Archivo temporal expirado: %s", file_id)
            # Eliminar archivo expirado
            try:
                file_path.unlink()
                timestamp_file.unlink()
            except Exception:
                pass
            return None
    except Exception as e:
        logger.warning("Error al leer timestamp: %s", e)
        return None
    
    return file_path


def delete_temp_file(file_id: str) -> bool:
    """
    Elimina un archivo temporal y su timestamp.
    
    Args:
        file_id: ID del archivo temporal
    
    Returns:
        True si se eliminó correctamente, False en caso contrario
    """
    try:
        # Buscar y eliminar archivo
        pattern = f"{file_id}.*"
        matches = list(TEMP_DIR.glob(pattern))
        
        deleted = False
        for match in matches:
            try:
                match.unlink()
                deleted = True
            except Exception:
                pass
        
        if deleted:
            logger.info("Archivo temporal eliminado: %s", file_id)
        
        return deleted
    except Exception as e:
        logger.error("Error al eliminar archivo temporal %s: %s", file_id, e)
        return False


def cleanup_expired_files() -> int:
    """
    Limpia archivos temporales expirados.
    
    Returns:
        Número de archivos eliminados
    """
    if not TEMP_DIR.exists():
        return 0
    
    deleted_count = 0
    current_time = time.time()
    
    try:
        # Buscar todos los archivos timestamp
        timestamp_files = list(TEMP_DIR.glob("*.timestamp"))
        
        for timestamp_file in timestamp_files:
            try:
                with open(timestamp_file, 'r') as f:
                    expiry_time = float(f.read().strip())
                
                if current_time > expiry_time:
                    # Archivo expirado, eliminar
                    file_id = timestamp_file.stem
                    
                    # Eliminar archivo principal
                    pattern = f"{file_id}.*"
                    matches = list(TEMP_DIR.glob(pattern))
                    for match in matches:
                        if match != timestamp_file:  # No eliminar el timestamp aquí
                            try:
                                match.unlink()
                            except Exception:
                                pass
                    
                    # Eliminar timestamp
                    try:
                        timestamp_file.unlink()
                        deleted_count += 1
                        logger.debug("Archivo expirado eliminado: %s", file_id)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("Error al procesar timestamp %s: %s", timestamp_file, e)
    
    except Exception as e:
        logger.error("Error durante limpieza de archivos temporales: %s", e)
    
    if deleted_count > 0:
        logger.info("Limpieza completada: %d archivos expirados eliminados", deleted_count)
    
    return deleted_count
