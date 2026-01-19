import logging
import re
import unicodedata

import fitz

logger = logging.getLogger(__name__)


def _color_to_rgb(color) -> tuple | None:
    if color is None:
        return None
    if isinstance(color, int):
        return ((color >> 16) & 255, (color >> 8) & 255, color & 255)
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        if all(0 <= c <= 1 for c in color[:3]):
            return (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        return (int(color[0]), int(color[1]), int(color[2]))
    return None


def _is_white(color) -> bool:
    rgb = _color_to_rgb(color)
    if rgb is None:
        return False
    r, g, b = rgb
    return r >= 230 and g >= 230 and b >= 230


def _is_blue(color) -> bool:
    rgb = _color_to_rgb(color)
    if rgb is None:
        return False
    r, g, b = rgb
    return b >= 140 and (b - max(r, g)) >= 30


def _collapse_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def _normalize_label_key(text: str) -> str:
    normalized = unicodedata.normalize('NFD', text)
    normalized = ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn')
    return _collapse_spaces(normalized).lower()


def _should_skip_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.match(r'^(Page|Página)\s*:', stripped, re.IGNORECASE):
        return True
    return False


def _is_metadata_line(text: str) -> bool:
    """Detecta si una línea es metadata del documento (ficha, versión, etc.)"""
    text_lower = text.lower()
    metadata_patterns = [
        r'ficha\s+etiquetad',
        r'ref\s*:\s*\d+',
        r'version\s*:\s*[\d,]+',
        r'fecha\s*:\s*\d{2}[/-]\d{2}[/-]\d{4}',
        r'pa\s*:\s*\d+',
        r'fo\s*:\s*\d+',
        r'lang\s*:\s*\(',
        r'page\s*:\s*\d+\s+of\s+\d+',
        r'\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}',
    ]
    return any(re.search(pattern, text_lower) for pattern in metadata_patterns)


def _extract_blue_shapes(page: fitz.Page) -> tuple[list, list]:
    blue_fills = []
    blue_lines = []
    for drawing in page.get_drawings():
        rect = drawing.get('rect')
        if rect is None:
            continue
        fill = drawing.get('fill')
        stroke = drawing.get('color')
        if fill and _is_blue(fill):
            blue_fills.append(fitz.Rect(rect))
        elif stroke and _is_blue(stroke) and fitz.Rect(rect).height <= 3:
            blue_lines.append(fitz.Rect(rect))
    return blue_fills, blue_lines


def _extract_labels_from_page(page: fitz.Page) -> tuple[list, list]:
    blue_fills, blue_lines = _extract_blue_shapes(page)
    text_dict = page.get_text('dict')
    spans = []
    for block in text_dict.get('blocks', []):
        if block.get('type') != 0:
            continue
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                text = span.get('text', '').strip()
                if not text:
                    continue
                spans.append(
                    {
                        'text': text,
                        'bbox': span.get('bbox'),
                        'color': span.get('color'),
                    }
                )

    labels = []
    seen = set()
    for rect in blue_fills:
        rect_expanded = fitz.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 1, rect.y1 + 1)
        spans_in = []
        for span in spans:
            if not span.get('bbox'):
                continue
            if rect_expanded.intersects(fitz.Rect(span['bbox'])) and _is_white(span.get('color')):
                spans_in.append(span)
        if not spans_in:
            continue
        spans_in.sort(key=lambda s: (s['bbox'][1], s['bbox'][0]))
        text = _collapse_spaces(' '.join(s['text'] for s in spans_in))
        if not text:
            continue
        x0 = min(span['bbox'][0] for span in spans_in)
        y0 = min(span['bbox'][1] for span in spans_in)
        x1 = max(span['bbox'][2] for span in spans_in)
        y1 = max(span['bbox'][3] for span in spans_in)
        text_bbox = (x0, y0, x1, y1)
        key = (text, round(rect_expanded.y0, 1))
        if key in seen:
            continue
        seen.add(key)
        labels.append(
            {'text': text, 'rect': rect_expanded, 'text_bbox': text_bbox, 'forced_value': None}
        )
    return labels, blue_lines


def _extract_page_lines(page: fitz.Page) -> list:
    words = page.get_text('words')
    lines_map = {}
    for x0, y0, x1, y1, word, block_no, line_no, _ in words:
        key = (block_no, line_no)
        lines_map.setdefault(key, []).append((x0, y0, x1, y1, word))
    lines = []
    for (block_no, line_no), items in lines_map.items():
        items.sort(key=lambda w: w[0])
        text = ' '.join(item[4] for item in items).strip()
        if not text:
            continue
        x0 = min(item[0] for item in items)
        y0 = min(item[1] for item in items)
        x1 = max(item[2] for item in items)
        y1 = max(item[3] for item in items)
        lines.append(
            {
                'block': block_no,
                'line': line_no,
                'text': text,
                'bbox': (x0, y0, x1, y1),
            }
        )
    lines.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))
    return lines


def _lines_in_region(lines: list, region: fitz.Rect, label_rects: list, stop_at_metadata: bool = True) -> list:
    filtered = []
    for line in lines:
        rect = fitz.Rect(line['bbox'])
        if not rect.intersects(region):
            continue
        if any(rect.intersects(label_rect) for label_rect in label_rects):
            continue
        if _should_skip_line(line['text']):
            continue
        
        # Opcional: detener si encontramos líneas de metadatos
        if stop_at_metadata and _is_metadata_line(line['text']):
            break
        
        filtered.append(line)
    return filtered


def _sorted_lines(lines: list) -> list:
    return sorted(lines, key=lambda l: (l['bbox'][1], l['bbox'][0]))


def _median(values: list) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    mid = len(values_sorted) // 2
    if len(values_sorted) % 2 == 0:
        return (values_sorted[mid - 1] + values_sorted[mid]) / 2
    return values_sorted[mid]


def _paragraphs_from_lines(lines: list, max_gap_multiplier: float = 1.5) -> list:
    ordered = _sorted_lines(lines)
    if not ordered:
        return []
    heights = [max(1.0, line['bbox'][3] - line['bbox'][1]) for line in ordered]
    median_height = _median(heights) or 1.0
    gap_threshold = median_height * max_gap_multiplier

    paragraphs = []
    current = [ordered[0]['text']]
    prev_y1 = ordered[0]['bbox'][3]

    for line in ordered[1:]:
        gap = line['bbox'][1] - prev_y1
        if gap > gap_threshold:
            paragraphs.append(current)
            current = [line['text']]
        else:
            current.append(line['text'])
        prev_y1 = line['bbox'][3]
    paragraphs.append(current)
    return paragraphs


def _paragraphs_to_text(paragraphs: list) -> str:
    blocks = ['\n'.join(lines) for lines in paragraphs if lines]
    return '\n\n'.join(blocks).strip()


def _paragraphs_to_list_items(paragraphs: list) -> list:
    items = []
    for lines in paragraphs:
        item = _collapse_spaces(' '.join(lines))
        if item:
            items.append(item)
    return items


def _group_lines_by_block(lines: list) -> list:
    blocks = {}
    for line in lines:
        blocks.setdefault(line['block'], []).append(line)
    result = []
    for block_no, block_lines in blocks.items():
        block_lines.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))
        text = '\n'.join(line['text'] for line in block_lines).strip()
        if not text:
            continue
        x0 = min(line['bbox'][0] for line in block_lines)
        y0 = min(line['bbox'][1] for line in block_lines)
        x1 = max(line['bbox'][2] for line in block_lines)
        y1 = max(line['bbox'][3] for line in block_lines)
        result.append(
            {
                'block': block_no,
                'text': text,
                'bbox': (x0, y0, x1, y1),
                'lines': block_lines,
            }
        )
    result.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))
    return result


def _split_paragraphs(text: str) -> list:
    parts = [part.strip() for part in re.split(r'\n\s*\n', text) if part.strip()]
    return parts


def _label_pattern(label_text: str) -> str:
    tokens = re.split(r'\s+', label_text.strip())
    return r'\\s+'.join(re.escape(token) for token in tokens if token)


def _extract_text_segments(page_text: str, labels: list) -> dict:
    positions = []
    for label in labels:
        label_text = label['text']
        pattern = _label_pattern(label_text)
        if not pattern:
            continue
        match = re.search(pattern, page_text)
        if not match:
            continue
        positions.append((match.start(), match.end(), label_text))
    positions.sort(key=lambda item: item[0])

    segments = {}
    for idx, (start, end, label_text) in enumerate(positions):
        next_start = positions[idx + 1][0] if idx + 1 < len(positions) else len(page_text)
        value = page_text[end:next_start].strip()
        segments[label_text] = value
    return segments


def _lines_from_text(value_text: str) -> list:
    lines = []
    y = 0.0
    for raw in value_text.splitlines():
        if not raw.strip():
            y += 20.0
            continue
        text = raw.strip()
        lines.append({'text': text, 'bbox': (0.0, y, 1.0, y + 10.0)})
        y += 12.0
    return lines


def _looks_like_label(text: str) -> bool:
    """Detecta si el texto parece ser el nombre de una etiqueta (no un valor)."""
    text_stripped = text.strip()
    if len(text_stripped) < 3:
        return False
    
    # Nombres comunes de etiquetas
    label_patterns = [
        r'^denominaci[oó]n\s+(comercial|legal)',
        r'^cantidad\s+neta',
        r'^fecha\s+de\s+consumo',
        r'^conservaci[oó]n',
        r'^raz[oó]n\s+social',
        r'^ingredientes',
        r'^informaci[oó]n\s+nutricional',
        r'^textos\s+comerciales',
        r'^marca\s+de\s+identificaci[oó]n',
    ]
    
    text_lower = text.lower()
    return any(re.match(pattern, text_lower) for pattern in label_patterns)


def _looks_like_different_section(text: str, current_label_key: str) -> bool:
    """Detecta si el texto parece pertenecer a una sección diferente."""
    text_lower = text.lower()
    text_normalized = _normalize_label_key(text)
    text_stripped = text.strip()
    
    # Si la línea está vacía, no es una sección diferente
    if not text_stripped:
        return False
    
    # 1. Detectar si es metadata del documento
    if _is_metadata_line(text):
        return True
    
    # 2. Detectar si el texto es exactamente una etiqueta conocida
    if _looks_like_label(text):
        return True
    
    # 3. Reglas específicas por tipo de etiqueta actual
    if current_label_key in ('ingredientes', 'ingredients'):
        # Ingredientes termina cuando empieza otra etiqueta
        markers = ['cantidad neta', 'fecha de consumo', 'conservacion', 'conservação']
        if any(text_normalized.startswith(_normalize_label_key(m)) for m in markers):
            return True
    
    elif current_label_key in ('cantidad neta', 'net quantity'):
        # Cantidad neta es usualmente corta, termina cuando empieza otra etiqueta
        markers = ['fecha de consumo', 'conservacion', 'ingredientes']
        if any(text_normalized.startswith(_normalize_label_key(m)) for m in markers):
            return True
    
    elif current_label_key in ('fecha de consumo', 'consumo preferente', 'best before'):
        # Fecha de consumo termina cuando empieza conservación
        markers = ['importante:', 'conservar', 'conserver', 'keep']
        # Pero solo si la línea EMPIEZA con estos marcadores
        if any(text_normalized.startswith(_normalize_label_key(m)) for m in markers):
            return True
    
    elif current_label_key in ('conservacion', 'conservacao', 'conservation'):
        # Conservación termina cuando empieza razón social o info nutricional
        markers = ['razon social', 'razón social', 'lp foodies', 'informacion nutricional', 'informação nutricional', 'ctra.', 'es 10.', 'es  10.']
        if any(text_normalized.startswith(_normalize_label_key(m)) for m in markers):
            return True
        # O si detectamos una empresa específica (razón social)
        if re.match(r'^[A-Z\s\.]+S\.A\.U\.|^[A-Z\s\.]+S\.L\.|^[A-Z\s\.]+S\.A\.', text_stripped):
            return True
    
    elif current_label_key in ('razon social', 'razón social'):
        # Razón social termina cuando empieza marca de identificación o info nutricional
        markers = ['es 10.', 'es  10.', 'ce', 'informacion nutricional', 'informação nutricional']
        # Razón social es corta, detener en marca CE o info nutricional
        if 'informacion nutricional' in text_normalized or 'informação nutricional' in text_normalized:
            return True
    
    elif 'gda' in current_label_key:
        # GDA termina cuando empiezan textos comerciales
        markers = ['textos comerciales', 'punto verde', 'logo']
        if any(text_normalized.startswith(_normalize_label_key(m)) for m in markers):
            return True
    
    return False


def _derive_list_items(label_key: str, blocks: list, lines: list) -> list:
    line_items_labels = {
        'textos comerciales',
        'telefono/contacto',
        'teléfono/contacto',
    }
    # Etiquetas que SIEMPRE deben ser lista (según prompt OCR)
    # Solo si tienen 2+ ítems separados visualmente
    force_list_labels = {
        'denominacion legal',  # ES y PT en líneas separadas
        'denominação legal',
        'denomination legal',
        'denomination legale',
        'ingredientes',  # ES y PT como bloques separados
        'ingredients',
        'ingrédients',
        'fecha de consumo',  # ES y PT
        'fecha de consumo preferente',
        'best before',
        'consumo preferente',
        'conservacion',  # ES y PT
        'conservacao',
        'conservação',
        'conservation',
        'textos comerciales',  # muchas líneas
        'telefono/contacto',
        'teléfono/contacto',
        'telefono / contacto',
    }
    # Para textos comerciales y teléfono: cada línea es un ítem
    if label_key in line_items_labels:
        items = [line['text'] for line in _sorted_lines(lines) if line['text'].strip()]
        # REGLA OCR: Solo devolver lista si hay 2+ ítems
        return items if len(items) >= 2 else []
    
    # Usar un umbral más estricto para detectar gaps entre párrafos
    paragraphs = _paragraphs_from_lines(lines, max_gap_multiplier=1.3)
    
    # Para etiquetas que DEBEN ser lista (si tienen 2+ ítems)
    if label_key in force_list_labels:
        if paragraphs:
            # Filtrar párrafos que pertenecen a otra sección
            filtered_paragraphs = []
            for i, para in enumerate(paragraphs):
                para_text = ' '.join(para).strip()
                # Detener si encontramos texto de otra sección
                if i > 0 and _looks_like_different_section(para_text, label_key):
                    break
                filtered_paragraphs.append(para)
            
            if filtered_paragraphs:
                items = _paragraphs_to_list_items(filtered_paragraphs)
                # REGLA OCR: Solo devolver lista si hay 2+ ítems
                if len(items) >= 2:
                    return items
    
    # Heurística: si hay 2+ bloques separados
    if len(blocks) >= 2:
        items = [block['text'] for block in blocks if block['text'].strip()]
        if len(items) >= 2:
            return items
    
    # Heurística: si hay 2+ párrafos
    if paragraphs and len(paragraphs) >= 2:
        items = _paragraphs_to_list_items(paragraphs)
        if len(items) >= 2:
            return items
    
    # Heurística: si un bloque tiene múltiples párrafos separados
    if blocks and len(blocks) == 1:
        parts = _split_paragraphs(blocks[0]['text'])
        if len(parts) >= 2:
            return parts
    
    # Heurística: si hay muchas líneas cortas (>= 3 líneas)
    if len(lines) >= 3:
        avg_len = sum(len(line['text']) for line in lines) / len(lines)
        if avg_len <= 50:
            items = [line['text'] for line in _sorted_lines(lines) if line['text'].strip()]
            if len(items) >= 2:
                return items
    
    # REGLA OCR: Si no hay 2+ ítems separados, NO es lista
    return []


def _build_text_value(blocks: list, lines: list) -> str:
    """
    Construye valor de texto conservando saltos de línea.
    REGLA OCR: Para Razón social, GDA, etc. conservar saltos de línea internos.
    """
    paragraphs = _paragraphs_from_lines(lines)
    if paragraphs:
        return _paragraphs_to_text(paragraphs)
    if blocks:
        # Conservar saltos de línea entre bloques
        return '\n\n'.join(block['text'] for block in blocks if block['text'].strip()).strip()
    # Unir líneas con saltos de línea simples
    return '\n'.join(line['text'] for line in _sorted_lines(lines) if line['text'].strip()).strip()


def _table_headers_from_lines(lines: list) -> list:
    text = ' '.join(line['text'] for line in lines).lower()
    text_norm = _normalize_label_key(text)
    headers = ['']
    has_100 = 'por 100' in text_norm or 'per 100' in text_norm or 'par 100' in text_norm
    has_vrn = '%vrn' in text_norm or 'vrn' in text_norm or 'nrvs' in text_norm or 'vnr' in text_norm
    has_serving = (
        'por racion' in text_norm
        or 'por dose' in text_norm
        or 'per serving' in text_norm
        or 'par portion' in text_norm
    )
    has_ir = '%ir' in text_norm or '%dr' in text_norm or '%ri' in text_norm or '%ar' in text_norm

    if has_100:
        header_100 = 'Por 100 g/ Por 100 g'
        if 'per 100' in text_norm or 'par 100' in text_norm:
            header_100 = 'Por 100 g/ Por 100 g/ Per 100 g/ Par 100 g'
        headers.append(header_100)
    if has_vrn:
        header_vrn = '%VRN*/%VRN*'
        if 'nrvs' in text_norm or 'vnr' in text_norm:
            header_vrn = '%VRN*/%VRN*/%NRVs*/%VNR*'
        headers.append(header_vrn)
    if has_serving:
        headers.append('Por ración*/ Por dose*')
    if has_ir:
        headers.append('%IR**/%DR**')
    if len(headers) == 1:
        headers = [
            '',
            'Por 100 g/ Por 100 g/ Per 100 g/ Par 100 g',
            '%VRN*/%VRN*/%NRVs*/%VNR*',
            'Por ración*/ Por dose*',
            '%IR**/%DR**',
        ]
    return headers


def _is_table_note_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith('*'):
        return True
    return bool(
        re.search(
            r'(ración recomendada|dose recomendada|ingesta de referencia|valores de referencia|vrn|%ir|%dr|contiene .*raciones|contém .*doses)',
            stripped,
            re.IGNORECASE,
        )
    )


def _is_nutritional_data_line(text: str) -> bool:
    """Detecta si una línea contiene datos nutricionales."""
    text_lower = text.lower()
    
    # Marcadores de filas nutricionales
    nutrient_markers = [
        'valor energético', 'valor energetico', 'energia', 'energy',
        'grasas', 'lípidos', 'lipidos', 'fat', 'matières grasses',
        'saturadas', 'saturados', 'saturated',
        'hidratos de carbono', 'carbohydrate', 'glucides',
        'azúcares', 'azucares', 'açúcares', 'acucares', 'sugar', 'sucres',
        'proteínas', 'proteinas', 'protein', 'protéines',
        'sal', 'salt', 'sel',
        'hierro', 'ferro', 'iron', 'fer',
        'fibra', 'fiber', 'fibre',
    ]
    
    # Verificar si contiene algún marcador de nutriente
    has_nutrient = any(marker in text_lower for marker in nutrient_markers)
    
    # Verificar si contiene números (datos nutricionales)
    has_numbers = bool(re.search(r'\d', text))
    
    return has_nutrient and has_numbers


def _build_table_value(label_text: str, lines: list) -> dict:
    """Construye el valor de tabla nutricional con raw y notas."""
    if not lines:
        return {
            'titulo': label_text,
            'headers': _table_headers_from_lines([]),
            'raw': None,
            'valorComplementario': None,
        }
    
    # Separar líneas en: encabezados, datos, notas
    header_lines = []
    data_lines = []
    notes = []
    
    capturing_data = False
    
    for line in lines:
        text = line['text']
        text_lower = text.lower()
        
        # 1. Detectar notas (siempre van al final)
        if _is_table_note_line(text):
            notes.append(text)
            continue
        
        # 2. Detectar línea de encabezado
        if not capturing_data and any(marker in text_lower for marker in ['por 100', 'per 100', '%vrn', '%ir', '%dr', 'por racion', 'por dose']):
            header_lines.append(text)
            capturing_data = True
            continue
        
        # 3. Capturar datos de la tabla
        if capturing_data:
            # Detener si encontramos metadata o nueva sección
            if _is_metadata_line(text):
                break
            # Agregar línea de datos
            data_lines.append(text)
    
    # Construir raw: combinar todas las líneas de datos nutricionales
    raw_text = None
    if data_lines:
        raw_text = '\n'.join(data_lines)
    
    value = {
        'titulo': label_text,
        'headers': _table_headers_from_lines(lines),
        'raw': raw_text,
        'valorComplementario': notes if notes else None,
    }
    if notes:
        value['tipoComplementario'] = 'lista'
    return value


def _detect_identification_mark(page: fitz.Page, blue_lines: list) -> tuple[str, fitz.Rect] | None:
    """Detecta la marca de identificación (ej: ES 10.01807/B CE)"""
    # Buscar en todos los bloques, no solo si hay líneas azules
    for block in page.get_text('blocks'):
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        if not text:
            continue
        compact = _collapse_spaces(text.replace('\n', ' '))
        
        # Patrón más específico para marca de identificación
        # Buscar: ES seguido de números y letras, luego CE
        if re.search(r'\bES\s+\d+\.[\d/]+[A-Z]*\s+CE\b', compact, re.IGNORECASE):
            # Extraer solo la parte de la marca de identificación
            match = re.search(r'(ES\s+\d+\.[\d/]+[A-Z]*\s+CE)', compact, re.IGNORECASE)
            if match:
                mark_text = match.group(1)
                return (mark_text, fitz.Rect(x0, y0, x1, y1))
        
        # Patrón alternativo: ES código CE
        elif re.search(r'\bES\b.*\bCE\b', compact) and re.search(r'\d+\.\d+', compact):
            # Intentar extraer el patrón completo
            lines = text.split('\n')
            mark_parts = []
            for line in lines:
                line_stripped = line.strip()
                if 'ES' in line_stripped or 'CE' in line_stripped or re.search(r'\d+\.\d+', line_stripped):
                    mark_parts.append(line_stripped)
            if mark_parts:
                mark_text = ' '.join(mark_parts)
                return (mark_text, fitz.Rect(x0, y0, x1, y1))
    
    return None


def _categoria_for_label(label_text: str) -> str:
    key = _normalize_label_key(label_text)
    if (
        'informacion nutricional' in key
        or 'informacao nutricional' in key
        or 'average nutrition' in key
        or 'information nutritionnelle' in key
        or 'declaracao nutricional' in key
        or key.startswith('gda')
    ):
        return 'NUTRICIONAL'
    if 'ingrediente' in key or 'alergen' in key:
        return 'INGREDIENTES'
    if 'fecha de consumo' in key or 'consumo preferente' in key or 'caducidad' in key:
        return 'CONSUMO_PREFERENTE'
    return 'INFO_GENERAL'


def _detect_languages(text: str) -> list:
    lowered = text.lower()
    normalized = _normalize_label_key(text)
    es_markers = [
        'denominación',
        'denominacion',
        'razón social',
        'razon social',
        'conservación',
        'conservacion',
        'consumir',
        'ingredientes',
        'ración',
        'racion',
        'cantidad neta',
        'consumo preferente',
    ]
    pt_markers = [
        'denominação',
        'denominacao',
        'razao social',
        'razão social',
        'conservação',
        'conservacao',
        'consumir',
        'ingredientes',
        'dose',
        'contém',
        'contem',
        'porção',
        'porcao',
    ]
    en_markers = [
        'ingredients',
        'best before',
        'keep in a cool',
        'net quantity',
        'nutrition information',
        'energy',
        'fat',
        'carbohydrate',
        'salt',
        'protein',
    ]
    fr_markers = [
        'ingredients',
        'a consommer',
        'a conserver',
        'information nutritionnelle',
        'valeurs nutritionnelles',
        'energie',
        'matieres grasses',
        'glucides',
        'sel',
        'proteines',
    ]

    es_score = sum(1 for marker in es_markers if marker in lowered or marker in normalized)
    pt_score = sum(1 for marker in pt_markers if marker in lowered or marker in normalized)
    en_score = sum(1 for marker in en_markers if marker in lowered or marker in normalized)
    fr_score = sum(1 for marker in fr_markers if marker in lowered or marker in normalized)
    idiomas = []
    if es_score >= 2:
        idiomas.append('ES')
    if pt_score >= 2:
        idiomas.append('PT')
    if en_score >= 2:
        idiomas.append('EN')
    if fr_score >= 2:
        idiomas.append('FR')
    return idiomas


def extract_pdf_fe(pdf_bytes: bytes) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    etiquetas = []
    seen_labels = set()
    all_text_parts = []
    title = ''
    total_words = 0
    labels_found = 0
    logger.info("PDF abierto: %d paginas", doc.page_count)

    skip_labels = {
        _normalize_label_key('Instrucciones diseño'),
        _normalize_label_key('Instrucciones de diseño'),
        _normalize_label_key('Instruções design'),
        _normalize_label_key('Instruções de design'),
        _normalize_label_key('Instrucoes de design'),
        _normalize_label_key('Design instructions'),
        _normalize_label_key('Instructions design'),
        _normalize_label_key('Información mínima en embalaje'),
        _normalize_label_key('Informacion minima en embalaje'),
        _normalize_label_key('Informação mínima na embalagem'),
        _normalize_label_key('Informacao minima na embalagem'),
        _normalize_label_key('Minimum packaging information'),
        _normalize_label_key('Minimum information on packaging'),
        _normalize_label_key('Information minimale sur l\'emballage'),
        _normalize_label_key('Informations minimales sur l\'emballage'),
    }
    # Etiquetas que SIEMPRE deben ser texto (según prompt OCR)
    always_text_labels = {
        _normalize_label_key('Razón social'),  # bloque único con saltos de línea
        _normalize_label_key('Razão social'),
        _normalize_label_key('Marca de Identificación'),  # unir líneas con espacios
        _normalize_label_key('Marca de Identificação'),
        _normalize_label_key('GDA'),  # conservar saltos de línea exactos
        _normalize_label_key('Denominación Comercial'),  # valor único
        _normalize_label_key('Denominação Comercial'),
        _normalize_label_key('Cantidad neta'),  # valor único
        _normalize_label_key('Quantidade líquida'),
        _normalize_label_key('Ingredientes a destacar'),  # valor único
    }
    # Etiquetas que típicamente tienen valores inline (a la derecha)
    # REGLA OCR: estas etiquetas suelen tener su valor en la misma línea, a la derecha
    prefer_inline_labels = {
        _normalize_label_key('Denominación Comercial'),  # valor único inline
        _normalize_label_key('Denominação Comercial'),
        _normalize_label_key('Cantidad neta'),  # ej: "75 g x 4"
        _normalize_label_key('Net quantity'),
        _normalize_label_key('Quantidade líquida'),
        _normalize_label_key('Ingredientes a destacar'),  # ej: "Hígado de cerdo"
        _normalize_label_key('Ingredientes a destacar'),
    }
    table_labels = {
        _normalize_label_key('Información nutricional media'),
        _normalize_label_key('Informação nutricional média'),
        _normalize_label_key('Average nutrition information'),
        _normalize_label_key('Information nutritionnelle moyenne'),
    }

    for page_index, page in enumerate(doc, start=1):
        page_words = page.get_text('words')
        total_words += len(page_words)
        logger.info("Pagina %d: words=%d", page_index, len(page_words))
        labels, blue_lines = _extract_labels_from_page(page)
        label_rects = [label['rect'] for label in labels]
        logger.info("Pagina %d: labels_azules=%d", page_index, len(labels))
        page_text = page.get_text('text') or ''
        text_segments = _extract_text_segments(page_text, labels) if labels else {}

        mark = _detect_identification_mark(page, blue_lines)
        if mark:
            mark_text, mark_rect = mark
            labels.append(
                {
                    'text': 'Marca de Identificación',
                    'rect': mark_rect,
                    'forced_value': mark_text,
                }
            )
            label_rects.append(mark_rect)
            logger.info("Pagina %d: marca_identificacion_detectada", page_index)

        if page_index == 1 and not title:
            if labels:
                top_label_y = min(label.get('text_bbox', label['rect'])[1] for label in labels)
                title_region = fitz.Rect(0, 0, page.rect.width, top_label_y - 2)
                lines_top = _lines_in_region(_extract_page_lines(page), title_region, [], stop_at_metadata=False)
                title = _collapse_spaces(' '.join(line['text'] for line in lines_top))
                logger.info("Titulo detectado (por region): %s", title or "<vacio>")
            if not title:
                title = _collapse_spaces(page.get_text('text').splitlines()[0]) if page.get_text('text') else ''
                logger.info("Titulo detectado (fallback): %s", title or "<vacio>")

        labels.sort(key=lambda l: (l['rect'].y0, l['rect'].x0))
        lines_all = _extract_page_lines(page)

        for idx, label in enumerate(labels):
            label_text = label['text']
            label_key = _normalize_label_key(label_text)
            if label_key in skip_labels:
                logger.info("Etiqueta omitida por regla: %s (pagina %d)", label_text, page_index)
                continue
            if label_key in seen_labels:
                logger.info("Etiqueta duplicada omitida: %s (pagina %d)", label_text, page_index)
                continue
            seen_labels.add(label_key)

            label_text_bbox = label.get('text_bbox', label['rect'])
            y_start = max(label['rect'].y1, label_text_bbox[3]) + 2
            if idx + 1 < len(labels):
                next_label = labels[idx + 1]
                next_bbox = next_label.get('text_bbox', next_label['rect'])
                y_end = min(next_label['rect'].y0, next_bbox[1]) - 2
            else:
                y_end = page.rect.height
            region = fitz.Rect(0, y_start, page.rect.width, max(y_start, y_end))
            
            # Estrategia 1: Buscar valores inline (a la derecha de la etiqueta, misma línea)
            # Expandir región inline para capturar mejor el texto
            inline_region = fitz.Rect(
                label['rect'].x1 + 2,  # Más cerca de la etiqueta
                label_text_bbox[1] - 3,  # Un poco más arriba
                page.rect.width,
                label_text_bbox[3] + 3,  # Un poco más abajo
            )
            # NO excluir label_rects para valores inline (permitir captura cerca de la etiqueta)
            inline_lines_raw = []
            for line in lines_all:
                line_rect = fitz.Rect(line['bbox'])
                if inline_region.intersects(line_rect):
                    # No excluir si está cerca de la etiqueta actual (solo excluir otras etiquetas)
                    is_other_label = False
                    for other_label_rect in label_rects:
                        if other_label_rect != label['rect'] and line_rect.intersects(other_label_rect):
                            is_other_label = True
                            break
                    if not is_other_label and not _should_skip_line(line['text']):
                        inline_lines_raw.append(line)
            
            inline_lines = inline_lines_raw
            
            # Verificar si hay contenido inline válido
            has_inline_content = False
            inline_text = ''
            if inline_lines:
                inline_text = ' '.join(line['text'] for line in inline_lines).strip()
                # Contenido inline es válido si:
                # - Tiene al menos 1 carácter (no vacío)
                # - NO parece ser otra etiqueta
                if inline_text and not _looks_like_label(inline_text):
                    has_inline_content = True
                    logger.info(
                        "Etiqueta %s (pagina %d): contenido inline detectado: '%s'",
                        label_text,
                        page_index,
                        inline_text[:50] if len(inline_text) > 50 else inline_text
                    )
                else:
                    logger.info(
                        "Etiqueta %s (pagina %d): texto inline descartado (parece etiqueta o vacío): '%s'",
                        label_text,
                        page_index,
                        inline_text[:50] if inline_text else "<vacío>"
                    )
            else:
                logger.info(
                    "Etiqueta %s (pagina %d): NO se encontró texto inline en región",
                    label_text,
                    page_index,
                )
            
            # Estrategia 2: Decidir si usar contenido inline o buscar debajo
            # Algunas etiquetas típicamente tienen valores inline (Denominación Comercial, Cantidad neta, etc.)
            if has_inline_content and label_key in prefer_inline_labels:
                # Usar SOLO el contenido inline para etiquetas que prefieren inline
                region_lines = inline_lines
                blocks = _group_lines_by_block(region_lines)
                logger.info(
                    "Etiqueta %s (pagina %d): usando SOLO contenido inline (prefer_inline_labels)",
                    label_text,
                    page_index,
                )
            elif has_inline_content:
                # Para otras etiquetas, usar inline + contenido debajo
                region_lines = inline_lines
                blocks = _group_lines_by_block(region_lines)
            else:
                # Extraer todas las líneas de la región
                all_region_lines = _lines_in_region(lines_all, region, label_rects, stop_at_metadata=False)
                
                # Filtrar líneas que claramente pertenecen a otra sección
                region_lines = []
                for line in all_region_lines:
                    # Detener si encontramos metadata o inicio de otra sección
                    if _is_metadata_line(line['text']) or _looks_like_different_section(line['text'], label_key):
                        logger.info(
                            "Etiqueta %s (pagina %d): deteniendo en línea: '%s'",
                            label_text,
                            page_index,
                            line['text'][:50]
                        )
                        break
                    region_lines.append(line)
                
                blocks = _group_lines_by_block(region_lines)
            
            # Estrategia 3: Fallback - usar texto segmentado
            if not region_lines:
                segment_text = text_segments.get(label_text, '').strip()
                if segment_text:
                    region_lines = _lines_from_text(segment_text)
                    blocks = []
                    logger.info(
                        "Etiqueta %s (pagina %d): usando texto segmentado (%d chars)",
                        label_text,
                        page_index,
                        len(segment_text),
                    )
                else:
                    logger.info(
                        "Etiqueta %s (pagina %d): sin texto en region, inline ni segmentado",
                        label_text,
                        page_index,
                    )
            logger.info(
                "Etiqueta %s (pagina %d): lines=%d blocks=%d",
                label_text,
                page_index,
                len(region_lines),
                len(blocks),
            )

            if label.get('forced_value'):
                tipo = 'texto'
                valor = label['forced_value']
            elif (
                label_key in table_labels
                or 'informacion nutricional' in label_key
                or 'informacao nutricional' in label_key
                or 'average nutrition' in label_key
                or 'information nutritionnelle' in label_key
                or 'declaracao nutricional' in label_key
            ):
                tipo = 'tabla'
                valor = _build_table_value(label_text, region_lines)
            else:
                list_items = _derive_list_items(label_key, blocks, region_lines)
                if label_key in always_text_labels:
                    tipo = 'texto'
                    valor = _build_text_value(blocks, region_lines)
                elif len(list_items) >= 2:
                    tipo = 'lista'
                    valor = list_items
                else:
                    tipo = 'texto'
                    valor = _build_text_value(blocks, region_lines)

            etiquetas.append(
                {
                    'etiqueta': label_text,
                    'valor': valor,
                    'tipo': tipo,
                    'categoria': _categoria_for_label(label_text),
                    'pagina': page_index,
                }
            )
            labels_found += 1

            if isinstance(valor, list):
                all_text_parts.extend(valor)
            elif isinstance(valor, dict):
                all_text_parts.append(valor.get('titulo', ''))
                if valor.get('valorComplementario'):
                    all_text_parts.extend(valor.get('valorComplementario') or [])
            else:
                all_text_parts.append(str(valor))

    if total_words == 0:
        doc.close()
        logger.warning("PDF sin texto embebido; total_words=0")
        raise ValueError('PDF sin texto embebido; OCR deshabilitado')

    if labels_found == 0:
        doc.close()
        logger.warning("PDF sin etiquetas detectables en texto embebido")
        raise ValueError('No se detectaron etiquetas en texto embebido; OCR deshabilitado')

    doc.close()
    full_text = ' '.join([title] + all_text_parts)
    idiomas = _detect_languages(full_text)

    return {
        'titulo': title,
        'idiomas_detectados': idiomas,
        'etiquetas': etiquetas,
    }
