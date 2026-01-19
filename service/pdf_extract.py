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


def _looks_like_different_section(text: str, current_label_key: str) -> bool:
    """Detecta si el texto parece pertenecer a una sección diferente."""
    text_lower = text.lower()
    text_normalized = _normalize_label_key(text)
    
    # Indicadores globales de metadatos que no son contenido de etiqueta
    metadata_markers = [
        'ficha etiquetado', 'ref:', 'fecha:', 'version:',
        'page:', 'página:', 'pagina:',
        'lote', 'pa:', 'fo:', 'lang:',
        'instrucciones', 'diseño', 'diseno',
    ]
    
    # Si el texto contiene marcadores de metadatos
    for marker in metadata_markers:
        if marker in text_lower:
            return True
    
    # Marcadores de inicio de nuevas secciones principales
    new_section_markers = {
        'razon social', 'razón social',
        'informacion nutricional', 'información nutricional',
        'informacao nutricional',
        'ean', 'codigo de barras',
    }
    
    # Si el texto empieza con un marcador de nueva sección
    text_start = text_normalized[:50]  # Primeros 50 caracteres
    for marker in new_section_markers:
        marker_normalized = _normalize_label_key(marker)
        if text_start.startswith(marker_normalized):
            return True
    
    # Reglas específicas por tipo de etiqueta
    if current_label_key in ('ingredientes', 'ingredients', 'ingredientes'):
        # Ingredientes no debe incluir cantidad neta ni fecha de consumo
        if text_normalized.startswith('cantidad neta') or text_normalized.startswith('fecha de consumo'):
            return True
    
    if current_label_key in ('conservacion', 'conservacao', 'conservation'):
        # Conservación no debe incluir razón social ni información nutricional
        if any(text_normalized.startswith(_normalize_label_key(m)) for m in ['razón social', 'razon social', 'lp foodies', 's.a.u', 'ctra.', 'informacion nutricional']):
            return True
    
    if current_label_key in ('fecha de consumo', 'consumo preferente', 'best before'):
        # Fecha de consumo no debe incluir conservación
        if any(_normalize_label_key(m) in text_normalized for m in ['importante:', 'conservar a', 'conservar em']):
            return True
    
    return False


def _derive_list_items(label_key: str, blocks: list, lines: list) -> list:
    line_items_labels = {
        'textos comerciales',
        'telefono/contacto',
        'teléfono/contacto',
    }
    force_list_labels = {
        'denominacion legal',
        'denomination legal',
        'denominacao legal',
        'denomination commerciale',
        'ingredientes',
        'ingredients',
        'ingredients a destacar',
        'ingredientes a destacar',
        'ingrédients',
        'fecha de consumo',
        'fecha de consumo preferente',
        'best before',
        'consumo preferente',
        'conservacion',
        'conservacao',
        'conservation',
        'textos comerciales',
        'telefono/contacto',
        'teléfono/contacto',
    }
    if label_key in line_items_labels:
        return [line['text'] for line in _sorted_lines(lines) if line['text']]
    
    # Usar un umbral más estricto para detectar gaps entre párrafos
    paragraphs = _paragraphs_from_lines(lines, max_gap_multiplier=1.2)
    
    # Para etiquetas específicas, detectar mejor los límites
    if label_key in force_list_labels:
        # Detectar si hay un gap grande que indica fin del contenido
        if paragraphs:
            filtered_paragraphs = []
            for i, para in enumerate(paragraphs):
                para_text = ' '.join(para).strip()
                # Detener si encontramos texto que claramente pertenece a otra sección
                if i > 0 and _looks_like_different_section(para_text, label_key):
                    break
                filtered_paragraphs.append(para)
            
            if filtered_paragraphs:
                items = _paragraphs_to_list_items(filtered_paragraphs)
                if len(items) >= 2:
                    return items
    
    if len(blocks) >= 2:
        return [block['text'] for block in blocks if block['text']]
    if paragraphs and len(paragraphs) >= 2:
        return _paragraphs_to_list_items(paragraphs)
    if blocks:
        parts = _split_paragraphs(blocks[0]['text'])
        if len(parts) >= 2:
            return parts
    if len(lines) >= 2:
        avg_len = sum(len(line['text']) for line in lines) / len(lines)
        if avg_len <= 60:
            return [line['text'] for line in lines if line['text']]
    return []


def _build_text_value(blocks: list, lines: list) -> str:
    paragraphs = _paragraphs_from_lines(lines)
    if paragraphs:
        return _paragraphs_to_text(paragraphs)
    if blocks:
        return '\n\n'.join(block['text'] for block in blocks if block['text']).strip()
    return '\n'.join(line['text'] for line in lines if line['text']).strip()


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
    notes = []
    table_content_lines = []
    header_detected = False
    
    for line in lines:
        text = line['text']
        # Separar notas de la tabla
        if _is_table_note_line(text):
            notes.append(text)
        # Detectar la línea de encabezados
        elif not header_detected and any(marker in text.lower() for marker in ['por 100', 'per 100', '%vrn', '%ir', '%dr']):
            header_detected = True
            # No agregar la línea de encabezados al contenido raw
            continue
        # Detectar si la línea contiene datos nutricionales
        elif _is_nutritional_data_line(text):
            table_content_lines.append(text)
        # Si ya empezamos a capturar datos y encontramos una línea que no es dato nutricional, detener
        elif header_detected and table_content_lines:
            # Verificar si es realmente una línea que no pertenece a la tabla
            if not any(marker in text.lower() for marker in ['energía', 'energia', 'grasa', 'lipido', 'hidrato', 'proteína', 'proteina', 'sal', 'hierro', 'ferro']):
                break
    
    # Construir el texto raw con formato estructurado
    raw_text = None
    if table_content_lines:
        raw_text = '\n'.join(table_content_lines)
    
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
    always_text_labels = {
        _normalize_label_key('Razón social'),
        _normalize_label_key('Marca de Identificación'),
        _normalize_label_key('GDA'),
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
            # Primero intentar detectar valores inline (a la derecha de la etiqueta)
            inline_region = fitz.Rect(
                label['rect'].x1 + 4,
                label_text_bbox[1] - 2,
                page.rect.width,
                label_text_bbox[3] + 2,
            )
            inline_lines = _lines_in_region(lines_all, inline_region, label_rects, stop_at_metadata=False)
            
            # Si hay líneas inline, usarlas como candidatas
            if inline_lines:
                # Verificar si hay contenido significativo inline
                inline_text = ' '.join(line['text'] for line in inline_lines).strip()
                if len(inline_text) > 3:  # Si hay contenido significativo inline
                    logger.info(
                        "Etiqueta %s (pagina %d): usando inline_lines=%d",
                        label_text,
                        page_index,
                        len(inline_lines),
                    )
                    region_lines = inline_lines
                    blocks = _group_lines_by_block(region_lines)
                else:
                    # Si el contenido inline es muy corto, buscar en la región debajo
                    region_lines = _lines_in_region(lines_all, region, label_rects)
                    blocks = _group_lines_by_block(region_lines)
            else:
                # Si no hay líneas inline, buscar en la región debajo
                region_lines = _lines_in_region(lines_all, region, label_rects)
                blocks = _group_lines_by_block(region_lines)
            
            # Si aún no hay contenido, usar texto segmentado como fallback
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
