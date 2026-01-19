import logging
import re
import unicodedata

import fitz

logger = logging.getLogger(__name__)


# =========================
# Helpers: color + text
# =========================
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
    return re.sub(r"\s+", " ", text).strip()


def _normalize_label_key(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return _collapse_spaces(normalized).lower()


def _should_skip_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.match(r"^(Page|Página)\s*:", stripped, re.IGNORECASE):
        return True
    return False


def _is_metadata_line(text: str) -> bool:
    """Detecta si una línea es metadata del documento (cabeceras/footers)."""
    text_lower = text.lower()
    metadata_patterns = [
        r'ficha\s+etiquetad',
        r'\blote\b',                 # <- NUEVO (LOTE 3 LATAS...)
        r'\bes/pt\b',                # <- NUEVO (ES/PT suelto)
        r'ref\s*:\s*\d+',
        r'version\s*:\s*[\d,]+',
        r'fecha\s*:\s*\d{2}[/-]\d{2}[/-]\d{4}',
        r'pa\s*:\s*\d+',
        r'fo\s*:\s*\d+',
        r'lang\s*:\s*\(',
        r'page\s*:\s*\d+\s+of\s+\d+',
        r'\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}',  # timestamp footer
    ]
    return any(re.search(pattern, text_lower) for pattern in metadata_patterns)


# =========================
# Blue labels detection
# =========================
def _extract_blue_shapes(page: fitz.Page) -> tuple[list, list]:
    """
    Extrae todas las formas azules del PDF.
    - Rectángulos azules con relleno: etiquetas
    - Líneas azules finas: decorativas
    """
    blue_fills = []
    blue_lines = []

    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue

        fill = drawing.get("fill")
        stroke = drawing.get("color")

        # Rellenos azules = etiquetas (alineadas a la izquierda)
        if fill and _is_blue(fill):
            rect_obj = fitz.Rect(rect)
            if rect_obj.x0 < 200:
                blue_fills.append(rect_obj)
        # Líneas azules finas = decorativas
        elif stroke and _is_blue(stroke) and fitz.Rect(rect).height <= 3:
            blue_lines.append(fitz.Rect(rect))

    return blue_fills, blue_lines


def _extract_labels_from_page(page: fitz.Page) -> tuple[list, list, list]:
    """
    Extrae etiquetas (rectángulos azules con texto blanco) de una página.

    Retorna: (labels_con_texto, blue_lines, rects_vacios)
    """
    blue_fills, blue_lines = _extract_blue_shapes(page)
    text_dict = page.get_text("dict")

    spans = []
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                spans.append(
                    {"text": text, "bbox": span.get("bbox"), "color": span.get("color")}
                )

    labels = []
    empty_rects = []
    seen = set()

    for rect in blue_fills:
        rect_expanded = fitz.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 1, rect.y1 + 1)

        spans_in = []
        for span in spans:
            if not span.get("bbox"):
                continue
            if rect_expanded.intersects(fitz.Rect(span["bbox"])) and _is_white(
                span.get("color")
            ):
                spans_in.append(span)

        if not spans_in:
            empty_rects.append(rect_expanded)
            continue

        spans_in.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
        text = _collapse_spaces(" ".join(s["text"] for s in spans_in))
        if not text:
            continue

        x0 = min(span["bbox"][0] for span in spans_in)
        y0 = min(span["bbox"][1] for span in spans_in)
        x1 = max(span["bbox"][2] for span in spans_in)
        y1 = max(span["bbox"][3] for span in spans_in)
        text_bbox = (x0, y0, x1, y1)

        key = (text, round(rect_expanded.y0, 1))
        if key in seen:
            continue
        seen.add(key)

        labels.append(
            {"text": text, "rect": rect_expanded, "text_bbox": text_bbox, "forced_value": None}
        )

    return labels, blue_lines, empty_rects


# =========================
# Text line extraction
# =========================
def _extract_page_lines(page: fitz.Page) -> list:
    words = page.get_text("words")
    lines_map = {}
    for x0, y0, x1, y1, word, block_no, line_no, _ in words:
        key = (block_no, line_no)
        lines_map.setdefault(key, []).append((x0, y0, x1, y1, word))
    lines = []
    for (block_no, line_no), items in lines_map.items():
        items.sort(key=lambda w: w[0])
        text = " ".join(item[4] for item in items).strip()
        if not text:
            continue
        x0 = min(item[0] for item in items)
        y0 = min(item[1] for item in items)
        x1 = max(item[2] for item in items)
        y1 = max(item[3] for item in items)
        lines.append({"block": block_no, "line": line_no, "text": text, "bbox": (x0, y0, x1, y1)})
    lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    return lines


def _lines_in_region(
    lines: list,
    region: fitz.Rect,
    label_rects: list,
    page_height: float = 0,
    top_content_y: float = 0,
    stop_at_metadata: bool = True,
    exclude_header: bool = True,
    exclude_footer: bool = True,
) -> list:
    """
    Filtra líneas dentro de una región, excluyendo etiquetas y (opcionalmente) header/footer.

    NOTA: Para extraer "exactamente igual" en regiones especiales (continuaciones de tabla o
    bloques largos), puedes desactivar exclude_header/exclude_footer/stop_at_metadata.
    """
    filtered = []

    # umbrales
    top_threshold = max(80, top_content_y) if (exclude_header and top_content_y > 0) else (80 if exclude_header else -1e9)
    bottom_threshold = (page_height - 50) if (exclude_footer and page_height > 0) else float("inf")

    for line in lines:
        rect = fitz.Rect(line["bbox"])

        if not rect.intersects(region):
            continue

        if exclude_header and rect.y1 < top_threshold:
            continue

        if exclude_footer and page_height > 0 and rect.y0 >= bottom_threshold:
            continue

        if any(rect.intersects(label_rect) for label_rect in label_rects):
            continue

        if _should_skip_line(line["text"]):
            continue

        # Filtrar timestamps (footer) siempre
        if re.match(r'^\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}$', line["text"].strip()):
            continue

        if stop_at_metadata and _is_metadata_line(line["text"]):
            break

        filtered.append(line)

    return filtered


def _sorted_lines(lines: list) -> list:
    return sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))


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
    heights = [max(1.0, line["bbox"][3] - line["bbox"][1]) for line in ordered]
    median_height = _median(heights) or 1.0
    gap_threshold = median_height * max_gap_multiplier

    paragraphs = []
    current = [ordered[0]["text"]]
    prev_y1 = ordered[0]["bbox"][3]

    for line in ordered[1:]:
        gap = line["bbox"][1] - prev_y1
        if gap > gap_threshold:
            paragraphs.append(current)
            current = [line["text"]]
        else:
            current.append(line["text"])
        prev_y1 = line["bbox"][3]
    paragraphs.append(current)
    return paragraphs


def _paragraphs_to_text(paragraphs: list) -> str:
    blocks = ["\n".join(lines) for lines in paragraphs if lines]
    return "\n\n".join(blocks).strip()


def _paragraphs_to_list_items(paragraphs: list) -> list:
    items = []
    for lines in paragraphs:
        item = _collapse_spaces(" ".join(lines))
        if item:
            items.append(item)
    return items


def _group_lines_by_block(lines: list) -> list:
    blocks = {}
    for line in lines:
        blocks.setdefault(line["block"], []).append(line)
    result = []
    for block_no, block_lines in blocks.items():
        block_lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
        text = "\n".join(line["text"] for line in block_lines).strip()
        if not text:
            continue
        x0 = min(line["bbox"][0] for line in block_lines)
        y0 = min(line["bbox"][1] for line in block_lines)
        x1 = max(line["bbox"][2] for line in block_lines)
        y1 = max(line["bbox"][3] for line in block_lines)
        result.append({"block": block_no, "text": text, "bbox": (x0, y0, x1, y1), "lines": block_lines})
    result.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return result


def _split_paragraphs(text: str) -> list:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


# =========================
# Value builders: text/list/table
# =========================
def _derive_list_items(label_key: str, blocks: list, lines: list) -> list:
    line_items_labels = {
        "textos comerciales",
        "telefono/contacto",
        "teléfono/contacto",
    }
    force_list_labels = {
        "denominacion legal",
        "denominação legal",
        "denomination legal",
        "denomination legale",
        "ingredientes",
        "ingredients",
        "ingrédients",
        "fecha de consumo",
        "fecha de consumo preferente",
        "best before",
        "consumo preferente",
        "conservacion",
        "conservacao",
        "conservação",
        "conservation",
        "textos comerciales",
        "telefono/contacto",
        "teléfono/contacto",
        "telefono / contacto",
    }

    if label_key in line_items_labels:
        items = [line["text"] for line in _sorted_lines(lines) if line["text"].strip()]
        return items if len(items) >= 2 else []

    paragraphs = _paragraphs_from_lines(lines, max_gap_multiplier=1.3)

    if label_key in force_list_labels:
        if paragraphs:
            items = _paragraphs_to_list_items(paragraphs)
            return items if len(items) >= 2 else []

    if len(blocks) >= 2:
        items = [block["text"] for block in blocks if block["text"].strip()]
        return items if len(items) >= 2 else []

    if paragraphs and len(paragraphs) >= 2:
        items = _paragraphs_to_list_items(paragraphs)
        return items if len(items) >= 2 else []

    if blocks and len(blocks) == 1:
        parts = _split_paragraphs(blocks[0]["text"])
        return parts if len(parts) >= 2 else []

    if len(lines) >= 3:
        avg_len = sum(len(line["text"]) for line in lines) / len(lines)
        if avg_len <= 50:
            items = [line["text"] for line in _sorted_lines(lines) if line["text"].strip()]
            return items if len(items) >= 2 else []

    return []


def _build_text_value(blocks: list, lines: list) -> str:
    paragraphs = _paragraphs_from_lines(lines)
    if paragraphs:
        return _paragraphs_to_text(paragraphs)
    if blocks:
        return "\n\n".join(block["text"] for block in blocks if block["text"].strip()).strip()
    return "\n".join(line["text"] for line in _sorted_lines(lines) if line["text"].strip()).strip()


def _table_headers_from_lines(lines: list) -> list:
    text = " ".join(line["text"] for line in lines).lower()
    text_norm = _normalize_label_key(text)
    headers = [""]

    has_100 = "por 100" in text_norm or "per 100" in text_norm or "par 100" in text_norm
    has_vrn = "%vrn" in text_norm or "vrn" in text_norm or "nrvs" in text_norm or "vnr" in text_norm
    has_serving = (
        "por racion" in text_norm
        or "por dose" in text_norm
        or "per serving" in text_norm
        or "par portion" in text_norm
    )
    has_ir = "%ir" in text_norm or "%dr" in text_norm or "%ri" in text_norm or "%ar" in text_norm

    if has_100:
        header_100 = "Por 100 g/ Por 100 g"
        if "per 100" in text_norm or "par 100" in text_norm:
            header_100 = "Por 100 g/ Por 100 g/ Per 100 g/ Par 100 g"
        headers.append(header_100)
    if has_vrn:
        header_vrn = "%VRN*/%VRN*"
        if "nrvs" in text_norm or "vnr" in text_norm:
            header_vrn = "%VRN*/%VRN*/%NRVs*/%VNR*"
        headers.append(header_vrn)
    if has_serving:
        headers.append("Por ración*/ Por dose*")
    if has_ir:
        headers.append("%IR**/%DR**")

    if len(headers) == 1:
        headers = [
            "",
            "Por 100 g/ Por 100 g/ Per 100 g/ Par 100 g",
            "%VRN*/%VRN*/%NRVs*/%VNR*",
            "Por ración*/ Por dose*",
            "%IR**/%DR**",
        ]
    return headers


def _is_table_header_line(text: str) -> bool:
    """Detecta si una línea es el header de columnas de la tabla nutricional."""
    t = _normalize_label_key(text)
    return (
        ("por 100" in t or "per 100" in t or "par 100" in t)
        and ("vrn" in t or "%vrn" in t or "nrvs" in t or "vnr" in t)
        and ("por racion" in t or "por dose" in t or "per serving" in t or "par portion" in t)
        and ("%ir" in t or "%dr" in t or "%ri" in t or "%ar" in t)
    )


def _is_table_note_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    # ✅ Si es header, NO es nota
    if _is_table_header_line(stripped):
        return False

    if stripped.startswith('*'):
        return True

    return bool(
        re.search(
            r'(ración recomendada|dose recomendada|ingesta de referencia|valores de referencia|contiene .*raciones|contém .*doses)',
            stripped,
            re.IGNORECASE,
        )
    )


def _is_nutrient_name_line(text: str) -> bool:
    """Líneas con nombre de nutriente (aunque no tengan números)."""
    t = _normalize_label_key(text)
    markers = [
        "valor energetico",
        "valor energético",
        "energia",
        "energy",
        "grasas",
        "lipidos",
        "lípidos",
        "fat",
        "saturadas",
        "saturados",
        "saturated",
        "hidratos de carbono",
        "carbohydrate",
        "glucides",
        "azucares",
        "azúcares",
        "acucares",
        "açúcares",
        "sugar",
        "sucres",
        "proteinas",
        "proteínas",
        "protein",
        "proteines",
        "protéines",
        "sal",
        "salt",
        "sel",
        "hierro",
        "ferro",
        "iron",
        "fer",
        "fibra",
        "fiber",
        "fibre",
    ]
    return any(m in t for m in markers)


def _dedupe_preserve_order(lines: list[str]) -> list[str]:
    """Deduplicar líneas manteniendo el orden, usando normalización."""
    seen = set()
    out = []
    for t in lines:
        k = _normalize_label_key(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _cluster_words_into_rows(words, y_tol=2.0):
    """Agrupa words por fila (cluster en Y), luego ordena por X en cada fila."""
    items = [(w[0], w[1], w[2], w[3], w[4]) for w in words if w[4].strip()]
    items.sort(key=lambda x: (x[1], x[0]))  # y, x

    rows = []
    cur = []
    cur_y = None

    for x0, y0, x1, y1, text in items:
        if cur_y is None:
            cur_y = y0
            cur = [(x0, text)]
            continue
        if abs(y0 - cur_y) <= y_tol:
            cur.append((x0, text))
        else:
            cur.sort(key=lambda t: t[0])
            rows.append(" ".join(t[1] for t in cur).strip())
            cur_y = y0
            cur = [(x0, text)]

    if cur:
        cur.sort(key=lambda t: t[0])
        rows.append(" ".join(t[1] for t in cur).strip())
    return rows


def _build_table_value(label_text: str, lines: list, page: fitz.Page = None, region: fitz.Rect = None) -> dict:
    """
    Construye la tabla:
    - raw: reconstruye filas usando clustering de words (orden correcto)
    - valorComplementario: notas (asteriscos, IR/VRN, raciones)
    """
    if not lines:
        return {"titulo": label_text, "headers": _table_headers_from_lines([]), "raw": None, "valorComplementario": None}

    notes = []
    titulo_line = None
    
    # Extraer notas y título de lines
    ordered = _sorted_lines(lines)
    for line in ordered:
        text = line["text"].strip()
        text_lower = text.lower()
        if not text:
            continue
        if _is_table_note_line(text):
            notes.append(text)
            continue
        if titulo_line is None and ("nutricional" in text_lower or "nutrition" in text_lower):
            titulo_line = text
            continue

    # Si tenemos page y region, reconstruimos filas con words (MÁS PRECISO)
    clean_rows = []
    if page and region:
        page_words = page.get_text("words")
        words_in_region = [w for w in page_words if fitz.Rect(w[0], w[1], w[2], w[3]).intersects(region)]
        rows = _cluster_words_into_rows(words_in_region, y_tol=2.0)
        
        for r in rows:
            rl = r.lower().strip()
            if not rl:
                continue
            if _is_metadata_line(r):
                continue
            if _is_table_note_line(r):
                if r not in notes:
                    notes.append(r)
                continue
            header_markers = [
                'por 100', 'per 100', 'par 100',
                '%vrn', 'vrn', 'nrvs', 'vnr',
                '%ir', '%dr', '%ri', '%ar',
                'por ración', 'por racion', 'por dose', 'per serving', 'par portion',
            ]
            if any(m in rl for m in header_markers):
                continue
            # Filtrar si es línea de título nutricional
            if 'nutricional' in rl or 'nutrition' in rl:
                continue
            clean_rows.append(r)
    else:
        # Fallback: usar el método anterior basado en lines
        content_lines = []
        for line in ordered:
            text = line["text"].strip()
            text_lower = text.lower()
            if not text:
                continue
            if _is_metadata_line(text):
                continue
            if _is_table_note_line(text):
                continue
            if titulo_line and text == titulo_line:
                continue
            header_markers = [
                'por 100', 'per 100', 'par 100',
                '%vrn', 'vrn', 'nrvs', 'vnr',
                '%ir', '%dr', '%ri', '%ar',
                'por ración', 'por racion', 'por dose', 'per serving', 'par portion',
            ]
            if any(marker in text_lower for marker in header_markers):
                continue
            content_lines.append(text)

        # Merge de nombres + números
        merged = []
        pending_name = None
        for t in content_lines:
            has_num = bool(re.search(r"\d", t))
            is_name = _is_nutrient_name_line(t) and not has_num
            if is_name:
                if pending_name:
                    merged.append(pending_name)
                pending_name = t
                continue
            if pending_name:
                merged.append(f"{pending_name} {t}")
                pending_name = None
            else:
                merged.append(t)
        if pending_name:
            merged.append(pending_name)
        clean_rows = merged

    # Deduplicar manteniendo orden
    clean_rows = _dedupe_preserve_order(clean_rows)

    raw_text = "\n".join(clean_rows).strip() if clean_rows else None
    titulo = titulo_line if titulo_line else label_text

    value = {
        "titulo": titulo,
        "headers": _table_headers_from_lines(lines),
        "raw": raw_text,
        "valorComplementario": notes if notes else None,
    }
    if notes:
        value["tipoComplementario"] = "lista"
    return value


# =========================
# Identification mark
# =========================
def _detect_identification_mark(page: fitz.Page, blue_lines: list) -> tuple[str, fitz.Rect] | None:
    candidates = []
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        if not text:
            continue

        compact = _collapse_spaces(text.replace("\n", " "))
        if re.search(r"\bES\s+[\d./A-Z]+\s+CE\b", compact, re.IGNORECASE):
            match = re.search(r"(ES\s+[\d./A-Z]+\s+CE)", compact, re.IGNORECASE)
            if match:
                mark_text = match.group(1)
                rect = fitz.Rect(x0, y0, x1, y1)
                block_area = rect.width * rect.height
                text_length = len(compact)
                candidates.append(
                    {
                        "text": mark_text,
                        "rect": rect,
                        "score": block_area + text_length * 10,
                    }
                )
    if not candidates:
        return None
    best = min(candidates, key=lambda c: c["score"])
    return (best["text"], best["rect"])


# =========================
# Categorization + languages
# =========================
def _categoria_for_label(label_text: str) -> str:
    key = _normalize_label_key(label_text)
    if (
        "informacion nutricional" in key
        or "informacao nutricional" in key
        or "average nutrition" in key
        or "information nutritionnelle" in key
        or "declaracao nutricional" in key
        or key.startswith("gda")
    ):
        return "NUTRICIONAL"
    if "ingrediente" in key or "alergen" in key:
        return "INGREDIENTES"
    if "fecha de consumo" in key or "consumo preferente" in key or "caducidad" in key:
        return "CONSUMO_PREFERENTE"
    return "INFO_GENERAL"


def _detect_languages(text: str) -> list:
    lowered = text.lower()
    normalized = _normalize_label_key(text)
    es_markers = [
        "denominación",
        "denominacion",
        "razón social",
        "razon social",
        "conservación",
        "conservacion",
        "consumir",
        "ingredientes",
        "ración",
        "racion",
        "cantidad neta",
        "consumo preferente",
    ]
    pt_markers = [
        "denominação",
        "denominacao",
        "razao social",
        "razão social",
        "conservação",
        "conservacao",
        "consumir",
        "ingredientes",
        "dose",
        "contém",
        "contem",
        "porção",
        "porcao",
    ]
    en_markers = [
        "ingredients",
        "best before",
        "keep in a cool",
        "net quantity",
        "nutrition information",
        "energy",
        "fat",
        "carbohydrate",
        "salt",
        "protein",
    ]
    fr_markers = [
        "ingredients",
        "a consommer",
        "a conserver",
        "information nutritionnelle",
        "valeurs nutritionnelles",
        "energie",
        "matieres grasses",
        "glucides",
        "sel",
        "proteines",
    ]

    es_score = sum(1 for marker in es_markers if marker in lowered or marker in normalized)
    pt_score = sum(1 for marker in pt_markers if marker in lowered or marker in normalized)
    en_score = sum(1 for marker in en_markers if marker in lowered or marker in normalized)
    fr_score = sum(1 for marker in fr_markers if marker in lowered or marker in normalized)

    idiomas = []
    if es_score >= 2:
        idiomas.append("ES")
    if pt_score >= 2:
        idiomas.append("PT")
    if en_score >= 2:
        idiomas.append("EN")
    if fr_score >= 2:
        idiomas.append("FR")
    return idiomas


# =========================
# Main extractor (FINAL)
# =========================
def extract_pdf_fe(pdf_bytes: bytes) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    etiquetas = []
    all_text_parts = []
    title = ""
    total_words = 0
    labels_found = 0

    logger.info("PDF abierto: %d paginas", doc.page_count)

    # Labels always text (keep line breaks)
    always_text_labels = {
        _normalize_label_key("Razón social"),
        _normalize_label_key("Razão social"),
        _normalize_label_key("Marca de Identificación"),
        _normalize_label_key("Marca de Identificação"),
        _normalize_label_key("GDA"),
        _normalize_label_key("Denominación Comercial"),
        _normalize_label_key("Denominação Comercial"),
        _normalize_label_key("Cantidad neta"),
        _normalize_label_key("Quantidade líquida"),
        _normalize_label_key("Ingredientes a destacar"),
    }

    table_labels = {
        _normalize_label_key("Información nutricional media"),
        _normalize_label_key("Informação nutricional média"),
        _normalize_label_key("Average nutrition information"),
        _normalize_label_key("Information nutritionnelle moyenne"),
    }

    # Labels that are long + commonly truncated if you exclude footer/metadata
    long_block_labels = {
        _normalize_label_key("Instrucciones diseño"),
        _normalize_label_key("Instrucciones de diseño"),
        _normalize_label_key("Instruções design"),
        _normalize_label_key("Instruções de design"),
        _normalize_label_key("Instrucoes de design"),
        _normalize_label_key("Design instructions"),
        _normalize_label_key("Instructions design"),
        _normalize_label_key("Información mínima en embalaje"),
        _normalize_label_key("Informacion minima en embalaje"),
        _normalize_label_key("Informação mínima na embalagem"),
        _normalize_label_key("Informacao minima na embalagem"),
        _normalize_label_key("Minimum packaging information"),
        _normalize_label_key("Minimum information on packaging"),
        _normalize_label_key("Information minimale sur l'emballage"),
        _normalize_label_key("Informations minimales sur l'emballage"),
    }

    # 1) collect all labels (page, rect)
    all_labels_with_page = []
    for page_index, page in enumerate(doc, start=1):
        page_words = page.get_text("words")
        total_words += len(page_words)

        labels, blue_lines, empty_rects = _extract_labels_from_page(page)
        logger.info("Pagina %d: labels_azules=%d rects_vacios=%d", page_index, len(labels), len(empty_rects))

        mark = _detect_identification_mark(page, blue_lines)
        if mark:
            mark_text, oval_rect = mark
            best_rect = None
            min_distance = float("inf")
            for empty_rect in empty_rects:
                empty_center_y = (empty_rect.y0 + empty_rect.y1) / 2
                oval_center_y = (oval_rect.y0 + oval_rect.y1) / 2
                distance = abs(empty_center_y - oval_center_y)
                if distance < min_distance and distance < 100:
                    min_distance = distance
                    best_rect = empty_rect
            mark_rect = best_rect if best_rect else oval_rect
            labels.append(
                {"text": "Marca de Identificación", "rect": mark_rect, "text_bbox": mark_rect, "forced_value": mark_text}
            )

        # title (page 1)
        if page_index == 1 and not title:
            if labels:
                top_label_y = min(label.get("text_bbox", label["rect"])[1] for label in labels)
                title_region = fitz.Rect(0, 0, page.rect.width, top_label_y - 2)
                # title: do not exclude header/footer/metadata (we want it as-is)
                lines_top = _lines_in_region(
                    _extract_page_lines(page),
                    title_region,
                    [],
                    page.rect.height,
                    top_content_y=0,
                    stop_at_metadata=False,
                    exclude_header=False,
                    exclude_footer=False,
                )
                title = _collapse_spaces(" ".join(line["text"] for line in lines_top))

        for label in labels:
            all_labels_with_page.append({"label": label, "page": page, "page_index": page_index})

    # sort labels by page + y + x
    all_labels_with_page.sort(key=lambda item: (item["page_index"], item["label"]["rect"].y0, item["label"]["rect"].x0))

    # 2) filter duplicates PER PAGE (keep first occurrence per page)
    filtered_labels = []
    seen_per_page = set()  # (page_idx, label_key)
    for item in all_labels_with_page:
        label = item["label"]
        page_idx = item["page_index"]
        label_key = _normalize_label_key(label["text"])
        key = (page_idx, label_key)
        if key in seen_per_page:
            continue
        seen_per_page.add(key)
        filtered_labels.append(item)
    all_labels_with_page = filtered_labels

    logger.info("Total etiquetas unicas (por pagina): %d", len(all_labels_with_page))

    # 3) process each label
    for idx, item in enumerate(all_labels_with_page):
        label = item["label"]
        page = item["page"]
        page_index = item["page_index"]

        label_text = label["text"]
        label_key = _normalize_label_key(label_text)

        # Determine capture region for this label
        label_text_bbox = label.get("text_bbox", label["rect"])
        y_start = max(label["rect"].y1, label_text_bbox[3]) + 2

        # is table?
        is_table_label = (
            label_key in table_labels
            or "informacion nutricional" in label_key
            or "informacao nutricional" in label_key
            or "average nutrition" in label_key
            or "information nutritionnelle" in label_key
            or "declaracao nutricional" in label_key
        )

        # find next label on same page (and next page for table continuation)
        next_label_same_page = None
        next_label_next_page = None
        next_page_obj = None

        for next_idx in range(idx + 1, len(all_labels_with_page)):
            next_item = all_labels_with_page[next_idx]
            if next_item["page_index"] == page_index:
                next_label_same_page = next_item["label"]
                break
            if next_item["page_index"] == page_index + 1:
                next_label_next_page = next_item["label"]
                next_page_obj = next_item["page"]
                break

        if next_label_same_page:
            next_bbox = next_label_same_page.get("text_bbox", next_label_same_page["rect"])
            y_end = min(next_label_same_page["rect"].y0, next_bbox[1]) - 2
            if y_end <= y_start:
                y_end = page.rect.height
        else:
            y_end = page.rect.height

        region = fitz.Rect(0, y_start, page.rect.width, y_end)

        # extract lines on current page
        lines_all = _extract_page_lines(page)

        page_labels = [it["label"] for it in all_labels_with_page if it["page_index"] == page_index]
        label_rects = [lbl["rect"] for lbl in page_labels]

        # top_content_y = first label y (to skip repeated header)
        top_content_y = 0
        if page_labels:
            first_label = min(page_labels, key=lambda lbl: lbl["rect"].y0)
            top_content_y = first_label["rect"].y0 - 5

        # label text bboxes to exclude (avoid capturing label name again)
        all_label_text_bboxes = []
        for lbl in page_labels:
            lbl_bbox = lbl.get("text_bbox")
            if lbl_bbox:
                all_label_text_bboxes.append(fitz.Rect(lbl_bbox))
            all_label_text_bboxes.append(lbl["rect"])

        # For "exactly like PDF": for long blocks, do not exclude footer, do not stop at metadata
        is_long_block = label_key in long_block_labels

        region_lines_raw = _lines_in_region(
            lines_all,
            region,
            label_rects,
            page.rect.height,
            top_content_y=top_content_y,
            stop_at_metadata=(False if is_long_block else True),
            exclude_header=True,
            exclude_footer=(False if is_long_block else True),
        )

        # remove any line that intersects any label bbox on same page
        region_lines = []
        for line in region_lines_raw:
            line_rect = fitz.Rect(line["bbox"])
            if any(line_rect.intersects(lbl_bbox) for lbl_bbox in all_label_text_bboxes):
                continue
            region_lines.append(line)

        # Table continuation: capture top part of next page WITHOUT header exclusion
        if is_table_label and not next_label_same_page and next_label_next_page and next_page_obj:
            next_lines_all = _extract_page_lines(next_page_obj)

            next_page_labels = [it["label"] for it in all_labels_with_page if it["page_index"] == page_index + 1]
            next_label_rects = [lbl["rect"] for lbl in next_page_labels]

            next_all_label_text_bboxes = []
            for lbl in next_page_labels:
                lbl_bbox = lbl.get("text_bbox")
                if lbl_bbox:
                    next_all_label_text_bboxes.append(fitz.Rect(lbl_bbox))
                next_all_label_text_bboxes.append(lbl["rect"])

            next_bbox = next_label_next_page.get("text_bbox", next_label_next_page["rect"])
            next_y_end = min(next_label_next_page["rect"].y0, next_bbox[1]) - 2
            next_region = fitz.Rect(0, 0, next_page_obj.rect.width, next_y_end)

            next_region_lines_raw = _lines_in_region(
                next_lines_all,
                next_region,
                next_label_rects,
                next_page_obj.rect.height,
                top_content_y=0,
                stop_at_metadata=False,   # capture notes even if metadata-like
                exclude_header=False,      # KEY: allow top-of-page notes (*, **, ***)
                exclude_footer=True,
            )

            for line in next_region_lines_raw:
                line_text = line["text"].strip()
                if not line_text:
                    continue

                # ✅ NO meter cabecera/metadata de la página (Ficha/LOTE/REF/Fecha/etc.)
                if _is_metadata_line(line_text):
                    continue

                line_rect = fitz.Rect(line["bbox"])
                is_label_text = False
                for lbl_bbox in next_all_label_text_bboxes:
                    if line_rect.intersects(lbl_bbox):
                        is_label_text = True
                        break
                if is_label_text:
                    continue

                region_lines.append(line)

        blocks = _group_lines_by_block(region_lines)

        # Determine type/value
        if label.get("forced_value"):
            tipo = "texto"
            valor = label["forced_value"]
        elif is_table_label:
            tipo = "tabla"
            valor = _build_table_value(label_text, region_lines, page, region)
        else:
            list_items = _derive_list_items(label_key, blocks, region_lines)
            if label_key in always_text_labels:
                tipo = "texto"
                valor = _build_text_value(blocks, region_lines)
            elif len(list_items) >= 2:
                tipo = "lista"
                valor = list_items
            else:
                tipo = "texto"
                valor = _build_text_value(blocks, region_lines)

        etiquetas.append(
            {"etiqueta": label_text, "valor": valor, "tipo": tipo, "categoria": _categoria_for_label(label_text), "pagina": page_index}
        )
        labels_found += 1

        # add text for language detection
        if isinstance(valor, list):
            all_text_parts.extend(valor)
        elif isinstance(valor, dict):
            all_text_parts.append(valor.get("titulo", ""))
            if valor.get("valorComplementario"):
                all_text_parts.extend(valor.get("valorComplementario") or [])
            if valor.get("raw"):
                all_text_parts.append(valor.get("raw") or "")
        else:
            all_text_parts.append(str(valor))

    if total_words == 0:
        doc.close()
        raise ValueError("PDF sin texto embebido; OCR deshabilitado")

    if labels_found == 0:
        doc.close()
        raise ValueError("No se detectaron etiquetas en texto embebido; OCR deshabilitado")

    doc.close()

    full_text = " ".join([title] + all_text_parts)
    idiomas = _detect_languages(full_text)

    return {"titulo": title, "idiomas_detectados": idiomas, "etiquetas": etiquetas}
