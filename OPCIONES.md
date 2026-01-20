# Opciones de Preprocesamiento

Documentación completa de todas las opciones disponibles para el endpoint `/preprocess`.

---

## 📋 Estructura Básica del Request

```json
{
  "image_url": "https://...",
  "force_strategy": "white_on_black",
  "upscale": true
}
```

**Nota:** Todas las opciones van en el nivel raíz del JSON (formato plano, sin anidación).

---

## 🔑 Campos Obligatorios

Debes especificar **una** de estas opciones:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `image_url` | `string` | URL de la imagen a procesar |
| `image_base64` | `string` | Imagen codificada en base64 |

---

## 🎨 Presets Predefinidos

| Preset | Descripción | Uso Recomendado |
|--------|-------------|-----------------|
| `table_ocr` | Balance entre calidad y velocidad (default) | Tablas generales |
| `table_ocr_aggressive` | Máxima calidad, más procesamiento | Imágenes muy degradadas |
| `white_text_on_color` | Optimizado para texto claro sobre fondos de color | Fondos azules, verdes, rojos |
| `red_table_blurry` ⭐ | Pipeline especializado para fondo rojo + texto blanco borroso | Tablas nutricionales |
| `smart_auto` | Detección automática inteligente | Cuando no sabes qué preset usar |
| `small_text_sharp` | Detección de ESTRUCTURA (bordes gruesos) | Identificar líneas/celdas de tabla |
| `ocr_preserve_details` ⭐ | Preserva detalles finos (suave) | OCR con símbolos (,.*<%) |
| `ocr_ultra_fine` ⭐⭐ | CLAHE + bilateral + adaptive + morfología | OCR tradicional (Tesseract) |
| `gemini_vision` ⭐⭐⭐ | **Escala de grises + filtros ultra-suaves** | **Gemini/GPT-Vision/Claude (EVITA ALUCINACIONES)** |
| `auto` 🤖 | **100% automático: detecta rotación completa + estrategia + crop** | **Cualquier tipo de imagen/PDF** |
| `grayscale_auto` 🎯 | **Escala de grises + auto-rotación completa (0-90-180-270°)** | **Enderezar tablas automáticamente** |
| `grayscale_only` | **Solo escala de grises, sin procesamiento** | **Conversión simple sin filtros** |
| `minimal` | Mínimo procesamiento | Imágenes de alta calidad |

### Uso:
```json
{
  "image_url": "https://...",
  "preset": "ocr_preserve_details"
}
```

**⚠️ IMPORTANTE - Elige el preset según tu OCR:**

### Para Modelos Multimodales (Gemini, GPT-Vision, Claude):
- **`gemini_vision`** ⭐⭐⭐: **Escala de grises + filtros MUY suaves**
  - ✅ Conversión a escala de grises (rojo → gris)
  - ✅ CLAHE suave (clip_limit=1.5) para contraste de números finos
  - ✅ Deblur ultra-suave (strength=0.3) para mejorar legibilidad
  - ✅ Upscale con Lanczos4 (hasta 2000px)
  - ❌ **SIN binarización** (evita alucinaciones)
  - Reduce errores: 7,2 vs 7,1, 344 vs 342, pérdida de comas/símbolos

### Para OCR Tradicional (Tesseract):
- **`ocr_ultra_fine`** ⭐⭐: Con binarización adaptativa + morfología
- **`ocr_preserve_details`**: Más suave
- **`small_text_sharp`**: Para detectar estructura (puede perder detalles)

---

## ⚙️ Opciones de Preprocesamiento

Todas las opciones son **opcionales** y sobrescriben el preset si están definidas.

### 🧠 Análisis Inteligente

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `smart_table_analysis` | `boolean` | `false` | Activa análisis inteligente de color de fondo y texto |
| `force_strategy` | `string` | `null` | Fuerza una estrategia de conversión específica |

#### Estrategias Disponibles (`force_strategy`)

| Estrategia | Descripción | Caso de Uso |
|------------|-------------|-------------|
| `white_on_black` | Texto blanco sobre fondo oscuro | Fondos negros/oscuros |
| `black_on_white` | Texto oscuro sobre fondo claro | Documentos estándar |
| `enhance_contrast` | Mejora contraste bajo | Imágenes deslavadas |
| `extract_luminosity` | Extrae luminosidad (ignora color) | Fondos de color saturado |
| `red_background_advanced` ⭐ | Pipeline LAB+HSV optimizado | Tablas nutricionales con fondo rojo |
| `invert_colors` | Invierte toda la imagen | Negativos |

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "smart_table_analysis": true,
  "force_strategy": "red_background_advanced"
}
```

---

### 🔍 Reducción de Borrosidad (Deblur)

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `deblur` | `boolean` | `false` | Activa reducción de borrosidad |
| `deblur_method` | `string` | `"unsharp"` | Método de deblur: `"unsharp"`, `"laplacian"`, `"aggressive"` ⭐ |
| `deblur_strength` | `float` | `1.0` | Intensidad del deblur (0.5-2.0) |

#### Métodos de Deblur

- **`unsharp`**: Balance entre calidad y velocidad (recomendado)
- **`laplacian`**: Enfoque en bordes
- **`aggressive`** ⭐: Deblur muy fuerte para texto MUY pequeño y pegado (mejor para tablas con letra diminuta)

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "deblur": true,
  "deblur_method": "aggressive",
  "deblur_strength": 1.5
}
```

---

### 📐 Escalado (Upscale)

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `upscale` | `boolean` | `true` | Agranda imágenes pequeñas |
| `min_size` | `integer` | `800` | Tamaño mínimo en píxeles |
| `max_scale` | `float` | `3.0` | Factor máximo de escalado |
| `upscale_method` | `string` | `"cubic"` | Método de interpolación: `"cubic"`, `"lanczos4"` ⭐, `"linear"` |

#### Métodos de Upscale

- **`cubic`**: Balance entre calidad y velocidad (default)
- **`lanczos4`** ⭐: Mejor calidad para preservar detalles finos (recomendado para texto pequeño)
- **`linear`**: Más rápido, menor calidad

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "upscale": true,
  "min_size": 1500,
  "max_scale": 5.0,
  "upscale_method": "lanczos4"
}
```

---

### 🔄 Rotación

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `rotate_90` | `boolean` | `false` | Rota la imagen 90° sentido horario (manual) |
| `rotate_180` | `boolean` | `false` | Rota la imagen 180° (manual) |
| `rotate_270` | `boolean` | `false` | Rota la imagen 270° (manual, equivalente a 90° antihorario) |
| `auto_rotate_all` 🤖 | `boolean` | `false` | **Detecta automáticamente rotación óptima (0°, 90°, 180°, 270°)** |
| `auto_detect_rotation` | `boolean` | `false` | Detecta automáticamente solo si está rotada 180° (más rápido) |

**Notas:**
- Solo se puede usar **una** opción de rotación a la vez
- Prioridad: manual (`rotate_X`) > `auto_rotate_all` > `auto_detect_rotation`
- **`auto_rotate_all` 🎯 RECOMENDADO:** Detecta automáticamente la mejor orientación analizando:
  - Densidad de texto en región superior
  - Presencia de líneas horizontales
  - Aspect ratio (vertical/horizontal)
  - Distribución estructurada del contenido
- `auto_detect_rotation`: Más rápido, solo detecta 180° (imágenes al revés)

**Ejemplo - Rotar 90° (horizontal a vertical):**
```json
{
  "image_url": "https://...",
  "rotate_90": true
}
```

**Ejemplo - Rotar 180° (al revés):**
```json
{
  "image_url": "https://...",
  "rotate_180": true
}
```

**Ejemplo - Rotar 270° (vertical a horizontal):**
```json
{
  "image_url": "https://...",
  "rotate_270": true
}
```

**Ejemplo - Auto-detección simple (solo 180°, más rápido):**
```json
{
  "image_url": "https://...",
  "auto_detect_rotation": true
}
```

**Ejemplo - Auto-detección completa (0-90-180-270°, más inteligente) 🤖:**
```json
{
  "image_url": "https://...",
  "auto_rotate_all": true
}
```

---

### ✨ Mejora de Contraste (CLAHE)

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `enhance_contrast` | `boolean` | `true` | Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) |
| `clip_limit` | `float` | `3.0` | Intensidad del CLAHE (1.0-10.0) |

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "enhance_contrast": true,
  "clip_limit": 4.0
}
```

---

### 🎨 Conversión y Eliminación de Fondos

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `convert_to_grayscale` | `boolean` | `false` | Convierte a escala de grises (rojo→gris, azul→gris, etc.) sin eliminar fondo ⭐ |
| `remove_color_bg` | `boolean` | `true` | Elimina fondos de color (rojo, azul, verde, amarillo) |
| `extract_white_text` | `boolean` | `false` | Extrae texto blanco de fondos de color |
| `extract_text_adaptive` | `boolean` | `false` | Extracción adaptativa (funciona con texto claro u oscuro) |

**⭐ `convert_to_grayscale`:** Ideal para LLMs multimodales (Gemini/GPT-Vision) - preserva toda la información visual pero sin colores que puedan confundir al modelo.

**Nota:** Solo una de `remove_color_bg`, `extract_white_text`, o `extract_text_adaptive` debe estar en `true` a la vez.

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "extract_white_text": true
}
```

---

### 📏 Corrección de Inclinación (Deskew)

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `deskew` | `boolean` | `true` | Corrige rotación/inclinación automáticamente |

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "deskew": true
}
```

---

### 🧹 Reducción de Ruido (Denoise)

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `denoise` | `boolean` | `true` | Activa reducción de ruido |
| `denoise_method` | `string` | `"bilateral"` | Método: `"gaussian"`, `"bilateral"`, `"nlm"` |

#### Métodos de Denoise

- **`gaussian`**: Rápido, suaviza uniformemente
- **`bilateral`**: Balance entre velocidad y calidad (recomendado)
- **`nlm`**: Mejor calidad, más lento (Non-Local Means)

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "denoise": true,
  "denoise_method": "nlm"
}
```

---

### 🔪 Nitidez (Sharpen)

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `sharpen` | `boolean` | `false` | Aumenta la nitidez |
| `sharpen_strength` | `float` | `1.0` | Intensidad del sharpening (0.5-3.0) |
| `sharpen_method` | `string` | `"kernel"` | Método: `"kernel"` (rápido), `"unsharp"` (mejor calidad) |
| `preserve_fine_details` | `boolean` | `false` | No aplicar median blur después de conversión (preserva texto pequeño) |

#### Métodos de Sharpen

- **`kernel`**: Sharpening tradicional con kernel (rápido)
- **`unsharp`**: Unsharp masking (mejor calidad para texto pequeño)

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "sharpen": true,
  "sharpen_strength": 0.8,
  "sharpen_method": "unsharp",
  "preserve_fine_details": true
}
```

---

### ⚫⚪ Binarización

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `binarize` | `boolean` | `false` | Convierte a blanco y negro puro |
| `binarize_method` | `string` | `"otsu"` | Método: `"otsu"` o `"adaptive_gaussian"` |

#### Métodos de Binarización

- **`otsu`**: Umbral automático global (más rápido)
- **`adaptive_gaussian`**: Umbral adaptativo local (mejor para iluminación desigual)

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "binarize": true,
  "binarize_method": "adaptive_gaussian"
}
```

---

### 🔁 Inversión Automática

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `auto_invert` | `boolean` | `true` | Invierte automáticamente si el fondo es oscuro |

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "auto_invert": true
}
```

---

## 📦 Configuraciones de Presets

### `table_ocr` (Default)
```json
{
  "upscale": true,
  "enhance_contrast": true,
  "remove_color_bg": true,
  "deskew": true,
  "denoise": true,
  "auto_invert": true,
  "clip_limit": 3.0
}
```

### `table_ocr_aggressive`
```json
{
  "upscale": true,
  "enhance_contrast": true,
  "remove_color_bg": true,
  "deskew": true,
  "denoise": true,
  "denoise_method": "nlm",
  "sharpen": true,
  "sharpen_strength": 1.5,
  "binarize": true,
  "binarize_method": "adaptive_gaussian",
  "auto_invert": true,
  "clip_limit": 4.0
}
```

### `white_text_on_color`
```json
{
  "rotate_180": false,
  "upscale": true,
  "min_size": 1200,
  "max_scale": 4.0,
  "enhance_contrast": true,
  "clip_limit": 5.0,
  "extract_white_text": true,
  "remove_color_bg": false,
  "deskew": true,
  "denoise": true,
  "sharpen": false,
  "binarize": false,
  "auto_invert": false
}
```

### `red_table_blurry` ⭐
```json
{
  "smart_table_analysis": true,
  "force_strategy": "red_background_advanced",
  "upscale": true,
  "min_size": 1000,
  "max_scale": 3.0,
  "deblur": true,
  "deblur_method": "unsharp"
}
```

### `smart_auto`
```json
{
  "smart_table_analysis": true,
  "upscale": true,
  "min_size": 1000,
  "max_scale": 3.0,
  "deblur": true,
  "deblur_method": "unsharp"
}
```

### `small_text_sharp` (Detección de Estructura)
**⚠️ Advertencia:** Engrosa bordes, puede perder detalles finos. Úsalo solo para detectar líneas/celdas.
```json
{
  "smart_table_analysis": true,
  "upscale": true,
  "min_size": 2000,
  "max_scale": 5.0,
  "upscale_method": "lanczos4",
  "deblur": true,
  "deblur_method": "aggressive",
  "deblur_strength": 1.0,
  "sharpen": true,
  "sharpen_strength": 0.5,
  "sharpen_method": "unsharp",
  "preserve_fine_details": true
}
```

### `ocr_preserve_details` ⭐ (Suave)
**✅ Para OCR:** Preserva comas, puntos, símbolos (<, %, *, etc.)
```json
{
  "smart_table_analysis": true,
  "upscale": true,
  "min_size": 1800,
  "max_scale": 4.0,
  "upscale_method": "lanczos4",
  "deblur": true,
  "deblur_method": "unsharp",
  "deblur_strength": 0.6,
  "sharpen": false,
  "preserve_fine_details": true
}
```

### `gemini_vision` ⭐⭐⭐ (EMPEZANDO DE CERO - PARA MODELOS MULTIMODALES)
**🎯 Optimizado para Gemini/GPT-Vision/Claude:** Evita alucinaciones

**Filosofía:** Minimalismo + Escala de grises natural
- ✅ **Conversión a escala de grises** (rojo → gris, sin eliminar fondo)
- ✅ **CLAHE suave** (1.5) para mejorar contraste de números finos (7,2 vs 7,1)
- ✅ **Deblur ultra-suave** (0.3) para aumentar legibilidad sin engrosar
- ✅ **Upscale con Lanczos4** (2000px) para preservar detalles
- ❌ **SIN binarización** (los LLMs prefieren escala de grises natural)
- ❌ **SIN denoise** (puede difuminar números pequeños)
- ❌ **SIN sharpen** (puede crear artefactos que confunden al LLM)
- **Resultado:** Gemini lee correctamente 7,2 (no 7,1), 344 kJ (no 342), <1% (no inventa), no pierde comas ni símbolos

```json
{
  "smart_table_analysis": false,
  "upscale": true,
  "min_size": 2000,
  "max_scale": 3.0,
  "upscale_method": "lanczos4",
  "convert_to_grayscale": true,
  "enhance_contrast": true,
  "clip_limit": 1.5,
  "clahe_tile_grid_size": [8, 8],
  "deblur": true,
  "deblur_method": "unsharp",
  "deblur_strength": 0.3,
  "denoise": false,
  "remove_color_bg": false,
  "sharpen": false,
  "binarize": false,
  "post_morphology": false,
  "deskew": false,
  "auto_invert": false,
  "preserve_fine_details": true
}
```

### `ocr_ultra_fine` ⭐⭐ (Para Tesseract OCR)
**🎯 Para OCR tradicional:** Control granular total
- CLAHE suave (1.8) para contraste local sin "quemar"
- Bilateral denoise (d=5) para suavizar sin difuminar
- Unsharp muy bajo (0.35) para evitar engrosar bordes
- Adaptive threshold (blockSize=51, C=9) mantiene comas y símbolos
- Morfología (open 2x2) elimina ruido pequeño

```json
{
  "smart_table_analysis": false,
  "upscale": true,
  "min_size": 2400,
  "max_scale": 3.0,
  "upscale_method": "lanczos4",
  "denoise": true,
  "denoise_method": "bilateral",
  "bilateral_d": 5,
  "bilateral_sigma_color": 50,
  "bilateral_sigma_space": 50,
  "enhance_contrast": true,
  "clip_limit": 1.8,
  "clahe_tile_grid_size": [8, 8],
  "remove_color_bg": true,
  "deblur": true,
  "deblur_method": "unsharp",
  "deblur_strength": 0.35,
  "sharpen": false,
  "binarize": true,
  "binarize_method": "adaptive_gaussian",
  "adaptive_block_size": 51,
  "adaptive_C": 9,
  "post_morphology": true,
  "morphology_mode": "open",
  "morphology_kernel": [2, 2],
  "morphology_iterations": 1,
  "preserve_fine_details": true,
  "deskew": false,
  "auto_invert": false
}
```

### `grayscale_auto` 🤖 (NUEVO)
**Escala de grises + auto-rotación inteligente completa**

Detecta automáticamente la rotación óptima (0°, 90°, 180°, 270°) y convierte a escala de grises.

```json
{
  "convert_to_grayscale": true,
  "auto_rotate_all": true,
  "upscale": false,
  "enhance_contrast": false,
  "remove_color_bg": false,
  "deskew": false,
  "denoise": false,
  "sharpen": false,
  "binarize": false,
  "auto_invert": false,
  "smart_table_analysis": false,
  "auto_crop_table": false,
  "deblur": false
}
```

**Caso de uso:**
- ✅ **Ideal cuando no sabes cómo está rotada la tabla**
- Analiza automáticamente las 4 orientaciones posibles
- Elige la mejor basándose en distribución de texto y estructura
- Sin procesamiento extra, solo conversión y rotación

**Ejemplo de uso:**
```json
{
  "image_url": "https://...",
  "preset": "grayscale_auto"
}
```

### `grayscale_only`
**Solo escala de grises, sin ningún otro procesamiento**

Para cuando ya sabes la orientación correcta o quieres rotación manual.

```json
{
  "convert_to_grayscale": true,
  "upscale": false,
  "enhance_contrast": false,
  "remove_color_bg": false,
  "deskew": false,
  "denoise": false,
  "sharpen": false,
  "binarize": false,
  "auto_invert": false,
  "smart_table_analysis": false,
  "auto_crop_table": false,
  "deblur": false
}
```

**Uso con rotación manual:**
```json
{
  "preset": "grayscale_only",
  "rotate_90": true
}
```

### `minimal`
```json
{
  "upscale": false,
  "enhance_contrast": true,
  "remove_color_bg": false,
  "deskew": false,
  "denoise": false,
  "auto_invert": true
}
```

---

## 📝 Ejemplos de Uso

### Ejemplo 1: Tabla Nutricional con Fondo Rojo
```json
{
  "image_url": "https://example.com/tabla-roja.jpg",
  "force_strategy": "red_background_advanced",
  "upscale": true,
  "deblur": true
}
```

### Ejemplo 2: Documento Estándar de Alta Calidad
```json
{
  "image_url": "https://example.com/documento.jpg",
  "preset": "minimal"
}
```

### Ejemplo 3: Imagen Muy Degradada
```json
{
  "image_url": "https://example.com/imagen-mala.jpg",
  "preset": "table_ocr_aggressive"
}
```

### Ejemplo 4: Análisis Automático Inteligente
```json
{
  "image_url": "https://example.com/tabla.jpg",
  "smart_table_analysis": true,
  "upscale": true,
  "min_size": 1200
}
```

### Ejemplo 5: Personalizado Sin Preset
```json
{
  "image_url": "https://example.com/custom.jpg",
  "upscale": true,
  "min_size": 1000,
  "enhance_contrast": true,
  "clip_limit": 5.0,
  "deskew": true,
  "denoise": true,
  "denoise_method": "bilateral",
  "auto_invert": true
}
```

### Ejemplo 6: OCR de Tabla Nutricional (con símbolos y números) ⭐⭐
**✅ RECOMENDADO:** Para extraer texto con OCR
```json
{
  "image_url": "https://example.com/tabla-nutricional.pdf",
  "preset": "ocr_preserve_details"
}
```

**Preserva:** Comas (7,2), símbolos (<1%), asteriscos (*), porcentajes (0,1%)

### Ejemplo 7: Detectar Líneas/Celdas de Tabla
**⚠️ Solo para detección de estructura** (no para OCR)
```json
{
  "image_url": "https://example.com/tabla-compleja.jpg",
  "preset": "small_text_sharp"
}
```

**Ventaja:** Bordes más gruesos y definidos  
**Desventaja:** Puede perder detalles finos (comas, puntos, símbolos)

### Ejemplo 8: Solo Escala de Grises (sin filtros)
**✅ IDEAL:** Para convertir a escala de grises sin aplicar ningún filtro
```json
{
  "image_url": "https://example.com/tabla.jpg",
  "preset": "grayscale_only"
}
```

### Ejemplo 9: Escala de Grises + Auto-Rotación Inteligente 🤖
**✅ IDEAL:** Cuando no sabes cómo está rotada la tabla
```json
{
  "image_url": "https://example.com/tabla-rotada.jpg",
  "preset": "grayscale_auto"
}
```

**Ventajas:**
- ✅ Detecta automáticamente la rotación óptima (0°, 90°, 180°, 270°)
- ✅ No necesitas saber cómo está orientada la imagen
- ✅ Analiza distribución de texto y estructura
- ✅ Solo escala de grises, sin otros filtros

### Ejemplo 10: Escala de Grises + Rotación Manual
**✅ IDEAL:** Cuando ya sabes la rotación exacta
```json
{
  "image_url": "https://example.com/tabla-rotada.jpg",
  "preset": "grayscale_only",
  "rotate_90": true
}
```

**Casos de uso:**
- `rotate_90: true` → Tabla horizontal que necesitas vertical
- `rotate_180: true` → Tabla al revés (cabeza abajo)
- `rotate_270: true` → Tabla vertical que necesitas horizontal

---

## 📤 Respuesta del Endpoint

```json
{
  "success": true,
  "processed_image": "base64...",
  "original_size": {
    "w": 800,
    "h": 600
  },
  "processed_size": {
    "w": 1600,
    "h": 1200
  },
  "preprocessing_metadata": {
    "applied_operations": [
      "upscale_2.0x",
      "smart_conversion_red_background_advanced",
      "median_blur"
    ],
    "smart_analysis_used": true,
    "strategy": "red_background_advanced",
    "strategy_forced": true,
    "color_analysis": {
      "text_color": [255, 255, 255],
      "text_luminosity": 0.92,
      "bg_color": [180, 42, 38],
      "bg_luminosity": 0.35,
      "contrast": 0.57
    },
    "detected_regions": 1
  }
}
```

---

## 🎯 Recomendaciones

### Para Tablas Nutricionales
- Usa `preset: "red_table_blurry"` o `force_strategy: "red_background_advanced"`
- Activa `deblur: true` si el texto está borroso
- Aumenta `min_size` a 1000-1200 para mejor calidad

### Para Gemini/GPT-Vision/Claude ⭐⭐⭐ (MEJOR)
- **Usa `preset: "gemini_vision"`** (¡RECOMENDADO!)
- Específicamente diseñado para evitar que los LLM alucinen
- SIN binarización (escala de grises natural)
- Ultra-alta resolución (3200px) + procesamiento mínimo
- Gemini lee correctamente: 7,2 (no 7,1), 344 kJ (no 342), <1% (no inventa)

### Para OCR Tradicional (Tesseract)
- **Usa `preset: "ocr_ultra_fine"`**
- Control granular: CLAHE suave + bilateral + adaptive threshold + morfología
- Evita engrosar trazos (deblur_strength: 0.35)
- Mantiene comas (7,2), símbolos (<1%), asteriscos (*)

### Para OCR General (menos agresivo)
- **Usa `preset: "ocr_preserve_details"`**
- Más suave, menos procesamiento
- Bueno para imágenes de mejor calidad

### Para Detectar Estructura/Líneas de Tabla
- Usa `preset: "small_text_sharp"` 
- Mejor para table detection
- ⚠️ Advertencia: Puede engrosar trazos y perder detalles finos

### Para Documentos Estándar
- Usa `preset: "table_ocr"` (default)
- No necesitas modificar opciones

### Para Imágenes de Baja Calidad
- Usa `preset: "table_ocr_aggressive"`
- Activa `sharpen: true`
- Usa `denoise_method: "nlm"`

### Para Texto Blanco sobre Fondos de Color
- Usa `preset: "white_text_on_color"`
- O activa `extract_white_text: true`

### Para Detección Automática
- Usa `preset: "smart_auto"`
- Deja que el sistema decida la mejor estrategia

### Para Conversión a Escala de Grises
- **Con auto-rotación:** Usa `preset: "grayscale_auto"` 🎯
  - Detecta automáticamente la rotación óptima (0-90-180-270°)
  - **RECOMENDADO** cuando no sabes cómo está orientada la imagen
  - Analiza distribución de texto, líneas horizontales, aspect ratio
- **Sin auto-rotación:** Usa `preset: "grayscale_only"`
  - Cuando ya sabes la orientación correcta
  - Combina con `rotate_90`, `rotate_180`, o `rotate_270` para rotación manual
- **Casos de uso:**
  - Preparar imágenes para procesamiento posterior
  - Reducir tamaño de archivo manteniendo calidad
  - Enderezar tablas rotadas automáticamente
  - Normalizar orientación de múltiples imágenes

---

## ⚠️ Notas Importantes

1. **Formato Plano**: Todas las opciones van en el nivel raíz del JSON (sin objeto `options` anidado)
2. **Prioridad**: Las opciones explícitas sobrescriben el preset
3. **Mutuamente Excluyentes**: Solo una de estas puede estar en `true`:
   - `remove_color_bg`
   - `extract_white_text`
   - `extract_text_adaptive`
4. **Performance**: Más opciones activadas = mayor tiempo de procesamiento
5. **Base64**: Si usas `image_base64`, omite el prefijo `data:image/...;base64,`

---

## 🔗 Endpoints Disponibles

### `/preprocess` (POST)
Devuelve JSON con la imagen en base64 y metadata completa.

**Request:**
```json
{
  "image_url": "https://...",
  "force_strategy": "white_on_black",
  "upscale": true
}
```

**Response:**
```json
{
  "success": true,
  "processed_image": "base64...",
  "original_size": {"w": 800, "h": 600},
  "processed_size": {"w": 1600, "h": 1200},
  "preprocessing_metadata": {...}
}
```

---

### `/preprocess-image` (POST) ⭐ NUEVO
Devuelve la imagen procesada **directamente** como archivo (no JSON).

**Request:**
```json
{
  "image_url": "https://...",
  "force_strategy": "white_on_black",
  "upscale": true,
  "format": "png"
}
```

**Opciones adicionales:**
- `format`: `"png"` (default), `"jpg"`, `"jpeg"`, `"webp"`

**Response:**
- Archivo de imagen directamente
- Content-Type: `image/png`, `image/jpeg`, o `image/webp`

**Headers de Respuesta:**
- `X-Original-Width`: Ancho de la imagen original
- `X-Original-Height`: Alto de la imagen original
- `X-Processed-Width`: Ancho de la imagen procesada
- `X-Processed-Height`: Alto de la imagen procesada
- `X-Applied-Operations`: Operaciones aplicadas (separadas por comas)
- `X-Smart-Analysis`: `true` si se usó análisis inteligente
- `X-Strategy`: Estrategia usada (si aplica)

**Ejemplo de uso con curl:**
```bash
# Descargar imagen procesada
curl -X POST http://localhost:5000/preprocess-image \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg", "upscale": true}' \
  --output processed.png

# Ver headers de metadata
curl -X POST http://localhost:5000/preprocess-image \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg"}' \
  -I
```

**Ejemplo de uso con JavaScript:**
```javascript
const response = await fetch('http://localhost:5000/preprocess-image', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image_url: 'https://example.com/image.jpg',
    force_strategy: 'red_background_advanced',
    upscale: true,
    format: 'png'
  })
});

// Obtener metadata de los headers
const originalWidth = response.headers.get('X-Original-Width');
const processedWidth = response.headers.get('X-Processed-Width');
const operations = response.headers.get('X-Applied-Operations');

// Obtener la imagen como blob
const blob = await response.blob();
const imageUrl = URL.createObjectURL(blob);

// Usar la imagen
document.getElementById('myImage').src = imageUrl;
```

**Ventajas:**
- ✅ Más eficiente (no hay codificación base64)
- ✅ Menos uso de memoria
- ✅ Descarga directa con curl/wget
- ✅ Fácil integración con `<img>` en HTML
- ✅ Metadata disponible en headers HTTP

---

### `/analyze` (POST)
Analiza una imagen y sugiere operaciones recomendadas.

**Request:**
```json
{
  "image_url": "https://..."
}
```

---

### `/extract-pdf-fe` (POST)
Extrae texto embebido de PDFs.

**Request:**
```json
{
  "pdf_url": "https://..."
}
```

---

## 📊 Comparación de Endpoints

| Característica | `/preprocess` | `/preprocess-image` |
|----------------|---------------|---------------------|
| Formato de respuesta | JSON con base64 | Archivo de imagen directo |
| Metadata | ✅ En JSON | ✅ En headers HTTP |
| Tamaño de respuesta | ~33% más grande | Más pequeño (sin base64) |
| Uso en navegador | Requiere decodificar base64 | Directo en `<img src>` |
| Descarga con curl | Requiere parsear JSON | Directo con `--output` |
| Uso de memoria | Mayor | Menor |
| **Recomendado para** | APIs, JavaScript avanzado | Visualización, descargas, CLIs |

---

**Última actualización:** Enero 2026
