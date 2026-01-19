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
| `minimal` | Mínimo procesamiento | Imágenes de alta calidad |

### Uso:
```json
{
  "image_url": "https://...",
  "preset": "red_table_blurry"
}
```

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
| `deblur_method` | `string` | `"unsharp"` | Método de deblur: `"unsharp"` o `"laplacian"` |

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "deblur": true,
  "deblur_method": "unsharp"
}
```

---

### 📐 Escalado (Upscale)

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `upscale` | `boolean` | `true` | Agranda imágenes pequeñas |
| `min_size` | `integer` | `800` | Tamaño mínimo en píxeles |
| `max_scale` | `float` | `3.0` | Factor máximo de escalado |

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "upscale": true,
  "min_size": 1000,
  "max_scale": 3.0
}
```

---

### 🔄 Rotación

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `rotate_180` | `boolean` | `false` | Rota la imagen 180 grados |

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "rotate_180": true
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

### 🎨 Eliminación de Fondos

| Opción | Tipo | Default | Descripción |
|--------|------|---------|-------------|
| `remove_color_bg` | `boolean` | `true` | Elimina fondos de color (rojo, azul, verde, amarillo) |
| `extract_white_text` | `boolean` | `false` | Extrae texto blanco de fondos de color |
| `extract_text_adaptive` | `boolean` | `false` | Extracción adaptativa (funciona con texto claro u oscuro) |

**Nota:** Solo una de estas opciones debe estar en `true` a la vez.

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

**Ejemplo:**
```json
{
  "image_url": "https://...",
  "sharpen": true,
  "sharpen_strength": 1.5
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

## 🔗 Endpoints Adicionales

### `/analyze`
Analiza una imagen y sugiere operaciones recomendadas.

```json
{
  "image_url": "https://..."
}
```

### `/extract-pdf-fe`
Extrae texto embebido de PDFs.

```json
{
  "pdf_url": "https://...",
}
```

---

**Última actualización:** Enero 2026
