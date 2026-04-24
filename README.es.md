<div align="center">

# ApexVision-Core

### Plataforma de Visión Computacional Empresarial

**Nivel producción · Auto-hospedado · Multi-task · Tiempo real**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Tests](https://img.shields.io/badge/tests-311%20pasando-22c55e)](tests/)
[![Cobertura](https://img.shields.io/badge/cobertura-55%25-eab308)](htmlcov/)
[![Licencia](https://img.shields.io/badge/licencia-MIT-6366f1)](LICENSE)

*Desarrollado por [Brashkie](https://github.com/Brashkie) · Hepein Oficial*

[English](README.md) · **Español**

</div>

---

## ¿Qué es ApexVision-Core?

ApexVision-Core es una plataforma de visión computacional auto-hospedada que unifica 7 modelos de inteligencia artificial de última generación bajo una única API REST. Fue construida como alternativa directa y más poderosa a Google Cloud Vision API, dándote control total sobre tus datos, modelos e infraestructura.

**7 capacidades. 1 llamada a la API. Sin dependencia de terceros.**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Entrada: imagen (base64 / URL / archivo)                           │
│  Tasks: ["detect", "classify", "ocr", "face", "embed", "depth",    │
│          "segment"]                                                  │
├─────────────┬───────────────────┬───────────────────────────────────┤
│  Detección  │  Clasificación    │  OCR                              │
│  YOLOv11   │  CLIP / ViT-B     │  EasyOCR · PaddleOCR              │
├─────────────┼───────────────────┼───────────────────────────────────┤
│  Rostros    │  Embeddings       │  Profundidad · Segmentación       │
│  InsightFace│  CLIP 512-d       │  DPT-Large · SAM                  │
└─────────────┴───────────────────┴───────────────────────────────────┘
  Salida: JSON estructurado con bboxes, texto, vectores, mapas de profundidad
```

---

## Tabla de Contenidos

1. [Arquitectura](#1-arquitectura)
2. [Stack Técnico](#2-stack-técnico)
3. [Instalación](#3-instalación)
4. [Configuración](#4-configuración)
5. [Levantar el Servidor](#5-levantar-el-servidor)
6. [Referencia de la API](#6-referencia-de-la-api)
7. [Tasks de Visión](#7-tasks-de-visión)
8. [Monitoreo con Flower](#8-monitoreo-con-flower)
9. [Tests](#9-tests)
10. [Integraciones con Clientes](#10-integraciones-con-clientes)
    - [TypeScript / Node.js](#typescript--nodejs)
    - [React / Next.js](#react--nextjs)
    - [Flutter / Dart](#flutter--dart)
    - [Kotlin / Android](#kotlin--android)
    - [Python (requests)](#python-requests)
    - [Python CustomTkinter](#python-customtkinter)
    - [Python Tkinter](#python-tkinter)
    - [JavaScript (navegador)](#javascript-navegador)
    - [cURL](#curl)
11. [Despliegue](#11-despliegue)
12. [Roadmap](#12-roadmap)
13. [Contribuir](#13-contribuir)

---

## 1. Arquitectura

```
                    ┌─────────────────────────────────────┐
                    │       Aplicaciones Cliente           │
                    │  Web · Móvil · Desktop · API         │
                    └──────────────┬──────────────────────┘
                                   │ HTTP / WS
          ┌────────────────────────▼──────────────────────────┐
          │                 ApexVision-Core                    │
          │                                                    │
          │  ┌──────────────┐  ┌────────────┐  ┌──────────┐  │
          │  │  FastAPI     │  │ Gateway TS │  │  Flower  │  │
          │  │  :8000       │  │  :3000     │  │  :5555   │  │
          │  └──────┬───────┘  └────────────┘  └──────────┘  │
          │         │                                          │
          │  ┌──────▼─────────────────────────────────────┐   │
          │  │           Pipeline de Visión               │   │
          │  │  detect · classify · ocr · face            │   │
          │  │  embed  · depth    · segment               │   │
          │  └──────────────────────────────────────────  ┘   │
          │         │                     │                    │
          │  ┌──────▼──────────┐  ┌───────▼────────────────┐  │
          │  │ Celery Worker   │  │   Celery Beat           │  │
          │  │ vision/batch    │  │   7 tareas programadas  │  │
          │  └──────┬──────────┘  └────────────────────────┘  │
          │         │                                          │
          │  ┌──────▼──────────────────────────────────────┐  │
          │  │  Redis (broker + caché)                      │  │
          │  └──────────────────────────────────────────────┘  │
          │                                                    │
          │  ┌──────────────────────┐  ┌────────────────────┐ │
          │  │  PostgreSQL 16       │  │  Delta Lake        │ │
          │  │  + Alembic           │  │  + Parquet         │ │
          │  └──────────────────────┘  └────────────────────┘ │
          └────────────────────────────────────────────────────┘
```

---

## 2. Stack Técnico

| Capa | Tecnología | Versión |
|------|-----------|---------|
| **Framework API** | FastAPI + Uvicorn | 0.115 / 0.30 |
| **Detección de objetos** | YOLOv11 (Ultralytics) | 8.3+ |
| **Clasificación** | ViT-Base · CLIP | HuggingFace 4.41 |
| **OCR** | EasyOCR · PaddleOCR · Tesseract | Triple backend |
| **Análisis facial** | InsightFace · DeepFace | Doble backend |
| **Embeddings** | CLIP ViT-B/32, L/14 · SigLIP | 512–1024d |
| **Estimación de profundidad** | DPT-Large · MiDaS v3.1 | Doble backend |
| **Segmentación** | SAM · SegFormer · Mask2Former | Meta / HF |
| **Cola de tareas** | Celery 5 + Redis | 5.6 / 7.x |
| **Planificador** | Celery Beat | 7 tareas |
| **Base de datos** | PostgreSQL 16 + SQLAlchemy 2 | Async |
| **Migraciones** | Alembic | 1.18 |
| **Storage** | Delta Lake (ACID) + Parquet | Analytics |
| **Caché** | Redis | hiredis |
| **Monitoreo** | Flower + Prometheus | :5555 / :8000/metrics |
| **Gateway TS** | Hono + Node.js | 4.4 |
| **Tests** | pytest + pytest-asyncio | 311 tests |
| **Linting** | Ruff + Mypy | Últimas versiones |

---

## 3. Instalación

### Prerrequisitos

| Requisito | Versión | Notas |
|-----------|---------|-------|
| Python | 3.11+ | Se recomienda 3.12 |
| Node.js | 20+ | Para el Gateway TS |
| PostgreSQL | 16+ | pgAdmin es opcional |
| Redis | 7.x | `winget install Redis.Redis` en Windows |
| Git | 2.x | |

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/Brashkie/ApexVision-Core.git
cd ApexVision-Core
```

### Paso 2 — Entorno virtual Python

```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### Paso 3 — Instalar dependencias Python

```bash
pip install -r python/requirements.txt
```

> Update

```bash
pip install aiofiles albucore albumentations alembic amqp annotated-doc annotated-types anyio arq arro3-core asgiref asyncpg billiard celery certifi cfgv charset-normalizer click click-didyoumean click-plugins click-repl colorama contourpy coverage cycler cython deltalake Deprecated distlib dnspython email-validator fastapi filelock flatbuffers flower fonttools fsspec greenlet grpcio gunicorn h11 h2 hf-xet hiredis hpack httpcore httptools httpx huggingface_hub humanize hyperframe hypothesis identify idna imageio importlib_metadata iniconfig Jinja2 kiwisolver kombu lazy-loader librt llvmlite loguru Mako markdown-it-py MarkupSafe matplotlib mdurl mpmath mypy mypy_extensions networkx nodeenv numba numpy onnxruntime opencv-python opencv-python-headless opentelemetry-api opentelemetry-instrumentation opentelemetry-instrumentation-asgi opentelemetry-instrumentation-fastapi opentelemetry-sdk opentelemetry-semantic-conventions opentelemetry-util-http packaging pathspec pillow platformdirs pluggy polars polars-runtime-32 pre_commit prometheus_client prompt_toolkit protobuf psutil pyarrow pydantic pydantic_core pydantic-settings Pygments PyJWT pyparsing pytest pytest-asyncio pytest-cov python-dateutil python-discovery python-dotenv python-multipart pytz PyYAML redis regex requests rich ruff safetensors scikit-image scipy shellingham simsimd six sortedcontainers sqlalchemy starlette stringzilla sympy tenacity tifffile timm tokenizers torch torchvision tornado tqdm transformers typer typing_extensions typing-inspection tzdata tzlocal ultralytics ultralytics-thop urllib3 uvicorn vine virtualenv watchfiles wcwidth websockets win32_setctime wrapt zipp
```

> No errors, it is in pip.text

**Dependencias ML por task (instalar según necesidad):**

```bash
# Detección de objetos
pip install ultralytics           # YOLOv11

# OCR — elige uno o todos
pip install easyocr               # recomendado: mejor precisión
pip install paddleocr paddlepaddle  # mejor en documentos y tablas

# Análisis facial — elige uno o todos
pip install insightface           # recomendado: más rápido
pip install deepface              # alternativa: más atributos

# Clasificación y Embeddings (se descargan automáticamente al primer uso)
pip install transformers timm     # modelos CLIP, ViT

# Profundidad y Segmentación (se descargan automáticamente desde HuggingFace Hub)
```

### Paso 4 — Instalar dependencias Node.js

```bash
npm install
```

### Paso 5 — Descargar modelo YOLO

```bash
mkdir models
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Windows
move yolo11n.pt models\yolo11n.pt

# Linux / macOS
mv yolo11n.pt models/yolo11n.pt
```

### Paso 6 — Configurar la base de datos

Abre pgAdmin o psql y ejecuta:

```sql
CREATE USER apex WITH PASSWORD 'apex';
CREATE DATABASE apexvision OWNER apex;
```

### Paso 7 — Aplicar migraciones

```bash
alembic upgrade head
```

Salida esperada:
```
INFO  Running upgrade  -> 001_initial, initial: create vision_results and batch_jobs
INFO  Running upgrade 001_initial -> 002_model_metrics, add model_metrics table
```

### Paso 8 — Ejecutar tests

```bash
# Tests unitarios (sin servicios externos)
python -m pytest tests/python/ -q

# Tests de integración (requiere Redis + PostgreSQL)
python -m pytest tests/integration/ -q

# Suite completa
python -m pytest tests/ -q
```

Resultado esperado: `311 passed`

---

## 4. Configuración

Toda la configuración se gestiona a través del archivo `.env` en la raíz del proyecto.

```env
# ── Aplicación ─────────────────────────────────────────────
DEBUG=true
LOG_LEVEL=DEBUG
PORT=8000
SECRET_KEY=tu-clave-secreta-minimo-32-caracteres
MASTER_API_KEY=tu-api-key-aqui

# ── Base de datos ───────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://apex:apex@localhost:5432/apexvision

# ── Redis ────────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# ── Machine Learning ─────────────────────────────────────────
DEVICE=cpu              # cpu | cuda | mps
YOLO_MODEL=yolo11n.pt   # yolo11n | yolo11s | yolo11m | yolo11l | yolo11x
CLIP_MODEL=openai/clip-vit-base-patch32

# ── Storage ──────────────────────────────────────────────────
DELTA_LAKE_PATH=./data/delta
PARQUET_PATH=./data/parquet
MODELS_PATH=./models

# ── Monitoreo ────────────────────────────────────────────────
FLOWER_USER=admin
FLOWER_PASSWORD=apexvision
```

### Configuración GPU (CUDA)

```env
DEVICE=cuda
```

Instala PyTorch con soporte CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 5. Levantar el Servidor

### Desarrollo (todos los servicios)

```bash
npm run dev
```

Levanta 4 procesos en paralelo:

| Servicio | URL | Descripción |
|----------|-----|-------------|
| **API** | http://localhost:8000 | Aplicación FastAPI |
| **Swagger UI** | http://localhost:8000/docs | Documentación interactiva |
| **ReDoc** | http://localhost:8000/redoc | Referencia de la API |
| **Métricas** | http://localhost:8000/metrics | Métricas Prometheus |
| **Gateway TS** | http://localhost:3000 | Demo UI + proxy TS |
| **Flower** | http://localhost:5555 | Monitor de tareas |

### Servicios individuales

```bash
# Solo la API
python -m python.main

# Celery worker
celery -A python.celery_app worker -l info -Q vision,batch,default --pool solo

# Celery Beat (planificador)
celery -A python.celery_app beat -l info

# Flower dashboard
celery -A python.celery_app flower --conf=flower_config.py --port=5555
```

### Referencia rápida de comandos

```bash
npm run dev            # Todos los servicios
npm run dev:api        # Solo API + Worker
npm run dev:no-ts      # API + Worker + Beat (sin TS)
npm run flower         # Flower dashboard
```

---

## 6. Referencia de la API

### Autenticación

Todos los endpoints (excepto `/health/*`) requieren el siguiente header:

```
X-ApexVision-Key: tu-api-key
```

### URL Base

```
http://localhost:8000/api/v1
```

### Endpoints Principales

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/vision/analyze` | Análisis multi-task de imagen |
| `POST` | `/vision/analyze/upload` | Subida de archivo para análisis |
| `POST` | `/vision/detect` | Solo detección de objetos |
| `POST` | `/vision/classify` | Solo clasificación |
| `POST` | `/vision/ocr` | OCR / extracción de texto |
| `POST` | `/vision/face` | Análisis facial |
| `POST` | `/vision/embed` | Embedding semántico |
| `POST` | `/vision/depth` | Estimación de profundidad |
| `POST` | `/vision/segment` | Segmentación de imagen |
| `GET`  | `/vision/tasks` | Listar tasks disponibles |
| `POST` | `/batch/submit` | Enviar job de batch |
| `GET`  | `/batch/{job_id}` | Estado del job de batch |
| `DELETE` | `/batch/{job_id}` | Cancelar job de batch |
| `GET`  | `/models/` | Listar modelos cargados |
| `GET`  | `/models/variants` | Variantes de YOLO disponibles |
| `DELETE` | `/models/cache` | Limpiar caché de modelos |
| `GET`  | `/health/` | Health check |
| `GET`  | `/health/live` | Liveness probe |
| `GET`  | `/health/ready` | Readiness probe |
| `GET`  | `/health/status` | Estado completo del sistema |

### Schema de Request

```json
POST /api/v1/vision/analyze

{
  "image": {
    "format": "base64",
    "data": "<imagen_en_base64>"
  },
  "tasks": ["detect", "ocr", "classify"],
  "options": {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_detections": 100,
    "top_k": 5,
    "ocr_language": "es",
    "ocr_mode": "full",
    "face_landmarks": true,
    "face_attributes": true,
    "face_embeddings": false,
    "use_cache": true,
    "classes_filter": ["person", "car"],
    "clip_labels": ["un perro", "un gato", "un auto"]
  },
  "store_result": false
}
```

### Schema de Respuesta

```json
{
  "request_id": "uuid",
  "status": "success",
  "tasks_ran": ["detect", "ocr"],
  "image_width": 1920,
  "image_height": 1080,
  "total_inference_ms": 342.5,

  "detection": {
    "boxes": [
      {
        "label": "person",
        "label_id": 0,
        "confidence": 0.94,
        "x1": 120, "y1": 80,
        "x2": 450, "y2": 820,
        "width": 330, "height": 740
      }
    ],
    "count": 1,
    "model_used": "yolo11n.pt",
    "inference_ms": 38.2
  },

  "ocr": {
    "text": "ALFABETO VINTAGE\nABCDEFGHI...",
    "blocks": [
      {
        "text": "ALFABETO VINTAGE",
        "confidence": 0.98,
        "bbox": { "x1": 0, "y1": 0, "x2": 400, "y2": 40,
                  "width": 400, "height": 40 }
      }
    ],
    "language_detected": "es",
    "inference_ms": 2341.0
  },

  "embedding": {
    "embedding": [0.0231, -0.0142, "..."],
    "dimensions": 512,
    "model_used": "openai/clip-vit-base-patch32",
    "inference_ms": 22.1
  }
}
```

---

## 7. Tasks de Visión

| Task | Modelo | Entrada | Salida | Tiempo promedio (CPU) |
|------|--------|---------|--------|-----------------------|
| `detect` | YOLOv11n | Imagen | Bounding boxes + etiquetas | 38–120ms |
| `classify` | ViT-Base / CLIP | Imagen | Top-K predicciones | 50–200ms |
| `ocr` | EasyOCR / PaddleOCR | Imagen | Texto + bloques + idioma | 500–4000ms |
| `face` | InsightFace | Imagen | Rostros + landmarks + atributos | 80–300ms |
| `embed` | CLIP ViT-B/32 | Imagen | Vector 512-d normalizado L2 | 20–80ms |
| `depth` | DPT-Large | Imagen | Mapa de profundidad JPEG + rango (m) | 300–1200ms |
| `segment` | SAM / SegFormer | Imagen | Máscaras de instancias + RLE | 500–2000ms |

### Request multi-task (ejecución paralela)

Las tasks se ejecutan en paralelo sobre la misma imagen. El costo es `max(tiempos)`, no `sum(tiempos)`:

```bash
curl -X POST http://localhost:8000/api/v1/vision/analyze \
  -H "X-ApexVision-Key: tu-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image": {"format": "url", "url": "https://ejemplo.com/imagen.jpg"},
    "tasks": ["detect", "classify", "ocr", "face", "embed"]
  }'
```

---

## 8. Monitoreo con Flower

Flower proporciona monitoreo en tiempo real de las tareas y workers de Celery.

```bash
# Iniciar Flower (requiere que el worker esté corriendo primero)
celery -A python.celery_app flower --conf=flower_config.py --port=5555
```

Acceso: **http://localhost:5555**  
Credenciales: `admin` / `apexvision` (configurable en `.env`)

### Schedule automático de Beat

| Nombre del Job | Schedule | Tarea |
|----------------|----------|-------|
| `health-check` | Cada 5 min | Verificar Redis + PostgreSQL |
| `compact-delta-vision-results` | 02:00 UTC | Compactar archivos Delta Lake (~128MB objetivo) |
| `compact-delta-batch-jobs` | 02:10 UTC | Compactar tabla batch_jobs |
| `vacuum-delta-vision-results` | 02:30 UTC | Eliminar archivos Parquet obsoletos (retención 7d) |
| `vacuum-delta-batch-jobs` | 02:40 UTC | Vacuum de batch_jobs |
| `cleanup-old-vision-results` | 03:00 UTC | Eliminar registros de más de 90 días |
| `metrics-hourly-summary` | Cada hora | Agregar métricas de latencia a `model_metrics` |

### Inspección por CLI

```bash
celery -A python.celery_app inspect active     # tareas activas
celery -A python.celery_app inspect stats      # estadísticas del worker
celery -A python.celery_app inspect scheduled  # tareas programadas
celery -A python.celery_app purge              # vaciar todas las colas
```

---

## 9. Tests

```bash
# Tests unitarios (sin servicios externos)
python -m pytest tests/python/ -v

# Tests de integración (requiere Redis + PostgreSQL)
python -m pytest tests/integration/ -v

# Suite completa
python -m pytest tests/ -q

# Con reporte de cobertura
python -m pytest tests/ --cov=python --cov-report=html
# Abrir: htmlcov/index.html

# Módulo específico
python -m pytest tests/python/test_detector.py -v

# Excluir tests lentos
python -m pytest tests/ -m "not slow"
```

### Resumen de tests

| Archivo | Tests | Descripción |
|---------|-------|-------------|
| `test_detector.py` | 16 | Parsing de YOLOv11, filtros, dibujo |
| `test_classifier.py` | 24 | Factory ViT/CLIP, caché, predicciones |
| `test_ocr_engine.py` | 39 | Triple backend, detección de idioma, bloques |
| `test_face_analyzer.py` | 22 | Doble backend, landmarks, atributos |
| `test_embedding_engine.py` | 24 | Similitud coseno, top-K, norma L2 |
| `test_depth_estimator.py` | 29 | DPT/MiDaS, colorizar, normalizar |
| `test_segmentor.py` | 33 | SAM/SegFormer, codificación/decodificación RLE |
| `test_pipeline_integration.py` | 24 | Pipeline multi-task, caché, storage |
| `test_storage.py` | 35 | Lectura/escritura Parquet, stats, exportar |
| `test_batch_tasks.py` | 16 | Ejecución de tareas Celery, progreso |
| `test_health_endpoints.py` | 8 | Health, liveness, readiness, docs |
| `test_auth.py` | 4 | Validación API key, rechazo |
| `test_vision_endpoints.py` | 28 | Todos los endpoints de visión, upload, errores |
| `test_batch_endpoints.py` | 6 | Submit, status, cancelar |
| `test_models_endpoints.py` | 3 | Listar modelos, variantes, limpiar caché |
| **Total** | **311** | **Cobertura: 55.6%** |

---

## 10. Integraciones con Clientes

### Header de Autenticación

Todos los clientes deben incluir este header en cada request:

```
X-ApexVision-Key: tu-api-key
```

---

### TypeScript / Node.js

```typescript
import ApexVisionClient from "./sdk/apexvision";
import { readFileSync } from "fs";

const client = new ApexVisionClient({
  baseUrl: "http://localhost:8000",
  apiKey:  "tu-api-key",
  timeout: 60_000,
  retries: 3,
});

// ── Analizar desde URL ─────────────────────────────────────────────
const result = await client.fromUrl(
  "https://ejemplo.com/foto.jpg",
  ["detect", "ocr"],
  { confidence_threshold: 0.5 }
);

console.log(`Detectados: ${result.detection?.count} objetos`);
console.log(`Texto extraído: ${result.ocr?.text}`);

// ── Multi-task en un solo request ──────────────────────────────────
const full = await client.analyze({
  image:   { format: "url", url: "https://ejemplo.com/foto.jpg" },
  tasks:   ["detect", "classify", "ocr", "embed"],
  options: { confidence_threshold: 0.5, top_k: 3 },
});

// ── Similitud imagen-texto con CLIP ────────────────────────────────
const sims = await client.imageTextSimilarity(
  { format: "url", url: "https://ejemplo.com/animal.jpg" },
  ["un gato", "un perro", "un pájaro", "un auto"]
);
sims.forEach(s => console.log(`${s.text}: ${(s.similarity * 100).toFixed(1)}%`));

// ── Procesamiento en batch ─────────────────────────────────────────
const { job_id } = await client.submitBatch({
  requests: urls.map(url => ({
    image: { format: "url", url },
    tasks: ["detect"],
  })),
  job_name: "catalogo-productos-batch",
});

const status = await client.waitForBatch(job_id, { pollIntervalMs: 2000 });
console.log(`Listo: ${status.completed}/${status.total}`);
```

---

### React / Next.js

```tsx
// hooks/useApexVision.ts
import { useState, useCallback } from "react";

export function useApexVision(apiKey: string, baseUrl = "http://localhost:8000") {
  const [result,  setResult]  = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const analyze = useCallback(async (
    file: File,
    tasks: string[] = ["detect"],
    confidence = 0.5
  ) => {
    setLoading(true);
    setError(null);
    try {
      const b64 = await new Promise<string>((res, rej) => {
        const reader = new FileReader();
        reader.onload  = () => res((reader.result as string).split(",")[1]);
        reader.onerror = () => rej(new Error("Error leyendo archivo"));
        reader.readAsDataURL(file);
      });

      const resp = await fetch(`${baseUrl}/api/v1/vision/analyze`, {
        method:  "POST",
        headers: {
          "Content-Type":    "application/json",
          "X-ApexVision-Key": apiKey,
        },
        body: JSON.stringify({
          image:   { format: "base64", data: b64 },
          tasks,
          options: { confidence_threshold: confidence },
        }),
      });
      if (!resp.ok) throw new Error(`Error API: ${resp.status}`);
      setResult(await resp.json());
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [apiKey, baseUrl]);

  return { result, loading, error, analyze };
}

// ── Componente ─────────────────────────────────────────────────────
export default function AnalizadorVisual() {
  const { analyze, result, loading, error } = useApexVision(
    process.env.NEXT_PUBLIC_APEX_KEY!
  );

  const handleArchivo = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) analyze(file, ["detect", "ocr", "classify"]);
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <input type="file" accept="image/*" onChange={handleArchivo} className="mb-4" />

      {loading && <p className="text-blue-500">Analizando...</p>}
      {error   && <p className="text-red-500">{error}</p>}

      {result && (
        <div className="space-y-4">
          {result.detection && (
            <div className="p-4 border rounded-lg">
              <h3 className="font-bold">
                Detección — {result.detection.count} objetos
              </h3>
              {result.detection.boxes.map((box: any, i: number) => (
                <div key={i} className="flex justify-between text-sm mt-1">
                  <span>{box.label}</span>
                  <span>{(box.confidence * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}

          {result.ocr?.text && (
            <div className="p-4 border rounded-lg">
              <h3 className="font-bold mb-2">Texto Extraído</h3>
              <pre className="text-sm bg-gray-50 p-3 rounded whitespace-pre-wrap">
                {result.ocr.text}
              </pre>
            </div>
          )}

          <p className="text-xs text-gray-400">
            Total: {result.total_inference_ms?.toFixed(1)}ms
          </p>
        </div>
      )}
    </div>
  );
}
```

**.env.local:**
```env
NEXT_PUBLIC_APEX_URL=http://localhost:8000
NEXT_PUBLIC_APEX_KEY=tu-api-key
```

---

### Flutter / Dart

**pubspec.yaml:**
```yaml
dependencies:
  http: ^1.2.0
  image_picker: ^1.1.0
```

```dart
// lib/services/apex_vision_service.dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

class ApexVisionService {
  final String baseUrl;
  final String apiKey;

  const ApexVisionService({
    required this.baseUrl,
    required this.apiKey,
  });

  Map<String, String> get _headers => {
    'Content-Type':     'application/json',
    'X-ApexVision-Key': apiKey,
  };

  Future<Map<String, dynamic>> analizar(
    Uint8List imageBytes, {
    List<String> tasks = const ['detect'],
    double confidence = 0.5,
  }) async {
    final b64 = base64Encode(imageBytes);

    final response = await http.post(
      Uri.parse('$baseUrl/api/v1/vision/analyze'),
      headers: _headers,
      body: jsonEncode({
        'image':   {'format': 'base64', 'data': b64},
        'tasks':   tasks,
        'options': {'confidence_threshold': confidence},
      }),
    ).timeout(const Duration(seconds: 60));

    if (response.statusCode != 200) {
      throw Exception('Error API ${response.statusCode}: ${response.body}');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> detectarObjetos(Uint8List bytes) =>
      analizar(bytes, tasks: ['detect']);

  Future<Map<String, dynamic>> extraerTexto(Uint8List bytes) =>
      analizar(bytes, tasks: ['ocr']);

  Future<Map<String, dynamic>> analizarRostros(Uint8List bytes) =>
      analizar(bytes, tasks: ['face'], confidence: 0.6);
}

// ── Widget Flutter ─────────────────────────────────────────────────
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class PantallaVision extends StatefulWidget {
  const PantallaVision({super.key});
  @override State<PantallaVision> createState() => _PantallaVisionState();
}

class _PantallaVisionState extends State<PantallaVision> {
  final _service = ApexVisionService(
    baseUrl: 'http://192.168.1.x:8000',  // ← IP de tu servidor
    apiKey:  'tu-api-key',
  );
  final _picker = ImagePicker();

  Map<String, dynamic>? _resultado;
  bool _cargando = false;
  String? _error;

  Future<void> _seleccionarYAnalizar() async {
    final picked = await _picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() { _cargando = true; _error = null; });
    try {
      final bytes     = await picked.readAsBytes();
      final resultado = await _service.analizar(
        bytes,
        tasks:      ['detect', 'ocr', 'classify'],
        confidence: 0.5,
      );
      setState(() { _resultado = resultado; });
    } catch (e) {
      setState(() { _error = e.toString(); });
    } finally {
      setState(() { _cargando = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('ApexVision')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ElevatedButton.icon(
              onPressed: _cargando ? null : _seleccionarYAnalizar,
              icon: const Icon(Icons.image_search),
              label: Text(_cargando ? 'Analizando...' : 'Seleccionar Imagen'),
            ),
            const SizedBox(height: 16),

            if (_error != null)
              Text(_error!, style: const TextStyle(color: Colors.red)),

            if (_resultado != null) ...[
              if (_resultado!['detection'] != null) ...[
                Text('📦 ${_resultado!['detection']['count']} objetos detectados',
                  style: const TextStyle(fontWeight: FontWeight.bold)),
                ...(_resultado!['detection']['boxes'] as List).take(5).map((b) =>
                  Text('  • ${b['label']} — ${(b['confidence']*100).toStringAsFixed(1)}%')
                ),
                const SizedBox(height: 8),
              ],

              if (_resultado!['ocr'] != null &&
                  (_resultado!['ocr']['text'] as String).isNotEmpty) ...[
                const Text('📝 Texto extraído:',
                  style: TextStyle(fontWeight: FontWeight.bold)),
                Container(
                  margin: const EdgeInsets.only(top: 4),
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.grey[100],
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(_resultado!['ocr']['text'] as String,
                    style: const TextStyle(fontFamily: 'monospace')),
                ),
              ],

              Text(
                'Total: ${_resultado!['total_inference_ms'].toStringAsFixed(1)}ms',
                style: const TextStyle(color: Colors.grey, fontSize: 12),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
```

---

### Kotlin / Android

**build.gradle.kts:**
```kotlin
dependencies {
    implementation("com.squareup.retrofit2:retrofit:2.11.0")
    implementation("com.squareup.retrofit2:converter-gson:2.11.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.0")
}
```

```kotlin
// ApexVisionClient.kt
import android.graphics.Bitmap
import android.util.Base64
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import okhttp3.*
import retrofit2.*
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

// ── Modelos de datos ──────────────────────────────────────────────

data class ImageInput(
    val format: String = "base64",
    val data: String? = null,
    val url:  String? = null,
)

data class OpcionesVision(
    @SerializedName("confidence_threshold") val confianza: Float = 0.5f,
    @SerializedName("top_k") val topK: Int = 5,
)

data class RequestVision(
    val image: ImageInput,
    val tasks: List<String>,
    val options: OpcionesVision = OpcionesVision(),
)

data class BoundingBox(
    val label: String, val confidence: Float,
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
)

data class ResultadoDeteccion(
    val boxes: List<BoundingBox>,
    val count: Int,
    @SerializedName("model_used")   val modelo: String,
    @SerializedName("inference_ms") val tiempoMs: Float,
)

data class ResultadoOCR(
    val text: String,
    @SerializedName("language_detected") val idioma: String,
    @SerializedName("inference_ms")      val tiempoMs: Float,
)

data class RespuestaVision(
    @SerializedName("request_id")         val requestId: String,
    val status: String,
    @SerializedName("tasks_ran")          val tasksEjecutadas: List<String>,
    val detection: ResultadoDeteccion?,
    val ocr: ResultadoOCR?,
    @SerializedName("image_width")        val ancho: Int,
    @SerializedName("image_height")       val alto: Int,
    @SerializedName("total_inference_ms") val tiempoTotalMs: Float,
)

// ── Interfaz Retrofit ─────────────────────────────────────────────

interface ApexVisionApi {
    @POST("api/v1/vision/analyze")
    suspend fun analizar(@Body request: RequestVision): RespuestaVision
}

// ── Cliente ───────────────────────────────────────────────────────

class ApexVisionClient(baseUrl: String, apiKey: String) {

    private val api = Retrofit.Builder()
        .baseUrl(baseUrl.trimEnd('/') + "/")
        .client(
            OkHttpClient.Builder()
                .connectTimeout(60, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .addInterceptor { chain ->
                    chain.proceed(
                        chain.request().newBuilder()
                            .addHeader("X-ApexVision-Key", apiKey)
                            .build()
                    )
                }
                .build()
        )
        .addConverterFactory(GsonConverterFactory.create())
        .build()
        .create(ApexVisionApi::class.java)

    suspend fun analizarBitmap(
        bitmap: Bitmap,
        tasks: List<String> = listOf("detect"),
        confianza: Float = 0.5f,
    ): RespuestaVision = api.analizar(
        RequestVision(
            image   = ImageInput(data = bitmapABase64(bitmap)),
            tasks   = tasks,
            options = OpcionesVision(confianza = confianza),
        )
    )

    suspend fun detectar(bitmap: Bitmap) = analizarBitmap(bitmap, listOf("detect"))
    suspend fun extraerTexto(bitmap: Bitmap) = analizarBitmap(bitmap, listOf("ocr"))

    private fun bitmapABase64(bitmap: Bitmap): String {
        val out = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
        return Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP)
    }
}

// ── ViewModel ─────────────────────────────────────────────────────

sealed class EstadoVision {
    object Inactivo  : EstadoVision()
    object Cargando  : EstadoVision()
    data class Exito(val datos: RespuestaVision) : EstadoVision()
    data class Error(val mensaje: String)        : EstadoVision()
}

class VisionViewModel : ViewModel() {
    private val cliente = ApexVisionClient(
        baseUrl = "http://192.168.1.x:8000",
        apiKey  = "tu-api-key",
    )

    private val _estado = MutableStateFlow<EstadoVision>(EstadoVision.Inactivo)
    val estado = _estado.asStateFlow()

    fun analizar(bitmap: Bitmap, tasks: List<String> = listOf("detect", "ocr")) {
        viewModelScope.launch {
            _estado.value = EstadoVision.Cargando
            _estado.value = runCatching {
                EstadoVision.Exito(cliente.analizarBitmap(bitmap, tasks))
            }.getOrElse {
                EstadoVision.Error(it.message ?: "Error desconocido")
            }
        }
    }
}
```

---

### Python (requests)

```python
import requests
import base64
from pathlib import Path

BASE_URL = "http://localhost:8000"
API_KEY  = "tu-api-key"

HEADERS = {
    "X-ApexVision-Key": API_KEY,
    "Content-Type":     "application/json",
}

def analizar_archivo(ruta: str, tasks: list[str], confianza: float = 0.5) -> dict:
    """Analiza un archivo de imagen local."""
    datos = base64.b64encode(Path(ruta).read_bytes()).decode()
    respuesta = requests.post(
        f"{BASE_URL}/api/v1/vision/analyze",
        headers=HEADERS,
        json={
            "image":   {"format": "base64", "data": datos},
            "tasks":   tasks,
            "options": {"confidence_threshold": confianza},
        },
        timeout=60,
    )
    respuesta.raise_for_status()
    return respuesta.json()

def analizar_url(url: str, tasks: list[str]) -> dict:
    """Analiza una imagen desde una URL."""
    respuesta = requests.post(
        f"{BASE_URL}/api/v1/vision/analyze",
        headers=HEADERS,
        json={"image": {"format": "url", "url": url}, "tasks": tasks},
        timeout=60,
    )
    respuesta.raise_for_status()
    return respuesta.json()

def subir_archivo(ruta: str, tasks: str = "detect") -> dict:
    """Sube un archivo directamente (multipart)."""
    with open(ruta, "rb") as f:
        respuesta = requests.post(
            f"{BASE_URL}/api/v1/vision/analyze/upload",
            headers={"X-ApexVision-Key": API_KEY},
            files={"file": f},
            data={"tasks": tasks, "confidence": "0.5"},
            timeout=60,
        )
    respuesta.raise_for_status()
    return respuesta.json()

# ── Ejemplos de uso ────────────────────────────────────────────────

# Extraer texto con OCR
resultado = analizar_archivo("documento.jpg", ["ocr"])
print("Texto extraído:", resultado["ocr"]["text"])
print("Idioma:", resultado["ocr"]["language_detected"])

# Detección de objetos
resultado = analizar_archivo("foto.jpg", ["detect"], confianza=0.6)
for caja in resultado["detection"]["boxes"]:
    print(f"{caja['label']}: {caja['confidence']*100:.1f}%")

# Multi-task
resultado = analizar_url(
    "https://ejemplo.com/imagen.jpg",
    ["detect", "classify", "ocr"]
)
print(f"Tiempo total: {resultado['total_inference_ms']:.1f}ms")
print(f"Clase top: {resultado['classification']['predictions'][0]['label']}")
```

---

### Python CustomTkinter

```python
import customtkinter as ctk
import requests
import base64
import threading
from tkinter import filedialog
from pathlib import Path

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AppApexVision(ctk.CTk):
    BASE_URL = "http://localhost:8000"
    API_KEY  = "tu-api-key"

    def __init__(self):
        super().__init__()
        self.title("ApexVision-Core")
        self.geometry("900x650")
        self._ruta_imagen = None
        self._construir_ui()

    def _construir_ui(self):
        # ── Barra lateral ─────────────────────────────────────────
        sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ctk.CTkLabel(sidebar, text="ApexVision-Core",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 4))
        ctk.CTkLabel(sidebar, text="Plataforma de Visión IA",
                     font=ctk.CTkFont(size=12), text_color="gray").pack(pady=(0, 20))

        ctk.CTkLabel(sidebar, text="TASKS",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="gray").pack(anchor="w", padx=16)

        self._tasks = {}
        opciones = [
            ("detect",   "📦 Detectar objetos"),
            ("classify", "🏷️  Clasificar imagen"),
            ("ocr",      "📝 Extraer texto (OCR)"),
            ("face",     "👤 Analizar rostros"),
            ("embed",    "🔢 Generar embedding"),
        ]
        for task_id, etiqueta in opciones:
            var = ctk.BooleanVar(value=(task_id == "ocr"))
            self._tasks[task_id] = var
            ctk.CTkCheckBox(sidebar, text=etiqueta, variable=var).pack(
                anchor="w", padx=16, pady=3)

        ctk.CTkLabel(sidebar, text="Umbral de confianza",
                     font=ctk.CTkFont(size=11), text_color="gray").pack(
                     anchor="w", padx=16, pady=(16, 0))
        self._confianza = ctk.CTkSlider(sidebar, from_=0.1, to=1.0, number_of_steps=18)
        self._confianza.set(0.5)
        self._confianza.pack(padx=16, fill="x")
        self._lbl_conf = ctk.CTkLabel(sidebar, text="0.50",
                                       font=ctk.CTkFont(size=12, weight="bold"))
        self._lbl_conf.pack()
        self._confianza.configure(command=lambda v:
            self._lbl_conf.configure(text=f"{v:.2f}"))

        ctk.CTkButton(sidebar, text="📂 Abrir imagen",
                      command=self._elegir_archivo).pack(padx=16, pady=(20, 8), fill="x")
        self._btn = ctk.CTkButton(
            sidebar, text="🚀 Analizar",
            command=self._analizar, state="disabled",
            fg_color="#6366f1", hover_color="#4f46e5")
        self._btn.pack(padx=16, fill="x")

        self._lbl_estado = ctk.CTkLabel(sidebar, text="",
                                         font=ctk.CTkFont(size=11),
                                         text_color="gray", wraplength=200)
        self._lbl_estado.pack(padx=16, pady=8)

        # ── Panel principal ───────────────────────────────────────
        main = ctk.CTkFrame(self)
        main.pack(side="left", fill="both", expand=True, padx=16, pady=16)

        ctk.CTkLabel(main, text="Resultados",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(
                     anchor="w", pady=(0, 8))

        self._texto = ctk.CTkTextbox(
            main, font=ctk.CTkFont(family="Courier", size=12), wrap="word")
        self._texto.pack(fill="both", expand=True)
        self._texto.insert("end", "Sube una imagen y haz click en Analizar...\n")
        self._texto.configure(state="disabled")

    def _elegir_archivo(self):
        ruta = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if ruta:
            self._ruta_imagen = ruta
            self._lbl_estado.configure(text=f"📷 {Path(ruta).name}")
            self._btn.configure(state="normal")

    def _analizar(self):
        tasks = [t for t, v in self._tasks.items() if v.get()]
        if not tasks:
            self._actualizar_texto("⚠️  Selecciona al menos una task.\n")
            return
        self._btn.configure(state="disabled", text="⏳ Analizando...")
        self._actualizar_texto("Enviando a la API...\n")
        threading.Thread(target=self._llamar_api, args=(tasks,), daemon=True).start()

    def _llamar_api(self, tasks: list):
        try:
            datos = base64.b64encode(Path(self._ruta_imagen).read_bytes()).decode()
            resp  = requests.post(
                f"{self.BASE_URL}/api/v1/vision/analyze",
                headers={"X-ApexVision-Key": self.API_KEY,
                         "Content-Type": "application/json"},
                json={"image":   {"format": "base64", "data": datos},
                      "tasks":   tasks,
                      "options": {"confidence_threshold": self._confianza.get()}},
                timeout=90,
            )
            resp.raise_for_status()
            self.after(0, self._mostrar_resultado, resp.json())
        except Exception as e:
            self.after(0, self._actualizar_texto, f"❌ Error: {e}\n")
        finally:
            self.after(0, self._btn.configure,
                       {"state": "normal", "text": "🚀 Analizar"})

    def _mostrar_resultado(self, data: dict):
        lineas = []
        lineas.append(f"✅  request_id  : {data['request_id']}")
        lineas.append(f"    resolución  : {data['image_width']}×{data['image_height']}px")
        lineas.append(f"    tiempo total: {data['total_inference_ms']:.1f}ms")
        lineas.append(f"    tasks       : {', '.join(data['tasks_ran'])}")
        lineas.append("")

        if data.get("detection"):
            det = data["detection"]
            lineas.append(f"📦  DETECCIÓN — {det['count']} objetos")
            for b in det["boxes"][:5]:
                lineas.append(f"    • {b['label']:<15} {b['confidence']*100:.1f}%")
            lineas.append("")

        if data.get("classification"):
            clf = data["classification"]
            lineas.append("🏷️   CLASIFICACIÓN")
            for i, p in enumerate(clf["predictions"][:3], 1):
                lineas.append(f"    #{i} {p['label']:<20} {p['confidence']*100:.1f}%")
            lineas.append("")

        if data.get("ocr") and data["ocr"].get("text"):
            ocr = data["ocr"]
            lineas.append(f"📝  OCR — {len(ocr['text'])} chars · idioma: {ocr['language_detected']}")
            lineas.append("    ┌────────────────────────────────────────┐")
            for linea in ocr["text"].splitlines():
                lineas.append(f"    │  {linea}")
            lineas.append("    └────────────────────────────────────────┘")
            lineas.append("")

        if data.get("face"):
            face = data["face"]
            lineas.append(f"👤  ROSTROS — {face['count']} detectados")
            for i, f in enumerate(face["faces"][:3], 1):
                a = f.get("attributes", {})
                info = []
                if a.get("age"):    info.append(f"edad:{a['age']}")
                if a.get("gender"): info.append(f"género:{a['gender']}")
                if a.get("emotion"):info.append(f"emoción:{a['emotion']}")
                lineas.append(f"    #{i} {' · '.join(info)}")

        self._actualizar_texto("\n".join(lineas) + "\n")

    def _actualizar_texto(self, texto: str):
        self._texto.configure(state="normal")
        self._texto.delete("1.0", "end")
        self._texto.insert("end", texto)
        self._texto.configure(state="disabled")

if __name__ == "__main__":
    app = AppApexVision()
    app.mainloop()
```

---

### Python Tkinter

```python
import tkinter as tk
from tkinter import filedialog, scrolledtext
import requests
import base64
import threading
from pathlib import Path


class AppApexVisionTk:
    BASE_URL = "http://localhost:8000"
    API_KEY  = "tu-api-key"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ApexVision-Core")
        self.root.geometry("850x600")
        self.root.configure(bg="#1a1a2e")
        self._ruta = None
        self._construir()

    def _construir(self):
        # Barra superior
        top = tk.Frame(self.root, bg="#16213e", pady=8)
        top.pack(fill="x")
        tk.Label(top, text="ApexVision-Core", bg="#16213e", fg="#6366f1",
                 font=("Helvetica", 16, "bold")).pack(side="left", padx=16)

        # Controles
        ctrl = tk.Frame(self.root, bg="#1a1a2e", pady=10)
        ctrl.pack(fill="x", padx=16)

        tk.Button(ctrl, text="📂 Abrir imagen", command=self._elegir,
                  bg="#6366f1", fg="white", relief="flat",
                  padx=12, pady=6, cursor="hand2").pack(side="left")
        self._lbl_archivo = tk.Label(ctrl, text="Sin archivo seleccionado",
                                      bg="#1a1a2e", fg="#64748b")
        self._lbl_archivo.pack(side="left", padx=10)

        tk.Label(ctrl, text="Tasks:", bg="#1a1a2e", fg="#94a3b8").pack(
            side="left", padx=(20, 4))
        self._tasks = {}
        for t in ["detect", "classify", "ocr", "face"]:
            var = tk.BooleanVar(value=(t == "ocr"))
            self._tasks[t] = var
            tk.Checkbutton(ctrl, text=t, variable=var, bg="#1a1a2e",
                           fg="white", selectcolor="#6366f1",
                           activebackground="#1a1a2e",
                           activeforeground="white").pack(side="left", padx=3)

        tk.Label(ctrl, text="Conf:", bg="#1a1a2e", fg="#94a3b8").pack(
            side="left", padx=(12, 4))
        self._confianza = tk.DoubleVar(value=0.5)
        tk.Scale(ctrl, from_=0.1, to=1.0, resolution=0.05,
                 orient="horizontal", variable=self._confianza, length=100,
                 bg="#1a1a2e", fg="white", troughcolor="#2d2d4e",
                 highlightthickness=0, showvalue=True).pack(side="left")

        self._btn = tk.Button(ctrl, text="🚀 Analizar", command=self._analizar,
                               state="disabled", bg="#10b981", fg="white",
                               relief="flat", padx=12, pady=6, cursor="hand2")
        self._btn.pack(side="right")

        # Resultados
        frame = tk.Frame(self.root, bg="#1a1a2e")
        frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        tk.Label(frame, text="Resultados", bg="#1a1a2e", fg="#94a3b8",
                 font=("Helvetica", 11, "bold")).pack(anchor="w")

        self._salida = scrolledtext.ScrolledText(
            frame, bg="#0f1117", fg="#e2e8f0",
            font=("Courier", 11), relief="flat", bd=0,
            padx=12, pady=10, insertbackground="white")
        self._salida.pack(fill="both", expand=True, pady=(4, 0))
        self._escribir("Sube una imagen y haz click en Analizar...\n")
        self._salida.configure(state="disabled")

    def _elegir(self):
        ruta = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if ruta:
            self._ruta = ruta
            self._lbl_archivo.config(text=Path(ruta).name, fg="#e2e8f0")
            self._btn.config(state="normal")

    def _analizar(self):
        tasks = [t for t, v in self._tasks.items() if v.get()]
        if not tasks: return
        self._btn.config(state="disabled", text="⏳ Analizando...")
        self._escribir("Enviando a la API...\n")
        threading.Thread(target=self._llamar, args=(tasks,), daemon=True).start()

    def _llamar(self, tasks):
        try:
            b64  = base64.b64encode(Path(self._ruta).read_bytes()).decode()
            resp = requests.post(
                f"{self.BASE_URL}/api/v1/vision/analyze",
                headers={"X-ApexVision-Key": self.API_KEY,
                         "Content-Type": "application/json"},
                json={"image":   {"format": "base64", "data": b64},
                      "tasks":   tasks,
                      "options": {"confidence_threshold": self._confianza.get()}},
                timeout=90,
            )
            resp.raise_for_status()
            self.root.after(0, self._mostrar, resp.json())
        except Exception as e:
            self.root.after(0, self._escribir, f"❌ Error: {e}\n")
        finally:
            self.root.after(0, lambda: self._btn.config(
                state="normal", text="🚀 Analizar"))

    def _mostrar(self, data):
        lineas = []
        lineas.append(f"✅ estado       : {data['status']}")
        lineas.append(f"   resolución   : {data['image_width']}x{data['image_height']}px")
        lineas.append(f"   tiempo total : {data['total_inference_ms']:.1f}ms")
        lineas.append(f"   tasks        : {', '.join(data['tasks_ran'])}")
        lineas.append("")

        if data.get("detection"):
            d = data["detection"]
            lineas.append(f"DETECCIÓN — {d['count']} objetos")
            for b in d["boxes"][:6]:
                barra = "█" * int(b["confidence"] * 20)
                lineas.append(f"  {b['label']:<14} {barra} {b['confidence']*100:.1f}%")
            lineas.append("")

        if data.get("classification"):
            c = data["classification"]
            lineas.append("CLASIFICACIÓN")
            for i, p in enumerate(c["predictions"][:5], 1):
                lineas.append(f"  #{i} {p['label']:<22} {p['confidence']*100:.1f}%")
            lineas.append("")

        if data.get("ocr") and data["ocr"].get("text"):
            o = data["ocr"]
            lineas.append(f"OCR — {len(o['text'])} chars · idioma: {o['language_detected']}")
            lineas.append("  " + "─" * 44)
            for linea in o["text"].splitlines():
                lineas.append(f"  {linea}")
            lineas.append("  " + "─" * 44)
            lineas.append("")

        if data.get("face"):
            f = data["face"]
            lineas.append(f"ROSTROS — {f['count']} detectados")
            for i, face in enumerate(f["faces"][:3], 1):
                a = face.get("attributes", {})
                partes = [f"edad:{a['age']}" if a.get("age") else "",
                          f"género:{a['gender']}" if a.get("gender") else "",
                          f"{a['emotion']}" if a.get("emotion") else ""]
                lineas.append(f"  #{i} {' | '.join(p for p in partes if p)}")

        self._escribir("\n".join(lineas) + "\n")

    def _escribir(self, texto: str):
        self._salida.configure(state="normal")
        self._salida.delete("1.0", "end")
        self._salida.insert("end", texto)
        self._salida.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    AppApexVisionTk(root)
    root.mainloop()
```

---

### JavaScript (Navegador)

```html
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>ApexVision Demo</title>
  <style>
    body { font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 0 20px; }
    pre  { background: #f1f5f9; padding: 16px; border-radius: 8px; overflow: auto; font-size: 13px; }
    button { padding: 10px 24px; background: #6366f1; color: white;
             border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }
    button:disabled { opacity: 0.5; }
    label { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 14px; }
    h1 { color: #1e293b; }
  </style>
</head>
<body>

<h1>🔍 ApexVision-Core — Demo</h1>

<input type="file" id="archivo" accept="image/*" style="margin: 12px 0; display: block">

<div style="margin: 8px 0">
  <label><input type="checkbox" value="detect"   checked> 📦 Detección</label>
  <label><input type="checkbox" value="classify"> 🏷️ Clasificación</label>
  <label><input type="checkbox" value="ocr"      checked> 📝 OCR / Texto</label>
  <label><input type="checkbox" value="face"> 👤 Rostros</label>
</div>

<br>
<button id="btnAnalizar" onclick="analizar()" disabled>🚀 Analizar Imagen</button>
<br><br>
<pre id="salida">Selecciona una imagen para comenzar...</pre>

<script>
const BASE_URL = "http://localhost:8000";
const API_KEY  = "tu-api-key";

document.getElementById("archivo").addEventListener("change", (e) => {
  document.getElementById("btnAnalizar").disabled = !e.target.files[0];
});

async function analizar() {
  const archivo = document.getElementById("archivo").files[0];
  if (!archivo) return;

  const tasks = [...document.querySelectorAll("input[type=checkbox]:checked")]
    .map(el => el.value);

  document.getElementById("btnAnalizar").disabled = true;
  document.getElementById("salida").textContent   = "⏳ Analizando...";

  try {
    const b64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload  = () => resolve(reader.result.split(",")[1]);
      reader.onerror = () => reject(new Error("Error al leer el archivo"));
      reader.readAsDataURL(archivo);
    });

    const resp = await fetch(`${BASE_URL}/api/v1/vision/analyze`, {
      method:  "POST",
      headers: {
        "Content-Type":     "application/json",
        "X-ApexVision-Key": API_KEY,
      },
      body: JSON.stringify({
        image:   { format: "base64", data: b64 },
        tasks,
        options: { confidence_threshold: 0.5 },
      }),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    let salida = `✅ estado: ${data.status} · ${data.total_inference_ms.toFixed(1)}ms\n`;
    salida    += `   tasks: ${data.tasks_ran.join(", ")}\n\n`;

    if (data.detection) {
      salida += `📦 DETECCIÓN — ${data.detection.count} objetos\n`;
      data.detection.boxes.slice(0, 5).forEach(b => {
        salida += `   • ${b.label.padEnd(12)} ${(b.confidence * 100).toFixed(1)}%\n`;
      });
      salida += "\n";
    }

    if (data.ocr?.text) {
      salida += `📝 TEXTO EXTRAÍDO\n   ${data.ocr.text.split("\n").join("\n   ")}\n\n`;
    }

    if (data.classification) {
      salida += `🏷️ CLASIFICACIÓN\n`;
      data.classification.predictions.slice(0, 3).forEach((p, i) => {
        salida += `   #${i+1} ${p.label.padEnd(20)} ${(p.confidence * 100).toFixed(1)}%\n`;
      });
    }

    document.getElementById("salida").textContent = salida;
  } catch (err) {
    document.getElementById("salida").textContent = `❌ Error: ${err.message}`;
  } finally {
    document.getElementById("btnAnalizar").disabled = false;
  }
}
</script>
</body>
</html>
```

---

### cURL

```bash
# ── Codificar imagen local ─────────────────────────────────────────
IMAGEN_B64=$(base64 -w 0 foto.jpg)         # Linux
IMAGEN_B64=$(base64 -i foto.jpg)           # macOS

# ── Detectar objetos ───────────────────────────────────────────────
curl -X POST http://localhost:8000/api/v1/vision/analyze \
  -H "X-ApexVision-Key: tu-api-key" \
  -H "Content-Type: application/json" \
  -d "{\"image\":{\"format\":\"base64\",\"data\":\"$IMAGEN_B64\"},\"tasks\":[\"detect\"]}"

# ── Extraer texto OCR ──────────────────────────────────────────────
curl -X POST http://localhost:8000/api/v1/vision/analyze \
  -H "X-ApexVision-Key: tu-api-key" \
  -H "Content-Type: application/json" \
  -d "{\"image\":{\"format\":\"base64\",\"data\":\"$IMAGEN_B64\"},\"tasks\":[\"ocr\"]}"

# ── Analizar desde URL ─────────────────────────────────────────────
curl -X POST http://localhost:8000/api/v1/vision/analyze \
  -H "X-ApexVision-Key: tu-api-key" \
  -H "Content-Type: application/json" \
  -d '{"image":{"format":"url","url":"https://ejemplo.com/img.jpg"},"tasks":["detect","ocr"]}'

# ── Subir archivo directamente ─────────────────────────────────────
curl -X POST http://localhost:8000/api/v1/vision/analyze/upload \
  -H "X-ApexVision-Key: tu-api-key" \
  -F "file=@foto.jpg" \
  -F "tasks=detect,ocr,classify" \
  -F "confidence=0.5"

# ── Health check ───────────────────────────────────────────────────
curl http://localhost:8000/health/

# ── Listar tasks disponibles ───────────────────────────────────────
curl -H "X-ApexVision-Key: tu-api-key" \
  http://localhost:8000/api/v1/vision/tasks

# ── Enviar batch ───────────────────────────────────────────────────
curl -X POST http://localhost:8000/api/v1/batch/submit \
  -H "X-ApexVision-Key: tu-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"image": {"format": "url", "url": "https://ejemplo.com/img1.jpg"}, "tasks": ["detect"]},
      {"image": {"format": "url", "url": "https://ejemplo.com/img2.jpg"}, "tasks": ["ocr"]}
    ],
    "job_name": "mi-batch"
  }'

# ── Verificar estado del batch ─────────────────────────────────────
curl -H "X-ApexVision-Key: tu-api-key" \
  http://localhost:8000/api/v1/batch/{job_id}
```

---

## 11. Despliegue

### Docker Compose

```bash
docker compose up -d

# Escalar workers
docker compose up -d --scale worker=4

# Ver logs
docker compose logs -f api
docker compose logs -f worker
```

### Variables de producción

```env
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=<clave-aleatoria-min-64-chars>
MASTER_API_KEY=<api-key-segura>
DEVICE=cuda
DATABASE_URL=postgresql+asyncpg://user:pass@db-host:5432/apexvision
REDIS_URL=redis://redis-host:6379/0
```

### Migraciones de base de datos

```bash
alembic upgrade head          # aplicar todas las migraciones
alembic downgrade -1          # revertir la última migración
alembic history --verbose     # historial de versiones
alembic current               # versión actual en la DB

# Generar nueva migración tras cambios en los modelos
alembic revision --autogenerate -m "agregar columna X a vision_results"
```

### Linting y verificación de tipos

```bash
ruff check python/            # verificar errores de estilo
ruff check python/ --fix      # corregir automáticamente
ruff format python/           # formatear código
mypy python/                  # verificación de tipos estáticos
```

---

## 12. Roadmap

### v2.1 — En progreso
- [ ] **Image captioning** — BLIP-2 / LLaVA para descripciones automáticas
- [ ] **Análisis de video** — procesar frames de video en batch
- [ ] **Modelos ONNX personalizados** — subir y servir tus propios modelos
- [ ] **Webhooks mejorados** — reintentos automáticos con firma HMAC
- [ ] **Rate limiting por API key** — cuotas configurables por cliente

### v2.2 — Planificado
- [ ] **Base de datos vectorial** — integración con Milvus / Qdrant
- [ ] **Soporte Multi-GPU** — distribución automática de carga
- [ ] **Model hot-swap** — cambiar modelos sin reiniciar el servidor
- [ ] **A/B testing de modelos** — comparar precisión entre variantes
- [ ] **Dashboard web propio** — UI para resultados y métricas

### v2.3 — Futuro
- [ ] **Multi-modal (ImageBind)** — imagen + texto + audio combinados
- [ ] **Fine-tuning API** — entrenar modelos con tus propios datos
- [ ] **Edge deployment** — exportar a ONNX / TensorRT para dispositivos
- [ ] **Plugin system** — agregar tasks sin modificar el core
- [ ] **API GraphQL** — alternativa flexible al REST

### v2.0 — Completado ✅
- [x] 7 engines ML (YOLO, CLIP, OCR, Face, Embed, Depth, SAM)
- [x] Pipeline async paralelo con caché Redis
- [x] Batch processing con Celery + Polars + Delta Lake
- [x] PostgreSQL + migraciones Alembic (2 versiones)
- [x] Celery Beat schedule (7 tareas automatizadas)
- [x] SDK TypeScript + ejemplos React / Flutter / Kotlin / Node
- [x] 311 tests (unitarios + integración) · cobertura 55.6%
- [x] pyproject.toml (pytest + ruff + mypy + coverage)
- [x] Dashboard de monitoreo Flower
- [x] Docker Compose para despliegue
- [x] Demo UI interactiva en localhost:3000

---

## 13. Contribuir

### Flujo de trabajo

```bash
# 1. Fork y clonar
git clone https://github.com/tu-usuario/ApexVision-Core.git
cd ApexVision-Core

# 2. Crear rama de feature
git checkout -b feature/nombre-del-feature

# 3. Hacer cambios y correr tests
python -m pytest tests/ -q

# 4. Linting y formateo
ruff check python/ --fix
ruff format python/

# 5. Commit con mensaje convencional
git commit -m "feat: agregar soporte de image captioning"

# 6. Push y abrir Pull Request
git push origin feature/nombre-del-feature
```

### Convención de commits

| Prefijo | Uso |
|---------|-----|
| `feat:` | Nueva funcionalidad |
| `fix:` | Corrección de bug |
| `test:` | Agregar/modificar tests |
| `docs:` | Documentación |
| `refactor:` | Refactorización sin cambio funcional |
| `perf:` | Mejora de rendimiento |
| `chore:` | Tareas de mantenimiento |
| `ci:` | Cambios en CI/CD |

### Agregar una nueva Task de Visión

1. Crear engine en `python/core/tu_engine.py`
2. Agregar task al enum `VisionTask` en `python/schemas/vision.py`
3. Registrar en `python/core/pipeline.py` — método `_dispatch_tasks()`
4. Agregar campo a `VisionResponse` en el schema
5. Escribir tests unitarios en `tests/python/test_tu_engine.py`
6. Actualizar esta documentación

---

<div align="center">

**ApexVision-Core** — Visión computacional de producción, sin dependencias de terceros.

Desarrollado con ❤️ · [Licencia MIT](LICENSE) · [Hepein Oficial](https://github.com/Brashkie)

</div>
