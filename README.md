<div align="center">

# ApexVision-Core

### Enterprise Computer Vision API Platform

**Production-grade · Self-hosted · Multi-task · Real-time**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Tests](https://img.shields.io/badge/tests-311%20passed-22c55e)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-55%25-eab308)](htmlcov/)
[![License](https://img.shields.io/badge/license-MIT-6366f1)](LICENSE)

*Built by [Brashkie](https://github.com/Brashkie) · Hepein Oficial*

</div>

---

## Overview

ApexVision-Core is a self-hosted computer vision platform that unifies 7 state-of-the-art AI models under a single REST API. It was built as a direct, more powerful alternative to Google Cloud Vision API — giving you full ownership of your data, models, and infrastructure.

**7 capabilities. 1 API call. Zero vendor lock-in.**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Input: image (base64 / URL / file)                                 │
│  Tasks: ["detect", "classify", "ocr", "face", "embed", "depth",    │
│          "segment"]                                                  │
├─────────────┬───────────────────┬───────────────────────────────────┤
│  Detection  │  Classification   │  OCR                              │
│  YOLOv11   │  CLIP / ViT-B     │  EasyOCR · PaddleOCR              │
├─────────────┼───────────────────┼───────────────────────────────────┤
│  Face       │  Embeddings       │  Depth · Segmentation             │
│  InsightFace│  CLIP 512-d       │  DPT-Large · SAM                  │
└─────────────┴───────────────────┴───────────────────────────────────┘
  Output: structured JSON with bboxes, text, vectors, depth maps
```

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Tech Stack](#2-tech-stack)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Running the Server](#5-running-the-server)
6. [API Reference](#6-api-reference)
7. [Vision Tasks](#7-vision-tasks)
8. [Monitoring — Flower](#8-monitoring--flower)
9. [Testing](#9-testing)
10. [Client Integrations](#10-client-integrations)
    - [TypeScript / Node.js](#typescript--nodejs)
    - [React / Next.js](#react--nextjs)
    - [Flutter / Dart](#flutter--dart)
    - [Kotlin / Android](#kotlin--android)
    - [Python (requests)](#python-requests)
    - [Python CustomTkinter](#python-customtkinter)
    - [Python Tkinter](#python-tkinter)
    - [JavaScript (browser)](#javascript-browser)
    - [cURL](#curl)
11. [Deployment](#11-deployment)
12. [Roadmap](#12-roadmap)
13. [Contributing](#13-contributing)

---

## 1. Architecture

```
                        ┌─────────────────────────────────┐
                        │         Client Applications      │
                        │  Web · Mobile · Desktop · API    │
                        └──────────────┬──────────────────┘
                                       │ HTTP / WS
              ┌────────────────────────▼────────────────────────┐
              │                ApexVision-Core                   │
              │                                                  │
              │  ┌──────────────┐  ┌────────────┐  ┌─────────┐ │
              │  │  FastAPI     │  │  TS Gateway│  │  Flower │ │
              │  │  :8000       │  │  :3000     │  │  :5555  │ │
              │  └──────┬───────┘  └────────────┘  └─────────┘ │
              │         │                                        │
              │  ┌──────▼───────────────────────────────────┐   │
              │  │           Vision Pipeline                 │   │
              │  │  detect · classify · ocr · face           │   │
              │  │  embed  · depth    · segment              │   │
              │  └──────────────────────────────────────────┘   │
              │         │                    │                   │
              │  ┌──────▼───────┐  ┌────────▼──────────────┐   │
              │  │ Celery Worker│  │   Celery Beat          │   │
              │  │ vision/batch │  │   7 scheduled tasks    │   │
              │  └──────┬───────┘  └───────────────────────┘   │
              │         │                                        │
              │  ┌──────▼──────────────────────────────────┐   │
              │  │  Redis (broker + cache)                   │   │
              │  └───────────────────────────────────────────┘  │
              │                                                  │
              │  ┌──────────────────┐  ┌──────────────────────┐ │
              │  │  PostgreSQL 16   │  │  Delta Lake + Parquet │ │
              │  │  + Alembic       │  │  (analytics storage)  │ │
              │  └──────────────────┘  └──────────────────────┘ │
              └─────────────────────────────────────────────────┘
```

---

## 2. Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **API Framework** | FastAPI + Uvicorn | 0.115 / 0.30 |
| **Object Detection** | YOLOv11 (Ultralytics) | 8.3+ |
| **Classification** | ViT-Base · CLIP | HuggingFace 4.41 |
| **OCR** | EasyOCR · PaddleOCR · Tesseract | Multi-backend |
| **Face Analysis** | InsightFace · DeepFace | Dual-backend |
| **Embeddings** | CLIP ViT-B/32, L/14 · SigLIP | 512–1024d |
| **Depth Estimation** | DPT-Large · MiDaS v3.1 | Dual-backend |
| **Segmentation** | SAM · SegFormer · Mask2Former | Meta / HF |
| **Task Queue** | Celery 5 + Redis | 5.6 / 7.x |
| **Scheduler** | Celery Beat | 7 tasks |
| **Database** | PostgreSQL 16 + SQLAlchemy 2 | Async |
| **Migrations** | Alembic | 1.18 |
| **Storage** | Delta Lake (ACID) + Parquet | Analytics |
| **Cache** | Redis | hiredis |
| **Monitoring** | Flower + Prometheus | :5555 / :8000/metrics |
| **TS Gateway** | Hono + Node.js | 4.4 |
| **Testing** | pytest + pytest-asyncio | 311 tests |
| **Linting** | Ruff + Mypy | Latest |

---

## 3. Installation

### Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Python | 3.11+ | 3.12 recommended |
| Node.js | 20+ | For TS Gateway |
| PostgreSQL | 16+ | pgAdmin optional |
| Redis | 7.x | `winget install Redis.Redis` on Windows |
| Git | 2.x | |

### Step 1 — Clone the repository

```bash
git clone https://github.com/Brashkie/ApexVision-Core.git
cd ApexVision-Core
```

### Step 2 — Python virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r python/requirements.txt
```

**Core ML dependencies (required for each task):**

```bash
# Object Detection
pip install ultralytics           # YOLOv11

# OCR — choose one or all
pip install easyocr               # recommended: best accuracy
pip install paddleocr paddlepaddle  # best for documents/tables

# Face Analysis — choose one or all
pip install insightface           # recommended: fastest
pip install deepface              # alternative: more attributes

# Classification & Embeddings (auto-downloaded on first use)
pip install transformers timm     # CLIP, ViT models

# Depth & Segmentation (auto-downloaded on first use)
# Models download automatically from HuggingFace Hub
```

### Step 4 — Install Node.js dependencies

```bash
npm install
```

### Step 5 — Download YOLO model

```bash
mkdir models
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Windows
move yolo11n.pt models\yolo11n.pt

# Linux / macOS
mv yolo11n.pt models/yolo11n.pt
```

### Step 6 — Database setup

Open pgAdmin or psql and run:

```sql
CREATE USER apex WITH PASSWORD 'apex';
CREATE DATABASE apexvision OWNER apex;
```

### Step 7 — Apply migrations

```bash
alembic upgrade head
```

Expected output:
```
INFO  Running upgrade  -> 001_initial, initial: create vision_results and batch_jobs
INFO  Running upgrade 001_initial -> 002_model_metrics, add model_metrics table
```

### Step 8 — Run tests

```bash
# Unit tests (no external services required)
python -m pytest tests/python/ -q

# Integration tests (requires Redis + PostgreSQL)
python -m pytest tests/integration/ -q

# All tests
python -m pytest tests/ -q
```

Expected: `311 passed`

---

## 4. Configuration

All configuration is managed through the `.env` file in the project root.

```env
# ── Application ────────────────────────────────────────────
DEBUG=true
LOG_LEVEL=DEBUG
PORT=8000
SECRET_KEY=your-secret-key-minimum-32-characters-here
MASTER_API_KEY=your-api-key-here

# ── Database ────────────────────────────────────────────────
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

# ── Monitoring ───────────────────────────────────────────────
FLOWER_USER=admin
FLOWER_PASSWORD=apexvision
```

### GPU Configuration (CUDA)

```env
DEVICE=cuda
```

Install PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 5. Running the Server

### Development (all services)

```bash
npm run dev
```

This starts 4 processes in parallel:

| Process | URL | Description |
|---------|-----|-------------|
| **API** | http://localhost:8000 | FastAPI application |
| **Swagger UI** | http://localhost:8000/docs | Interactive API docs |
| **ReDoc** | http://localhost:8000/redoc | API reference |
| **Metrics** | http://localhost:8000/metrics | Prometheus metrics |
| **TS Gateway** | http://localhost:3000 | Demo UI + TS proxy |
| **Flower** | http://localhost:5555 | Task monitor |

### Start individual services

```bash
# API only
python -m python.main

# Celery worker
celery -A python.celery_app worker -l info -Q vision,batch,default --pool solo

# Celery Beat (scheduler)
celery -A python.celery_app beat -l info

# Flower dashboard
celery -A python.celery_app flower --conf=flower_config.py --port=5555
```

### Quick commands reference

```bash
npm run dev            # All services
npm run dev:api        # API + Worker only
npm run dev:no-ts      # API + Worker + Beat (no TS)
npm run flower         # Flower dashboard
```

---

## 6. API Reference

### Authentication

All API endpoints (except `/health/*`) require the following header:

```
X-ApexVision-Key: your-api-key
```

### Base URL

```
http://localhost:8000/api/v1
```

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/vision/analyze` | Multi-task image analysis |
| `POST` | `/vision/analyze/upload` | File upload analysis |
| `POST` | `/vision/detect` | Object detection only |
| `POST` | `/vision/classify` | Classification only |
| `POST` | `/vision/ocr` | OCR / text extraction |
| `POST` | `/vision/face` | Face analysis |
| `POST` | `/vision/embed` | Semantic embedding |
| `POST` | `/vision/depth` | Depth estimation |
| `POST` | `/vision/segment` | Image segmentation |
| `GET`  | `/vision/tasks` | List available tasks |
| `POST` | `/batch/submit` | Submit batch job |
| `GET`  | `/batch/{job_id}` | Batch job status |
| `DELETE` | `/batch/{job_id}` | Cancel batch job |
| `GET`  | `/models/` | List loaded models |
| `GET`  | `/models/variants` | YOLO model variants |
| `DELETE` | `/models/cache` | Clear model cache |
| `GET`  | `/health/` | Health check |
| `GET`  | `/health/live` | Liveness probe |
| `GET`  | `/health/ready` | Readiness probe |
| `GET`  | `/health/status` | Full system status |

### Request Schema

```json
POST /api/v1/vision/analyze

{
  "image": {
    "format": "base64",
    "data": "<base64_encoded_image>"
  },
  "tasks": ["detect", "ocr", "classify"],
  "options": {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_detections": 100,
    "top_k": 5,
    "ocr_language": "en",
    "ocr_mode": "full",
    "face_landmarks": true,
    "face_attributes": true,
    "face_embeddings": false,
    "use_cache": true,
    "classes_filter": ["person", "car"],
    "clip_labels": ["a dog", "a cat", "a car"]
  },
  "store_result": false
}
```

### Response Schema

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

  "classification": {
    "predictions": [
      { "label": "person", "confidence": 0.87, "label_id": 0 }
    ],
    "model_used": "google/vit-base-patch16-224",
    "inference_ms": 55.1
  },

  "ocr": {
    "text": "VINTAGE ALPHABET\nABCDEFGHI...",
    "blocks": [
      {
        "text": "VINTAGE ALPHABET",
        "confidence": 0.98,
        "bbox": { "x1": 0, "y1": 0, "x2": 400, "y2": 40,
                  "width": 400, "height": 40 }
      }
    ],
    "language_detected": "en",
    "inference_ms": 2341.0
  },

  "face": {
    "faces": [
      {
        "bbox": { "x1": 100, "y1": 80, "confidence": 0.97, ... },
        "landmarks": { "left_eye": {"x": 120, "y": 140}, ... },
        "attributes": { "age": 28, "gender": "female", "emotion": "happiness" }
      }
    ],
    "count": 1,
    "inference_ms": 88.4
  },

  "embedding": {
    "embedding": [0.0231, -0.0142, ...],
    "dimensions": 512,
    "model_used": "openai/clip-vit-base-patch32",
    "inference_ms": 22.1
  },

  "depth": {
    "depth_map_base64": "<jpeg_base64>",
    "min_depth": 0.5,
    "max_depth": 18.3,
    "inference_ms": 412.0
  },

  "segmentation": {
    "masks": [
      {
        "label": "object", "score": 0.92, "area": 48200,
        "bbox": { ... },
        "mask_rle": { "counts": [...], "size": [1080, 1920] },
        "backend": "sam"
      }
    ],
    "count": 3,
    "inference_ms": 890.0
  }
}
```

---

## 7. Vision Tasks

| Task | Model | Input | Output | Avg. Time (CPU) |
|------|-------|-------|--------|----------------|
| `detect` | YOLOv11n | Image | Bounding boxes + labels | 38–120ms |
| `classify` | ViT-Base / CLIP | Image | Top-K predictions | 50–200ms |
| `ocr` | EasyOCR / PaddleOCR | Image | Text + blocks + language | 500–4000ms |
| `face` | InsightFace | Image | Faces + landmarks + attributes | 80–300ms |
| `embed` | CLIP ViT-B/32 | Image | 512-d L2 normalized vector | 20–80ms |
| `depth` | DPT-Large | Image | Depth map JPEG + range (m) | 300–1200ms |
| `segment` | SAM / SegFormer | Image | Instance masks + RLE | 500–2000ms |

### Multi-task request (parallel execution)

Tasks run in parallel on the same image. Cost is `max(task_times)` not `sum(task_times)`:

```bash
# Single request — 7 tasks in parallel
curl -X POST http://localhost:8000/api/v1/vision/analyze \
  -H "X-ApexVision-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image": {"format": "url", "url": "https://example.com/image.jpg"},
    "tasks": ["detect", "classify", "ocr", "face", "embed", "depth", "segment"]
  }'
```

---

## 8. Monitoring — Flower

Flower provides real-time monitoring of Celery tasks and workers.

```bash
# Start Flower (requires worker running)
celery -A python.celery_app flower --conf=flower_config.py --port=5555
```

Access: **http://localhost:5555**  
Credentials: `admin` / `apexvision` (configurable in `.env`)

### Automated Beat Schedule

| Job Name | Schedule | Task |
|----------|----------|------|
| `health-check` | Every 5 min | Verify Redis + PostgreSQL connectivity |
| `compact-delta-vision-results` | 02:00 UTC | Compact Delta Lake files (~128MB target) |
| `compact-delta-batch-jobs` | 02:10 UTC | Compact batch_jobs table |
| `vacuum-delta-vision-results` | 02:30 UTC | Remove orphan Parquet files (7d retention) |
| `vacuum-delta-batch-jobs` | 02:40 UTC | Vacuum batch_jobs table |
| `cleanup-old-vision-results` | 03:00 UTC | Delete records older than 90 days |
| `metrics-hourly-summary` | Every hour | Aggregate latency metrics to `model_metrics` |

### CLI inspection

```bash
celery -A python.celery_app inspect active     # active tasks
celery -A python.celery_app inspect stats      # worker statistics
celery -A python.celery_app inspect scheduled  # scheduled tasks
celery -A python.celery_app purge              # clear all queues
```

---

## 9. Testing

```bash
# Unit tests (no external services)
python -m pytest tests/python/ -v

# Integration tests (requires Redis + PostgreSQL)
python -m pytest tests/integration/ -v

# Full suite
python -m pytest tests/ -q

# With coverage report
python -m pytest tests/ --cov=python --cov-report=html
# Open htmlcov/index.html

# Run specific module
python -m pytest tests/python/test_detector.py -v

# Skip slow tests
python -m pytest tests/ -m "not slow"
```

### Test Summary

| File | Tests | Description |
|------|-------|-------------|
| `test_detector.py` | 16 | YOLOv11 parsing, filtering, drawing |
| `test_classifier.py` | 24 | ViT/CLIP factory, cache, predictions |
| `test_ocr_engine.py` | 39 | Tri-backend, language detection, blocks |
| `test_face_analyzer.py` | 22 | Dual-backend, landmarks, attributes |
| `test_embedding_engine.py` | 24 | Cosine similarity, top-K, L2 norm |
| `test_depth_estimator.py` | 29 | DPT/MiDaS, colorize, normalize |
| `test_segmentor.py` | 33 | SAM/SegFormer, RLE encode/decode |
| `test_pipeline_integration.py` | 24 | Multi-task pipeline, cache, storage |
| `test_storage.py` | 35 | Parquet read/write, stats, export |
| `test_batch_tasks.py` | 16 | Celery task execution, progress |
| `test_health_endpoints.py` | 8 | Health, liveness, readiness, docs |
| `test_auth.py` | 4 | API key validation, rejection |
| `test_vision_endpoints.py` | 28 | All vision endpoints, upload, errors |
| `test_batch_endpoints.py` | 6 | Submit, status, cancel |
| `test_models_endpoints.py` | 3 | List models, variants, cache clear |
| **Total** | **311** | **Coverage: 55.6%** |

---

## 10. Client Integrations

### Authentication Header

All clients must include this header in every request:

```
X-ApexVision-Key: your-api-key
```

---

### TypeScript / Node.js

Install the SDK (included in the repository):

```bash
# Copy src/sdk/apexvision.ts to your project
# Or use with tsx/ts-node directly
```

```typescript
import ApexVisionClient, { VisionTask } from "./sdk/apexvision";
import { readFileSync } from "fs";

const client = new ApexVisionClient({
  baseUrl: "http://localhost:8000",
  apiKey:  "your-api-key",
  timeout: 60_000,
  retries: 3,
});

// ── Analyze from URL ──────────────────────────────────────────
const result = await client.fromUrl(
  "https://example.com/photo.jpg",
  ["detect", "ocr"],
  { confidence_threshold: 0.5 }
);

console.log(`Detected ${result.detection?.count} objects`);
console.log(`Extracted text: ${result.ocr?.text}`);

// ── Analyze from file ─────────────────────────────────────────
import { bufferToBase64 } from "./sdk/apexvision";
const bytes = readFileSync("./image.jpg");
const result2 = await client.fromBase64(bufferToBase64(bytes), ["classify"]);
console.log(result2.classification?.predictions[0]);

// ── Multi-task in one request ─────────────────────────────────
const full = await client.analyze({
  image:   { format: "url", url: "https://example.com/photo.jpg" },
  tasks:   ["detect", "classify", "ocr", "embed"],
  options: { confidence_threshold: 0.5, top_k: 3 },
});

// ── CLIP zero-shot similarity ─────────────────────────────────
const sims = await client.imageTextSimilarity(
  { format: "url", url: "https://example.com/animal.jpg" },
  ["a cat", "a dog", "a bird", "a car"]
);
sims.forEach(s => console.log(`${s.text}: ${(s.similarity * 100).toFixed(1)}%`));

// ── Batch processing ──────────────────────────────────────────
const { job_id } = await client.submitBatch({
  requests: imageUrls.map(url => ({
    image: { format: "url", url },
    tasks: ["detect"],
  })),
  job_name: "product-catalog-batch",
});

const status = await client.waitForBatch(job_id, { pollIntervalMs: 2000 });
console.log(`Done: ${status.completed}/${status.total}`);
```

---

### React / Next.js

```tsx
// hooks/useApexVision.ts
import { useState, useCallback } from "react";

interface VisionResult {
  detection?:      { count: number; boxes: any[] };
  classification?: { predictions: any[] };
  ocr?:            { text: string; blocks: any[] };
  total_inference_ms: number;
}

export function useApexVision(apiKey: string, baseUrl = "http://localhost:8000") {
  const [result,  setResult]  = useState<VisionResult | null>(null);
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
      const b64 = await fileToBase64(file);
      const res = await fetch(`${baseUrl}/api/v1/vision/analyze`, {
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
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      setResult(await res.json());
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [apiKey, baseUrl]);

  return { result, loading, error, analyze };
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = () => resolve((reader.result as string).split(",")[1]);
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

// ── Component ──────────────────────────────────────────────────
export default function VisionAnalyzer() {
  const { analyze, result, loading, error } = useApexVision(
    process.env.NEXT_PUBLIC_APEX_KEY!,
    process.env.NEXT_PUBLIC_APEX_URL
  );

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) analyze(file, ["detect", "ocr", "classify"]);
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <input
        type="file" accept="image/*"
        onChange={handleFile}
        className="mb-4"
      />

      {loading && <p className="text-blue-500">Analyzing...</p>}
      {error   && <p className="text-red-500">{error}</p>}

      {result && (
        <div className="space-y-4">
          {result.detection && (
            <div className="p-4 border rounded-lg">
              <h3 className="font-bold">
                Detection — {result.detection.count} objects
              </h3>
              {result.detection.boxes.map((box, i) => (
                <div key={i} className="flex justify-between text-sm mt-1">
                  <span>{box.label}</span>
                  <span>{(box.confidence * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}

          {result.ocr?.text && (
            <div className="p-4 border rounded-lg">
              <h3 className="font-bold mb-2">Extracted Text</h3>
              <pre className="text-sm bg-gray-50 p-3 rounded whitespace-pre-wrap">
                {result.ocr.text}
              </pre>
            </div>
          )}

          <p className="text-xs text-gray-400">
            Total: {result.total_inference_ms.toFixed(1)}ms
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
NEXT_PUBLIC_APEX_KEY=your-api-key
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

  Future<Map<String, dynamic>> analyze(
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
      throw Exception('API error ${response.statusCode}: ${response.body}');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  // Shortcuts
  Future<Map<String, dynamic>> detectObjects(Uint8List bytes) =>
      analyze(bytes, tasks: ['detect']);

  Future<Map<String, dynamic>> extractText(Uint8List bytes) =>
      analyze(bytes, tasks: ['ocr']);

  Future<Map<String, dynamic>> analyzeFaces(Uint8List bytes) =>
      analyze(bytes, tasks: ['face'], confidence: 0.6);
}

// ── Flutter Widget ──────────────────────────────────────────────
// lib/screens/vision_screen.dart
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class VisionScreen extends StatefulWidget {
  const VisionScreen({super.key});
  @override State<VisionScreen> createState() => _VisionScreenState();
}

class _VisionScreenState extends State<VisionScreen> {
  final _service = ApexVisionService(
    baseUrl: 'http://192.168.1.x:8000',  // ← your server IP
    apiKey:  'your-api-key',
  );
  final _picker = ImagePicker();

  Map<String, dynamic>? _result;
  bool _loading = false;
  String? _error;

  Future<void> _pickAndAnalyze() async {
    final picked = await _picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() { _loading = true; _error = null; });
    try {
      final bytes  = await picked.readAsBytes();
      final result = await _service.analyze(
        bytes,
        tasks:      ['detect', 'ocr', 'classify'],
        confidence: 0.5,
      );
      setState(() { _result = result; });
    } catch (e) {
      setState(() { _error = e.toString(); });
    } finally {
      setState(() { _loading = false; });
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
              onPressed: _loading ? null : _pickAndAnalyze,
              icon: const Icon(Icons.image_search),
              label: Text(_loading ? 'Analyzing...' : 'Select Image'),
            ),
            const SizedBox(height: 16),

            if (_error != null)
              Text(_error!, style: const TextStyle(color: Colors.red)),

            if (_result != null) ...[
              // Detection results
              if (_result!['detection'] != null) ...[
                Text(
                  '📦 ${_result!['detection']['count']} objects detected',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
                ...(_result!['detection']['boxes'] as List).take(5).map((b) =>
                  Text('  • ${b['label']} — ${(b['confidence']*100).toStringAsFixed(1)}%')
                ),
                const SizedBox(height: 8),
              ],

              // OCR results
              if (_result!['ocr'] != null &&
                  (_result!['ocr']['text'] as String).isNotEmpty) ...[
                const Text('📝 Extracted text:',
                  style: TextStyle(fontWeight: FontWeight.bold)),
                Container(
                  margin: const EdgeInsets.only(top: 4),
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.grey[100],
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    _result!['ocr']['text'] as String,
                    style: const TextStyle(fontFamily: 'monospace'),
                  ),
                ),
              ],

              Text(
                'Total: ${_result!['total_inference_ms'].toStringAsFixed(1)}ms',
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
import kotlinx.coroutines.*
import okhttp3.*
import retrofit2.*
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

// ── Data classes ─────────────────────────────────────────────────

data class ImageInput(
    val format: String = "base64",
    val data: String? = null,
    val url:  String? = null,
)

data class VisionOptions(
    @SerializedName("confidence_threshold") val confidence: Float = 0.5f,
    @SerializedName("top_k") val topK: Int = 5,
)

data class VisionRequest(
    val image: ImageInput,
    val tasks: List<String>,
    val options: VisionOptions = VisionOptions(),
)

data class BoundingBox(
    val label: String, val confidence: Float,
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
)

data class DetectionResult(
    val boxes: List<BoundingBox>,
    val count: Int,
    @SerializedName("model_used") val modelUsed: String,
    @SerializedName("inference_ms") val inferenceMs: Float,
)

data class OCRResult(
    val text: String,
    @SerializedName("language_detected") val language: String,
    @SerializedName("inference_ms") val inferenceMs: Float,
)

data class VisionResponse(
    @SerializedName("request_id") val requestId: String,
    val status: String,
    @SerializedName("tasks_ran") val tasksRan: List<String>,
    val detection: DetectionResult?,
    val ocr: OCRResult?,
    @SerializedName("image_width")  val width:  Int,
    @SerializedName("image_height") val height: Int,
    @SerializedName("total_inference_ms") val totalMs: Float,
)

// ── Retrofit interface ────────────────────────────────────────────

interface ApexVisionApi {
    @POST("api/v1/vision/analyze")
    suspend fun analyze(@Body request: VisionRequest): VisionResponse
}

// ── Client ────────────────────────────────────────────────────────

class ApexVisionClient(baseUrl: String, apiKey: String) {

    private val api: ApexVisionApi = Retrofit.Builder()
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

    suspend fun analyzeBitmap(
        bitmap: Bitmap,
        tasks: List<String> = listOf("detect"),
        confidence: Float = 0.5f,
    ): VisionResponse {
        val b64 = bitmapToBase64(bitmap)
        return api.analyze(
            VisionRequest(
                image   = ImageInput(data = b64),
                tasks   = tasks,
                options = VisionOptions(confidence = confidence),
            )
        )
    }

    suspend fun detect(bitmap: Bitmap) =
        analyzeBitmap(bitmap, listOf("detect"))

    suspend fun extractText(bitmap: Bitmap) =
        analyzeBitmap(bitmap, listOf("ocr"))

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val out = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
        return Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP)
    }
}

// ── ViewModel ─────────────────────────────────────────────────────

sealed class VisionState {
    object Idle    : VisionState()
    object Loading : VisionState()
    data class Success(val data: VisionResponse) : VisionState()
    data class Error(val message: String)        : VisionState()
}

class VisionViewModel : ViewModel() {
    private val client = ApexVisionClient(
        baseUrl = "http://192.168.1.x:8000",
        apiKey  = "your-api-key",
    )

    private val _state = MutableStateFlow<VisionState>(VisionState.Idle)
    val state = _state.asStateFlow()

    fun analyze(bitmap: Bitmap, tasks: List<String> = listOf("detect", "ocr")) {
        viewModelScope.launch {
            _state.value = VisionState.Loading
            _state.value = runCatching {
                VisionState.Success(client.analyzeBitmap(bitmap, tasks))
            }.getOrElse {
                VisionState.Error(it.message ?: "Unknown error")
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
API_KEY  = "your-api-key"

HEADERS = {
    "X-ApexVision-Key": API_KEY,
    "Content-Type":     "application/json",
}

def analyze_file(path: str, tasks: list[str], confidence: float = 0.5) -> dict:
    """Analyze a local image file."""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    response = requests.post(
        f"{BASE_URL}/api/v1/vision/analyze",
        headers=HEADERS,
        json={
            "image":   {"format": "base64", "data": data},
            "tasks":   tasks,
            "options": {"confidence_threshold": confidence},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()

def analyze_url(url: str, tasks: list[str]) -> dict:
    """Analyze an image from a URL."""
    response = requests.post(
        f"{BASE_URL}/api/v1/vision/analyze",
        headers=HEADERS,
        json={"image": {"format": "url", "url": url}, "tasks": tasks},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()

def upload_file(path: str, tasks: str = "detect") -> dict:
    """Upload a file directly (multipart)."""
    with open(path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/api/v1/vision/analyze/upload",
            headers={"X-ApexVision-Key": API_KEY},
            files={"file": f},
            data={"tasks": tasks, "confidence": "0.5"},
            timeout=60,
        )
    response.raise_for_status()
    return response.json()

# ── Usage ──────────────────────────────────────────────────────────

# OCR example
result = analyze_file("document.jpg", ["ocr"])
print("Extracted text:", result["ocr"]["text"])
print("Language:", result["ocr"]["language_detected"])

# Object detection
result = analyze_file("photo.jpg", ["detect"], confidence=0.6)
for box in result["detection"]["boxes"]:
    print(f"{box['label']}: {box['confidence']*100:.1f}%")

# Multi-task
result = analyze_url(
    "https://example.com/image.jpg",
    ["detect", "classify", "ocr"]
)
print(f"Total time: {result['total_inference_ms']:.1f}ms")
print(f"Top class: {result['classification']['predictions'][0]['label']}")
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

class ApexVisionApp(ctk.CTk):
    BASE_URL = "http://localhost:8000"
    API_KEY  = "your-api-key"

    def __init__(self):
        super().__init__()
        self.title("ApexVision-Core")
        self.geometry("900x650")
        self.resizable(True, True)
        self._image_path = None
        self._setup_ui()

    def _setup_ui(self):
        # ── Sidebar ──────────────────────────────────────────────
        sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        sidebar.pack(side="left", fill="y", padx=0, pady=0)
        sidebar.pack_propagate(False)

        ctk.CTkLabel(sidebar, text="ApexVision-Core",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 4))
        ctk.CTkLabel(sidebar, text="Ultra Vision API Platform",
                     font=ctk.CTkFont(size=12), text_color="gray").pack(pady=(0, 20))

        # Tasks
        ctk.CTkLabel(sidebar, text="TASKS",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="gray").pack(anchor="w", padx=16)

        self._tasks = {}
        task_config = [
            ("detect",   "📦 Object Detection"),
            ("classify", "🏷️  Classification"),
            ("ocr",      "📝 OCR / Text"),
            ("face",     "👤 Face Analysis"),
            ("embed",    "🔢 Embedding"),
        ]
        for task_id, label in task_config:
            var = ctk.BooleanVar(value=(task_id == "ocr"))
            self._tasks[task_id] = var
            ctk.CTkCheckBox(sidebar, text=label, variable=var).pack(
                anchor="w", padx=16, pady=3)

        ctk.CTkLabel(sidebar, text="Confidence",
                     font=ctk.CTkFont(size=11), text_color="gray").pack(
                     anchor="w", padx=16, pady=(16, 0))
        self._conf = ctk.CTkSlider(sidebar, from_=0.1, to=1.0, number_of_steps=18)
        self._conf.set(0.5)
        self._conf.pack(padx=16, fill="x")
        self._conf_label = ctk.CTkLabel(sidebar, text="0.50",
                                         font=ctk.CTkFont(size=12, weight="bold"))
        self._conf_label.pack()
        self._conf.configure(command=lambda v: self._conf_label.configure(
            text=f"{v:.2f}"))

        # Buttons
        ctk.CTkButton(sidebar, text="Select Image",
                      command=self._pick_file).pack(padx=16, pady=(20, 8), fill="x")
        self._analyze_btn = ctk.CTkButton(
            sidebar, text="🚀 Analyze",
            command=self._analyze, state="disabled",
            fg_color="#6366f1", hover_color="#4f46e5")
        self._analyze_btn.pack(padx=16, fill="x")

        self._status = ctk.CTkLabel(sidebar, text="", font=ctk.CTkFont(size=11),
                                     text_color="gray", wraplength=200)
        self._status.pack(padx=16, pady=8)

        # ── Main area ─────────────────────────────────────────────
        main = ctk.CTkFrame(self)
        main.pack(side="left", fill="both", expand=True, padx=16, pady=16)

        ctk.CTkLabel(main, text="Results",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", pady=(0, 8))

        self._result_text = ctk.CTkTextbox(
            main, font=ctk.CTkFont(family="Courier", size=12),
            wrap="word")
        self._result_text.pack(fill="both", expand=True)
        self._result_text.insert("end", "Upload an image and click Analyze...\n")
        self._result_text.configure(state="disabled")

    def _pick_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if path:
            self._image_path = path
            name = Path(path).name
            self._status.configure(text=f"📷 {name}")
            self._analyze_btn.configure(state="normal")

    def _analyze(self):
        if not self._image_path: return
        selected = [t for t, v in self._tasks.items() if v.get()]
        if not selected:
            self._set_text("⚠️  Select at least one task.\n")
            return

        self._analyze_btn.configure(state="disabled", text="Analyzing...")
        self._set_text("⏳ Sending to ApexVision API...\n")
        threading.Thread(target=self._run_analysis, args=(selected,),
                         daemon=True).start()

    def _run_analysis(self, tasks: list):
        try:
            data = base64.b64encode(Path(self._image_path).read_bytes()).decode()
            resp = requests.post(
                f"{self.BASE_URL}/api/v1/vision/analyze",
                headers={"X-ApexVision-Key": self.API_KEY,
                         "Content-Type": "application/json"},
                json={"image": {"format": "base64", "data": data},
                      "tasks": tasks,
                      "options": {"confidence_threshold": self._conf.get()}},
                timeout=90,
            )
            resp.raise_for_status()
            self.after(0, self._show_result, resp.json())
        except Exception as e:
            self.after(0, self._set_text, f"❌ Error: {e}\n")
        finally:
            self.after(0, self._analyze_btn.configure,
                       {"state": "normal", "text": "🚀 Analyze"})

    def _show_result(self, data: dict):
        lines = []
        lines.append(f"✅  request_id : {data['request_id']}")
        lines.append(f"    resolution : {data['image_width']}×{data['image_height']}px")
        lines.append(f"    total time : {data['total_inference_ms']:.1f}ms")
        lines.append(f"    tasks ran  : {', '.join(data['tasks_ran'])}")
        lines.append("")

        if data.get("detection"):
            det = data["detection"]
            lines.append(f"📦  DETECTION — {det['count']} objects")
            for b in det["boxes"][:5]:
                lines.append(f"    • {b['label']:<15} {b['confidence']*100:.1f}%")
            lines.append("")

        if data.get("classification"):
            clf = data["classification"]
            lines.append(f"🏷️   CLASSIFICATION")
            for i, p in enumerate(clf["predictions"][:3], 1):
                lines.append(f"    #{i} {p['label']:<20} {p['confidence']*100:.1f}%")
            lines.append("")

        if data.get("ocr") and data["ocr"].get("text"):
            ocr = data["ocr"]
            lines.append(f"📝  OCR — {len(ocr['text'])} chars · lang: {ocr['language_detected']}")
            lines.append("    ┌─────────────────────────────────────┐")
            for line in ocr["text"].splitlines():
                lines.append(f"    │  {line}")
            lines.append("    └─────────────────────────────────────┘")
            lines.append("")

        if data.get("face"):
            face = data["face"]
            lines.append(f"👤  FACES — {face['count']} detected")
            for i, f in enumerate(face["faces"][:3], 1):
                a = f.get("attributes", {})
                info = []
                if a.get("age"):    info.append(f"age:{a['age']}")
                if a.get("gender"): info.append(f"gender:{a['gender']}")
                if a.get("emotion"):info.append(f"emotion:{a['emotion']}")
                lines.append(f"    #{i} {' · '.join(info)}")
            lines.append("")

        if data.get("embedding"):
            emb = data["embedding"]
            lines.append(f"🔢  EMBEDDING — {emb['dimensions']}d · model: {emb['model_used']}")
            vec_preview = str(emb["embedding"][:6])[:-1] + ", ...]"
            lines.append(f"    {vec_preview}")

        self._set_text("\n".join(lines) + "\n")

    def _set_text(self, text: str):
        self._result_text.configure(state="normal")
        self._result_text.delete("1.0", "end")
        self._result_text.insert("end", text)
        self._result_text.configure(state="disabled")

if __name__ == "__main__":
    app = ApexVisionApp()
    app.mainloop()
```

---

### Python Tkinter

```python
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import requests
import base64
import threading
from pathlib import Path


class ApexVisionTk:
    BASE_URL = "http://localhost:8000"
    API_KEY  = "your-api-key"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ApexVision-Core")
        self.root.geometry("850x600")
        self.root.configure(bg="#1a1a2e")
        self._path = None
        self._build()

    def _build(self):
        # Top frame
        top = tk.Frame(self.root, bg="#16213e", pady=8)
        top.pack(fill="x")
        tk.Label(top, text="ApexVision-Core", bg="#16213e", fg="#6366f1",
                 font=("Helvetica", 16, "bold")).pack(side="left", padx=16)

        # Controls frame
        ctrl = tk.Frame(self.root, bg="#1a1a2e", pady=10)
        ctrl.pack(fill="x", padx=16)

        # File
        tk.Button(ctrl, text="📂 Open Image", command=self._pick,
                  bg="#6366f1", fg="white", relief="flat",
                  padx=12, pady=6, cursor="hand2").pack(side="left")
        self._file_lbl = tk.Label(ctrl, text="No file selected",
                                   bg="#1a1a2e", fg="#64748b")
        self._file_lbl.pack(side="left", padx=10)

        # Tasks
        tk.Label(ctrl, text="Tasks:", bg="#1a1a2e", fg="#94a3b8").pack(
            side="left", padx=(20, 4))
        self._tasks = {}
        for t in ["detect", "classify", "ocr", "face", "embed"]:
            var = tk.BooleanVar(value=(t == "ocr"))
            self._tasks[t] = var
            tk.Checkbutton(ctrl, text=t, variable=var,
                           bg="#1a1a2e", fg="white",
                           selectcolor="#6366f1", activebackground="#1a1a2e",
                           activeforeground="white").pack(side="left", padx=3)

        # Confidence
        tk.Label(ctrl, text="Conf:", bg="#1a1a2e", fg="#94a3b8").pack(
            side="left", padx=(12, 4))
        self._conf = tk.DoubleVar(value=0.5)
        tk.Scale(ctrl, from_=0.1, to=1.0, resolution=0.05,
                 orient="horizontal", variable=self._conf, length=120,
                 bg="#1a1a2e", fg="white", troughcolor="#2d2d4e",
                 highlightthickness=0, showvalue=True).pack(side="left")

        # Analyze button
        self._btn = tk.Button(ctrl, text="🚀 Analyze", command=self._analyze,
                               state="disabled", bg="#10b981", fg="white",
                               relief="flat", padx=12, pady=6, cursor="hand2")
        self._btn.pack(side="right")

        # Results
        results_frame = tk.Frame(self.root, bg="#1a1a2e")
        results_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        tk.Label(results_frame, text="Results",
                 bg="#1a1a2e", fg="#94a3b8",
                 font=("Helvetica", 11, "bold")).pack(anchor="w")

        self._out = scrolledtext.ScrolledText(
            results_frame,
            bg="#0f1117", fg="#e2e8f0",
            font=("Courier", 11),
            insertbackground="white",
            relief="flat", bd=0,
            padx=12, pady=10,
        )
        self._out.pack(fill="both", expand=True, pady=(4, 0))
        self._write("Upload an image and click Analyze...\n")
        self._out.configure(state="disabled")

    def _pick(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if path:
            self._path = path
            self._file_lbl.config(text=Path(path).name, fg="#e2e8f0")
            self._btn.config(state="normal")

    def _analyze(self):
        tasks = [t for t, v in self._tasks.items() if v.get()]
        if not tasks:
            messagebox.showwarning("ApexVision", "Select at least one task.")
            return
        self._btn.config(state="disabled", text="⏳ Analyzing...")
        self._write("Sending to API...\n")
        threading.Thread(target=self._call, args=(tasks,), daemon=True).start()

    def _call(self, tasks):
        try:
            b64  = base64.b64encode(Path(self._path).read_bytes()).decode()
            resp = requests.post(
                f"{self.BASE_URL}/api/v1/vision/analyze",
                headers={"X-ApexVision-Key": self.API_KEY,
                         "Content-Type": "application/json"},
                json={"image": {"format": "base64", "data": b64},
                      "tasks": tasks,
                      "options": {"confidence_threshold": self._conf.get()}},
                timeout=90,
            )
            resp.raise_for_status()
            self.root.after(0, self._show, resp.json())
        except Exception as e:
            self.root.after(0, self._write, f"❌ Error: {e}\n")
        finally:
            self.root.after(0, lambda: self._btn.config(
                state="normal", text="🚀 Analyze"))

    def _show(self, data):
        out = []
        out.append(f"✅ status       : {data['status']}")
        out.append(f"   request_id   : {data['request_id']}")
        out.append(f"   resolution   : {data['image_width']}x{data['image_height']}px")
        out.append(f"   total time   : {data['total_inference_ms']:.1f}ms")
        out.append(f"   tasks ran    : {', '.join(data['tasks_ran'])}")
        out.append("")

        if data.get("detection"):
            d = data["detection"]
            out.append(f"DETECTION — {d['count']} objects detected")
            for b in d["boxes"][:6]:
                bar = "█" * int(b["confidence"] * 20)
                out.append(f"  {b['label']:<14} {bar} {b['confidence']*100:.1f}%")
            out.append("")

        if data.get("classification"):
            c = data["classification"]
            out.append(f"CLASSIFICATION")
            for i, p in enumerate(c["predictions"][:5], 1):
                out.append(f"  #{i} {p['label']:<22} {p['confidence']*100:.1f}%")
            out.append("")

        if data.get("ocr") and data["ocr"].get("text"):
            o = data["ocr"]
            out.append(f"OCR TEXT — {len(o['text'])} chars · lang: {o['language_detected']}")
            out.append("  " + "─" * 44)
            for line in o["text"].splitlines():
                out.append(f"  {line}")
            out.append("  " + "─" * 44)
            out.append("")

        if data.get("face"):
            f = data["face"]
            out.append(f"FACES — {f['count']} detected")
            for i, face in enumerate(f["faces"][:3], 1):
                a = face.get("attributes", {})
                parts = [f"age:{a['age']}" if a.get("age") else "",
                         f"gender:{a['gender']}" if a.get("gender") else "",
                         f"{a['emotion']}" if a.get("emotion") else ""]
                out.append(f"  #{i} {' | '.join(p for p in parts if p)}")
            out.append("")

        if data.get("embedding"):
            e = data["embedding"]
            out.append(f"EMBEDDING — {e['dimensions']}d · {e['model_used']}")

        self._write("\n".join(out) + "\n")

    def _write(self, text):
        self._out.configure(state="normal")
        self._out.delete("1.0", "end")
        self._out.insert("end", text)
        self._out.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    ApexVisionTk(root)
    root.mainloop()
```

---

### JavaScript (Browser)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ApexVision Demo</title>
  <style>
    body { font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 0 20px; }
    pre  { background: #f1f5f9; padding: 16px; border-radius: 8px; overflow: auto; }
    button { padding: 10px 24px; background: #6366f1; color: white;
             border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }
    button:disabled { opacity: 0.5; }
    input[type=file] { display: block; margin: 12px 0; }
    label { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 14px; }
  </style>
</head>
<body>

<h1>ApexVision-Core Demo</h1>

<input type="file" id="fileInput" accept="image/*">

<div id="tasks">
  <label><input type="checkbox" value="detect"   checked> Object Detection</label>
  <label><input type="checkbox" value="classify"> Classification</label>
  <label><input type="checkbox" value="ocr"      checked> OCR / Text</label>
  <label><input type="checkbox" value="face"> Face Analysis</label>
</div>

<br>
<button id="analyzeBtn" onclick="analyze()" disabled>🚀 Analyze Image</button>
<br><br>
<pre id="output">Select an image to get started...</pre>

<script>
const BASE_URL = "http://localhost:8000";
const API_KEY  = "your-api-key";

document.getElementById("fileInput").addEventListener("change", (e) => {
  document.getElementById("analyzeBtn").disabled = !e.target.files[0];
});

async function analyze() {
  const file = document.getElementById("fileInput").files[0];
  if (!file) return;

  const tasks = [...document.querySelectorAll("#tasks input:checked")]
    .map(el => el.value);

  document.getElementById("analyzeBtn").disabled = true;
  document.getElementById("output").textContent   = "Analyzing...";

  try {
    // Convert file to base64
    const b64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload  = () => resolve(reader.result.split(",")[1]);
      reader.onerror = () => reject(new Error("File read error"));
      reader.readAsDataURL(file);
    });

    const response = await fetch(`${BASE_URL}/api/v1/vision/analyze`, {
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

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    // Format output
    let out = `✅ status: ${data.status} · ${data.total_inference_ms.toFixed(1)}ms\n`;
    out += `   tasks: ${data.tasks_ran.join(", ")}\n\n`;

    if (data.detection) {
      out += `📦 DETECTION — ${data.detection.count} objects\n`;
      data.detection.boxes.slice(0, 5).forEach(b => {
        out += `   • ${b.label.padEnd(12)} ${(b.confidence * 100).toFixed(1)}%\n`;
      });
      out += "\n";
    }

    if (data.ocr?.text) {
      out += `📝 OCR TEXT\n`;
      out += `   ${data.ocr.text}\n\n`;
    }

    if (data.classification) {
      out += `🏷️  CLASSIFICATION\n`;
      data.classification.predictions.slice(0, 3).forEach((p, i) => {
        out += `   #${i+1} ${p.label.padEnd(20)} ${(p.confidence * 100).toFixed(1)}%\n`;
      });
    }

    document.getElementById("output").textContent = out;
  } catch (err) {
    document.getElementById("output").textContent = `❌ Error: ${err.message}`;
  } finally {
    document.getElementById("analyzeBtn").disabled = false;
  }
}
</script>
</body>
</html>
```

---

### cURL

```bash
# ── Base64 encode a local file ────────────────────────────────────
IMAGE_B64=$(base64 -w 0 photo.jpg)        # Linux
IMAGE_B64=$(base64 -i photo.jpg)          # macOS

# ── Detect objects ────────────────────────────────────────────────
curl -X POST http://localhost:8000/api/v1/vision/analyze \
  -H "X-ApexVision-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d "{\"image\":{\"format\":\"base64\",\"data\":\"$IMAGE_B64\"},\"tasks\":[\"detect\"]}"

# ── Analyze from URL ──────────────────────────────────────────────
curl -X POST http://localhost:8000/api/v1/vision/analyze \
  -H "X-ApexVision-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"image":{"format":"url","url":"https://example.com/img.jpg"},"tasks":["detect","ocr"]}'

# ── File upload ────────────────────────────────────────────────────
curl -X POST http://localhost:8000/api/v1/vision/analyze/upload \
  -H "X-ApexVision-Key: your-api-key" \
  -F "file=@photo.jpg" \
  -F "tasks=detect,ocr" \
  -F "confidence=0.5"

# ── Health check ───────────────────────────────────────────────────
curl http://localhost:8000/health/

# ── List available tasks ───────────────────────────────────────────
curl -H "X-ApexVision-Key: your-api-key" \
  http://localhost:8000/api/v1/vision/tasks

# ── Batch submit ───────────────────────────────────────────────────
curl -X POST http://localhost:8000/api/v1/batch/submit \
  -H "X-ApexVision-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"requests":[{"image":{"format":"url","url":"https://example.com/img1.jpg"},"tasks":["detect"]},{"image":{"format":"url","url":"https://example.com/img2.jpg"},"tasks":["ocr"]}],"job_name":"my-batch"}'
```

---

## 11. Deployment

### Docker Compose

```bash
docker compose up -d

# Scale workers
docker compose up -d --scale worker=4

# View logs
docker compose logs -f api
docker compose logs -f worker
```

### Production environment variables

```env
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=<64-char-random-string>
MASTER_API_KEY=<secure-api-key>
DEVICE=cuda
DATABASE_URL=postgresql+asyncpg://user:pass@db-host:5432/apexvision
REDIS_URL=redis://redis-host:6379/0
```

### Database migrations

```bash
alembic upgrade head       # apply all migrations
alembic downgrade -1       # rollback last migration
alembic history --verbose  # view migration history
alembic current            # current version in DB

# Generate new migration after model changes
alembic revision --autogenerate -m "add column X to vision_results"
```

### Linting and type checking

```bash
ruff check python/           # lint
ruff check python/ --fix     # auto-fix
ruff format python/          # format
mypy python/                 # type check
```

---

## 12. Roadmap

### v2.1 — In Progress
- [ ] **Image captioning** — BLIP-2 / LLaVA automatic descriptions
- [ ] **Video frame analysis** — process video streams via batch
- [ ] **Custom ONNX models** — upload and serve your own models via API
- [ ] **Enhanced webhooks** — automatic retries with HMAC signature
- [ ] **Per-key rate limiting** — configurable quotas per API client

### v2.2 — Planned
- [ ] **Vector database** — Milvus / Qdrant integration for large-scale similarity search
- [ ] **Multi-GPU support** — automatic load distribution across GPU devices
- [ ] **Model hot-swap** — update models without server restart
- [ ] **A/B model testing** — compare accuracy across model variants
- [ ] **Web dashboard** — built-in UI for results visualization and analytics

### v2.3 — Future
- [ ] **Multi-modal (ImageBind)** — combine image + text + audio inputs
- [ ] **Fine-tuning API** — train custom models with your own labeled data
- [ ] **Edge deployment** — export to ONNX / TensorRT for edge devices
- [ ] **Plugin system** — add custom tasks without modifying the core
- [ ] **GraphQL API** — flexible query alternative to REST

### v2.0 — Completed ✅
- [x] 7 ML engines (YOLO, CLIP, OCR, Face, Embed, Depth, SAM)
- [x] Async parallel pipeline with Redis cache
- [x] Batch processing with Celery + Polars + Delta Lake
- [x] PostgreSQL + Alembic migrations (2 versions)
- [x] Celery Beat schedule (7 automated tasks)
- [x] TypeScript SDK + React / Flutter / Kotlin / Node examples
- [x] 311 tests (unit + integration) · 55.6% coverage
- [x] pyproject.toml (pytest + ruff + mypy + coverage)
- [x] Flower monitoring dashboard
- [x] Docker Compose deployment
- [x] Interactive demo UI at localhost:3000

---

## 13. Contributing

### Workflow

```bash
# 1. Fork and clone
git clone https://github.com/your-username/ApexVision-Core.git
cd ApexVision-Core

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make changes and run tests
python -m pytest tests/ -q

# 4. Lint and format
ruff check python/ --fix
ruff format python/

# 5. Commit with conventional message
git commit -m "feat: add image captioning support"

# 6. Push and open pull request
git push origin feature/your-feature-name
```

### Commit convention

| Prefix | Use case |
|--------|----------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `test:` | Add / modify tests |
| `docs:` | Documentation |
| `refactor:` | Code refactoring |
| `perf:` | Performance improvement |
| `chore:` | Maintenance tasks |
| `ci:` | CI/CD changes |

### Adding a new Vision Task

1. Create engine in `python/core/your_engine.py`
2. Add task to `VisionTask` enum in `python/schemas/vision.py`
3. Register in `python/core/pipeline.py` — `_dispatch_tasks()`
4. Add response field to `VisionResponse` schema
5. Write unit tests in `tests/python/test_your_engine.py`
6. Update this documentation

---

<div align="center">

**ApexVision-Core** — Production computer vision, zero vendor lock-in.

Built with ❤️ · [MIT License](LICENSE) · [Hepein Oficial](https://github.com/Brashkie)

</div>
