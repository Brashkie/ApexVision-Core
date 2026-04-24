# ApexVision-Core — Roadmap

> Living document. Updated as features are completed or priorities shift.  
> Last updated: 2025 · Version 2.0.0

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Completed |
| 🔄 | In progress |
| 📋 | Planned — next quarter |
| 🔮 | Future — not yet scheduled |
| ❌ | Cancelled or deprioritized |

---

## v2.0 — Foundation ✅ RELEASED

The complete production-ready platform with all core capabilities.

### Infrastructure
- [x] FastAPI async server with Uvicorn + Gunicorn
- [x] Celery 5 task queue with Redis broker
- [x] Celery Beat — 7 automated maintenance tasks
- [x] PostgreSQL 16 + SQLAlchemy 2 async ORM
- [x] Alembic migrations — `001_initial` + `002_model_metrics`
- [x] Delta Lake (ACID) + Apache Parquet storage
- [x] Redis cache for pipeline results + rate limiting
- [x] Prometheus metrics at `/metrics`
- [x] Flower monitoring dashboard at `:5555`
- [x] Docker Compose deployment
- [x] TypeScript Gateway (Hono) at `:3000`
- [x] Interactive demo UI

### Vision Engines
- [x] Object detection — YOLOv11 (nano / small / medium / large / extra)
- [x] Classification — ViT-Base · CLIP ViT-B/32 · zero-shot with custom labels
- [x] OCR — EasyOCR (default) · PaddleOCR · Tesseract (tri-backend)
- [x] Face analysis — InsightFace · DeepFace (dual-backend) + landmarks + attributes
- [x] Embeddings — CLIP 512-d L2-normalized · cosine similarity · top-K gallery search
- [x] Depth estimation — DPT-Large · MiDaS v3.1 (dual-backend)
- [x] Segmentation — SAM · SegFormer · Mask2Former + RLE encoding

### API
- [x] `POST /api/v1/vision/analyze` — multi-task in one request
- [x] `POST /api/v1/vision/analyze/upload` — multipart file upload
- [x] Shortcut endpoints per task (`/detect`, `/ocr`, etc.)
- [x] `POST /api/v1/batch/submit` — async batch processing
- [x] Batch status polling + webhook notification
- [x] Models registry API
- [x] Health / liveness / readiness probes
- [x] API key authentication middleware
- [x] API key rate limiting

### SDK & Integrations
- [x] TypeScript SDK (`src/sdk/apexvision.ts`) — full type coverage
- [x] React hooks + components (`examples/react/`)
- [x] Flutter / Dart client (`examples/flutter/`)
- [x] Kotlin / Android client (`examples/kotlin/`)
- [x] Node.js examples with 9 runnable scripts (`examples/nodejs/`)

### Quality
- [x] 311 tests (262 unit + 49 integration) · 55.6% coverage
- [x] `pyproject.toml` — pytest + ruff + mypy + coverage config
- [x] `conftest.py` with ML mocks for fast integration tests
- [x] Windows-compatible dev server (`scripts/dev.js` with `--pool solo`)
- [x] `alembic upgrade head` — one-command DB setup

---

## v2.1 — Enhanced Capabilities 🔄 IN PROGRESS

**Target: Q2 2025**

### New Vision Tasks
- [ ] **Image Captioning** — BLIP-2 and/or LLaVA
  - Auto-generate descriptive text for any image
  - Multilingual output support
  - Integration with existing pipeline as `caption` task
  - Estimated time: 800–3000ms on CPU

- [ ] **Video Frame Analysis** — real-time via WebSocket + batch mode
  - Process video streams frame by frame
  - Track objects across frames (SORT/ByteTrack)
  - Export results to Delta Lake per-video-session

### API Enhancements
- [ ] **Custom ONNX Models** — bring your own model
  - Upload endpoint: `POST /api/v1/models/upload`
  - Auto-detect input/output shapes
  - Run via `custom` task with `custom_model_id` option
  - Model versioning + rollback

- [ ] **Enhanced Webhooks**
  - Automatic retry with exponential backoff (3 attempts)
  - HMAC-SHA256 signature for payload verification
  - Configurable headers per webhook endpoint
  - Delivery logs in PostgreSQL

- [ ] **Per-key Rate Limiting**
  - Configurable RPM (requests per minute) per API key
  - Burst allowance for batch submissions
  - Rate limit headers in responses (`X-RateLimit-Remaining`)
  - Admin endpoint to update limits without restart

### Infrastructure
- [ ] **Prometheus + Grafana stack** — pre-configured dashboards
  - Request latency p50/p95/p99 per task
  - Queue depth and worker utilization
  - Error rate and cache hit ratio
  - Docker Compose profile: `docker compose --profile monitoring up`

---

## v2.2 — Scale & Intelligence 📋 PLANNED

**Target: Q3 2025**

### Vector Search at Scale
- [ ] **Milvus / Qdrant integration** — vector database backend
  - Store CLIP embeddings from every analyzed image
  - `GET /api/v1/search/similar?embedding=...&top_k=20`
  - Batch indexing pipeline via Celery
  - Filter by metadata (date, source, labels)

- [ ] **Image deduplication** — find visually identical or near-duplicate images
  - Configurable similarity threshold
  - Used internally for cache optimization

### Multi-GPU & Performance
- [ ] **Multi-GPU support** — distribute ML workloads
  - CUDA device selection per task type
  - Automatic load balancing across available GPUs
  - GPU utilization metrics in Prometheus

- [ ] **Model hot-swap** — zero-downtime model updates
  - `PATCH /api/v1/models/{task}/variant` — switch model variant
  - Graceful drain: finish in-flight requests before swap
  - A/B testing: route X% of traffic to new model

- [ ] **TensorRT / ONNX optimization** — faster inference
  - Auto-export loaded PyTorch models to TensorRT
  - 2–4x speedup on NVIDIA GPUs
  - Quantization: FP16 and INT8 support

### Dashboard
- [ ] **Built-in Web Dashboard**
  - Real-time analytics: requests/min, avg latency, top tasks
  - Image browser: view recently analyzed images + results
  - Model performance comparison charts
  - API key management UI (create, revoke, set limits)
  - Batch job history with result download

---

## v2.3 — Intelligence & Edge 🔮 FUTURE

**Target: 2026**

### Multi-Modal
- [ ] **ImageBind integration** — joint embeddings for image + text + audio
  - Cross-modal search: find images from text, audio from images
  - Zero-shot audio classification via vision

- [ ] **Document Understanding** — beyond OCR
  - Table extraction with structure preservation
  - Form field detection and extraction
  - Layout analysis (header / footer / columns / figures)
  - Export to structured JSON / CSV / Excel

### Fine-tuning
- [ ] **Fine-tuning API** — train on your own data
  - Dataset upload: `POST /api/v1/datasets` (COCO / YOLO format)
  - Training job: `POST /api/v1/training/start`
  - Training progress via WebSocket stream
  - Automatic evaluation on validation set
  - One-click deploy trained model to production endpoint

### Edge Deployment
- [ ] **ONNX export endpoint** — `GET /api/v1/models/{task}/export?format=onnx`
- [ ] **TFLite export** — for Android / iOS on-device inference
- [ ] **Raspberry Pi / Jetson** deployment guide
- [ ] **Edge SDK** — minimal Python client for constrained devices

### Ecosystem
- [ ] **Plugin system** — community-extensible tasks
  - Define new tasks as Python packages
  - Auto-registration on server startup
  - Plugin marketplace (future)

- [ ] **GraphQL API** — alternative to REST
  - Flexible field selection (only request what you need)
  - Subscriptions for real-time task updates
  - DataLoader for batch efficiency

- [ ] **LangChain / LlamaIndex tool** — use ApexVision as an agent tool
  - `ApexVisionTool` for detecting objects in agent pipelines
  - `ApexVisionOCRTool` for document reading chains

---

## Known Issues & Technical Debt

| Priority | Issue | Status |
|----------|-------|--------|
| High | EasyOCR not auto-installed (requires separate `pip install easyocr`) | Open |
| High | Depth and segment tasks return null when models not downloaded | Open |
| Medium | `--pool solo` required on Windows (prefork IPC bug with billiard) | Workaround |
| Medium | Coverage threshold at 55% — engines not testable without real models | Accepted |
| Low | `DuplicateNodenameWarning` from Celery when restarting worker quickly | Open |
| Low | WebSocket streaming endpoint (`/stream/ws`) not fully implemented | Open |

---

## Architecture Decisions

### Why not serverless (AWS Lambda / Vercel)?
ML models in YOLOv11, DPT, and SAM range from 5MB to 2GB. Cold start on serverless would be 30–120 seconds unacceptable for a vision API. Self-hosted workers with persistent model loading gives sub-100ms response on cached models.

### Why Delta Lake instead of S3/GCS directly?
Delta Lake gives ACID guarantees over Parquet files — concurrent batch writes don't corrupt each other. The `compact_delta` Beat task runs at 02:00 UTC to merge small files from batch jobs into read-optimized chunks.

### Why tri-backend OCR?
- **EasyOCR** — best accuracy on natural images and signage (GPU recommended)
- **PaddleOCR** — best for structured documents, tables, Chinese/Japanese/Korean
- **Tesseract** — lightweight, no GPU, reliable for CI/CD and offline environments

The `auto` backend selection chooses the best available without config changes.

### Why dual-backend for Face?
- **InsightFace** — faster (RetinaFace), better face detection in crowds
- **DeepFace** — more attribute models (emotion, race), easier to extend

---

*Maintained by Moisés Yaurivilca ([@Brashkie](https://github.com/Brashkie)) · Hepein Oficial*
