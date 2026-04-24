/**
 * ApexVision-Core SDK — TypeScript
 * @version 2.0.0
 *
 * Compatible con: Node.js, Bun, React, Next.js, Flutter Web, Electron
 */

// ─────────────────────────────────────────────
//  Tipos
// ─────────────────────────────────────────────

export type VisionTask =
  | "detect" | "classify" | "ocr" | "face"
  | "embed"  | "depth"   | "segment" | "caption"
  | "similarity" | "custom";

export type OutputFormat = "json" | "parquet" | "delta";
export type ImageFormat  = "base64" | "url";

export interface ImageInput {
  format: ImageFormat;
  data?: string;   // base64 (con o sin prefijo data URI)
  url?:  string;   // URL pública
}

export interface VisionOptions {
  confidence_threshold?: number;   // 0.0–1.0, default 0.5
  iou_threshold?:        number;   // default 0.45
  max_detections?:       number;   // default 100
  top_k?:                number;   // clasificación, default 5
  ocr_language?:         string;   // default "en"
  ocr_mode?:             "full" | "lines" | "words";
  face_landmarks?:       boolean;
  face_attributes?:      boolean;
  face_embeddings?:      boolean;
  use_cache?:            boolean;
  classes_filter?:       string[];
  clip_labels?:          string[]; // CLIP zero-shot labels
  custom_model_id?:      string;
}

export interface BoundingBox {
  x1: number; y1: number; x2: number; y2: number;
  width: number; height: number;
  confidence: number; label: string; label_id: number;
}

export interface DetectionResult {
  boxes: BoundingBox[];
  count: number;
  model_used: string;
  inference_ms: number;
}

export interface ClassificationResult {
  predictions: Array<{ label: string; confidence: number; label_id: number }>;
  model_used: string;
  inference_ms: number;
}

export interface OCRResult {
  text: string;
  blocks: Array<{
    text: string; confidence: number;
    bbox: { x1: number; y1: number; x2: number; y2: number; width: number; height: number };
  }>;
  language_detected: string;
  inference_ms: number;
}

export interface FaceResult {
  faces: Array<{
    bbox: BoundingBox;
    landmarks?: Record<string, { x: number; y: number }>;
    attributes?: { age?: number; gender?: string; emotion?: string; emotion_scores?: Record<string, number> };
    embedding?: number[];
    embedding_dim?: number;
  }>;
  count: number;
  inference_ms: number;
}

export interface EmbeddingResult {
  embedding: number[];
  dimensions: number;
  model_used: string;
  inference_ms: number;
}

export interface DepthResult {
  depth_map_base64: string;  // JPEG coloreado (JET colormap)
  min_depth: number;         // metros
  max_depth: number;
  inference_ms: number;
}

export interface SegmentationResult {
  masks: Array<{
    label: string; label_id: number; score: number; area: number;
    bbox: BoundingBox;
    mask_rle: { counts: number[]; size: [number, number] };
    backend: string;
  }>;
  count: number;
  inference_ms: number;
}

export interface VisionRequest {
  image: ImageInput;
  tasks?: VisionTask[];
  options?: VisionOptions;
  output_format?: OutputFormat;
  store_result?: boolean;
}

export interface VisionResponse {
  request_id: string;
  status: "success" | "error";
  tasks_ran: VisionTask[];
  detection?:      DetectionResult;
  classification?: ClassificationResult;
  ocr?:            OCRResult;
  face?:           FaceResult;
  embedding?:      EmbeddingResult;
  depth?:          DepthResult;
  segmentation?:   SegmentationResult;
  image_width: number;
  image_height: number;
  total_inference_ms: number;
  stored_at?: string;
}

export interface BatchRequest {
  requests: VisionRequest[];
  job_name?: string;
  webhook_url?: string;
  output_format?: OutputFormat;
}

export interface BatchJobStatus {
  job_id: string;
  status: "pending" | "running" | "done" | "done_with_errors" | "failed";
  total: number;
  completed: number;
  failed: number;
  progress_pct: number;
  result_path?: string;
  elapsed_ms?: number;
  avg_ms_per_image?: number;
  updated_at?: string;
}

export interface ApexVisionConfig {
  baseUrl:     string;
  apiKey:      string;
  timeout?:    number;   // ms, default 60000
  retries?:    number;   // default 3
  retryDelay?: number;   // ms base entre reintentos, default 200
}

// ─────────────────────────────────────────────
//  Error
// ─────────────────────────────────────────────

export class ApexVisionError extends Error {
  constructor(
    message: string,
    public readonly statusCode: number,
    public readonly body?: unknown,
    public readonly requestId?: string,
  ) {
    super(message);
    this.name = "ApexVisionError";
  }
  get isClientError()  { return this.statusCode >= 400 && this.statusCode < 500; }
  get isServerError()  { return this.statusCode >= 500; }
  get isUnauthorized() { return this.statusCode === 401; }
  get isForbidden()    { return this.statusCode === 403; }
  get isRateLimit()    { return this.statusCode === 429; }
}

// ─────────────────────────────────────────────
//  Cliente
// ─────────────────────────────────────────────

export class ApexVisionClient {
  private readonly base:       string;
  private readonly apiKey:     string;
  private readonly timeout:    number;
  private readonly retries:    number;
  private readonly retryDelay: number;
  private readonly prefix:     string;

  constructor(config: ApexVisionConfig) {
    this.base       = config.baseUrl.replace(/\/$/, "");
    this.apiKey     = config.apiKey;
    this.timeout    = config.timeout    ?? 60_000;
    this.retries    = config.retries    ?? 3;
    this.retryDelay = config.retryDelay ?? 200;
    this.prefix     = `${this.base}/api/v1`;
  }

  // ── HTTP con retry ─────────────────────────────────────────────────

  private async req<T>(path: string, init: RequestInit = {}, attempt = 0): Promise<T> {
    const ctrl = new AbortController();
    const tid  = setTimeout(() => ctrl.abort(), this.timeout);
    try {
      const res = await fetch(`${this.prefix}${path}`, {
        ...init,
        signal: ctrl.signal,
        headers: {
          "Content-Type": "application/json",
          "X-ApexVision-Key": this.apiKey,
          ...init.headers,
        },
      });
      clearTimeout(tid);
      if (!res.ok) {
        const body = await res.json().catch(() => ({})) as Record<string, unknown>;
        throw new ApexVisionError(
          (body.message as string) ?? `HTTP ${res.status}`,
          res.status, body, body.request_id as string | undefined,
        );
      }
      return res.json() as Promise<T>;
    } catch (err) {
      clearTimeout(tid);
      if (attempt < this.retries && !(err instanceof ApexVisionError && err.isClientError)) {
        await sleep(Math.min(this.retryDelay * 2 ** attempt, 5_000));
        return this.req<T>(path, init, attempt + 1);
      }
      throw err;
    }
  }

  // ── Vision ─────────────────────────────────────────────────────────

  /** Multi-task: analiza una imagen con varias tasks en un solo request */
  async analyze(request: VisionRequest): Promise<VisionResponse> {
    return this.req<VisionResponse>("/vision/analyze", {
      method: "POST", body: JSON.stringify(request),
    });
  }

  detect  (img: ImageInput, opts?: VisionOptions) { return this.analyze({ image: img, tasks: ["detect"],   options: opts }); }
  classify(img: ImageInput, opts?: VisionOptions) { return this.analyze({ image: img, tasks: ["classify"], options: opts }); }
  ocr     (img: ImageInput, opts?: VisionOptions) { return this.analyze({ image: img, tasks: ["ocr"],      options: opts }); }
  face    (img: ImageInput, opts?: VisionOptions) { return this.analyze({ image: img, tasks: ["face"],     options: opts }); }
  embed   (img: ImageInput, opts?: VisionOptions) { return this.analyze({ image: img, tasks: ["embed"],    options: opts }); }
  depth   (img: ImageInput, opts?: VisionOptions) { return this.analyze({ image: img, tasks: ["depth"],    options: opts }); }
  segment (img: ImageInput, opts?: VisionOptions) { return this.analyze({ image: img, tasks: ["segment"],  options: opts }); }

  // ── Helpers de input ───────────────────────────────────────────────

  fromUrl(url: string, tasks: VisionTask[] = ["detect"], opts?: VisionOptions) {
    return this.analyze({ image: { format: "url", url }, tasks, options: opts });
  }

  fromBase64(data: string, tasks: VisionTask[] = ["detect"], opts?: VisionOptions) {
    const clean = data.includes(",") ? data.split(",")[1] : data;
    return this.analyze({ image: { format: "base64", data: clean }, tasks, options: opts });
  }

  async fromFile(file: File | Blob, tasks: VisionTask[] = ["detect"], opts?: VisionOptions): Promise<VisionResponse> {
    const form = new FormData();
    form.append("file", file);
    form.append("tasks", tasks.join(","));
    if (opts?.confidence_threshold !== undefined)
      form.append("confidence", String(opts.confidence_threshold));

    const ctrl = new AbortController();
    const tid  = setTimeout(() => ctrl.abort(), this.timeout);
    try {
      const res = await fetch(`${this.prefix}/vision/analyze/upload`, {
        method: "POST",
        headers: { "X-ApexVision-Key": this.apiKey },
        body: form,
        signal: ctrl.signal,
      });
      clearTimeout(tid);
      if (!res.ok) {
        const b = await res.json().catch(() => ({})) as Record<string, unknown>;
        throw new ApexVisionError((b.message as string) ?? `HTTP ${res.status}`, res.status, b);
      }
      return res.json() as Promise<VisionResponse>;
    } catch (err) { clearTimeout(tid); throw err; }
  }

  // ── Preview anotado ────────────────────────────────────────────────

  /** Detecta + devuelve imagen anotada con cajas dibujadas (base64 JPEG) */
  async detectWithPreview(image: ImageInput, opts?: VisionOptions): Promise<{
    result: VisionResponse;
    preview_base64: string | null;
    preview_mime: string;
  }> {
    return this.req("/vision/detect/preview", {
      method: "POST",
      body: JSON.stringify({ image, tasks: ["detect"], options: opts }),
    });
  }

  // ── Similarity texto-imagen ────────────────────────────────────────

  /** CLIP zero-shot: qué tan relacionada está la imagen con cada texto */
  async imageTextSimilarity(image: ImageInput, texts: string[]) {
    const r = await this.analyze({
      image, tasks: ["classify"],
      options: { clip_labels: texts, top_k: texts.length },
    });
    return (r.classification?.predictions ?? []).map(p => ({
      text: p.label, similarity: p.confidence,
    }));
  }

  // ── Batch ──────────────────────────────────────────────────────────

  async submitBatch(request: BatchRequest): Promise<{ job_id: string }> {
    return this.req("/batch/submit", { method: "POST", body: JSON.stringify(request) });
  }

  getBatchStatus(jobId: string): Promise<BatchJobStatus> {
    return this.req<BatchJobStatus>(`/batch/${jobId}`);
  }

  async waitForBatch(
    jobId: string,
    opts: { pollIntervalMs?: number; timeoutMs?: number } = {},
  ): Promise<BatchJobStatus> {
    const { pollIntervalMs = 2_000, timeoutMs = 300_000 } = opts;
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const s = await this.getBatchStatus(jobId);
      if (["done", "done_with_errors", "failed"].includes(s.status)) return s;
      await sleep(pollIntervalMs);
    }
    throw new ApexVisionError(`Batch ${jobId} timed out`, 408);
  }

  cancelBatch(jobId: string) {
    return this.req(`/batch/${jobId}`, { method: "DELETE" });
  }

  // ── Models ─────────────────────────────────────────────────────────

  listModels()    { return this.req<Record<string, unknown>>("/models/"); }
  listVariants()  { return this.req<{ variants: Record<string, string> }>("/models/variants"); }
  clearModelCache() { return this.req<{ status: string }>("/models/cache", { method: "DELETE" }); }

  // ── Health ─────────────────────────────────────────────────────────

  async health() {
    const res = await fetch(`${this.base}/health`);
    return res.json() as Promise<{ status: string; service: string; version: string }>;
  }

  async ping(): Promise<boolean> {
    try { const { status } = await this.health(); return status === "ok"; }
    catch { return false; }
  }

  // ── WebSocket ──────────────────────────────────────────────────────

  createStreamSession(tasks: VisionTask[] = ["detect"], opts?: VisionOptions): StreamSession {
    const wsUrl = this.base.replace(/^http/, "ws") + "/api/v1/stream/ws";
    return new StreamSession(wsUrl, tasks, opts);
  }
}

// ─────────────────────────────────────────────
//  WebSocket streaming
// ─────────────────────────────────────────────

export class StreamSession {
  private ws: WebSocket | null = null;

  onResult:     (r: VisionResponse) => void = () => {};
  onError:      (e: Event) => void          = () => {};
  onConnect:    () => void                  = () => {};
  onDisconnect: () => void                  = () => {};

  constructor(
    private readonly wsUrl: string,
    private readonly tasks: VisionTask[],
    private readonly options?: VisionOptions,
  ) {}

  connect(): void {
    this.ws = new WebSocket(this.wsUrl);
    this.ws.onopen    = () => this.onConnect();
    this.ws.onclose   = () => this.onDisconnect();
    this.ws.onerror   = (e) => this.onError(e);
    this.ws.onmessage = (e) => {
      try { this.onResult(JSON.parse(e.data as string)); } catch {}
    };
  }

  sendFrame(base64Image: string): void {
    if (this.ws?.readyState !== WebSocket.OPEN)
      throw new Error("Not connected. Call connect() first.");
    this.ws.send(JSON.stringify({
      image: base64Image,
      tasks: this.tasks,
      confidence: this.options?.confidence_threshold ?? 0.5,
    }));
  }

  close(): void { this.ws?.close(); this.ws = null; }
  get isConnected() { return this.ws?.readyState === WebSocket.OPEN; }
}

// ─────────────────────────────────────────────
//  Utilidades
// ─────────────────────────────────────────────

export async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload  = () => resolve((r.result as string).split(",")[1]);
    r.onerror = () => reject(new Error("Failed to read file"));
    r.readAsDataURL(file);
  });
}

export function bufferToBase64(buf: Buffer): string { return buf.toString("base64"); }

export function cosineSimilarity(a: number[], b: number[]): number {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}

export function topKSimilar(
  query: number[],
  gallery: number[][],
  k = 5,
): Array<{ index: number; similarity: number }> {
  return gallery
    .map((emb, i) => ({ index: i, similarity: cosineSimilarity(query, emb) }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, k);
}

function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }

export default ApexVisionClient;
