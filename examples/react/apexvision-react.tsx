/**
 * ApexVision-Core — React Integration Example
 * Hooks y componentes listos para usar en cualquier proyecto React/Next.js
 *
 * Instalación:
 *   Copia src/sdk/apexvision.ts a tu proyecto
 *   npm install  (solo deps que ya tienes: react, typescript)
 */

import { useState, useCallback, useRef, useEffect } from "react";
import ApexVisionClient, {
  VisionResponse,
  VisionTask,
  VisionOptions,
  BatchJobStatus,
  ApexVisionError,
  StreamSession,
  fileToBase64,
} from "../sdk/apexvision";

// ─────────────────────────────────────────────
//  Configuración
// ─────────────────────────────────────────────

const client = new ApexVisionClient({
  baseUrl: process.env.NEXT_PUBLIC_APEX_URL ?? "http://localhost:8000",
  apiKey:  process.env.NEXT_PUBLIC_APEX_KEY ?? "apexvision-master-dev-key",
});

// ─────────────────────────────────────────────
//  Hook: useVisionAnalyze
//  Analiza una imagen con cualquier combinación de tasks
// ─────────────────────────────────────────────

interface UseVisionAnalyzeReturn {
  result:   VisionResponse | null;
  loading:  boolean;
  error:    string | null;
  analyze:  (file: File, tasks?: VisionTask[], opts?: VisionOptions) => Promise<void>;
  analyzeUrl: (url: string, tasks?: VisionTask[], opts?: VisionOptions) => Promise<void>;
  reset:    () => void;
}

export function useVisionAnalyze(): UseVisionAnalyzeReturn {
  const [result,  setResult]  = useState<VisionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const analyze = useCallback(async (
    file: File,
    tasks: VisionTask[] = ["detect"],
    opts?: VisionOptions,
  ) => {
    setLoading(true);
    setError(null);
    try {
      const res = await client.fromFile(file, tasks, opts);
      setResult(res);
    } catch (e) {
      setError(e instanceof ApexVisionError ? e.message : "Error inesperado");
    } finally {
      setLoading(false);
    }
  }, []);

  const analyzeUrl = useCallback(async (
    url: string,
    tasks: VisionTask[] = ["detect"],
    opts?: VisionOptions,
  ) => {
    setLoading(true);
    setError(null);
    try {
      const res = await client.fromUrl(url, tasks, opts);
      setResult(res);
    } catch (e) {
      setError(e instanceof ApexVisionError ? e.message : "Error inesperado");
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, analyze, analyzeUrl, reset };
}

// ─────────────────────────────────────────────
//  Hook: useBatchVision
//  Submit + polling automático
// ─────────────────────────────────────────────

interface UseBatchReturn {
  jobStatus:  BatchJobStatus | null;
  submitting: boolean;
  error:      string | null;
  submitBatch: (files: File[], tasks?: VisionTask[]) => Promise<void>;
}

export function useBatchVision(): UseBatchReturn {
  const [jobStatus,  setJobStatus]  = useState<BatchJobStatus | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error,      setError]      = useState<string | null>(null);

  const submitBatch = useCallback(async (files: File[], tasks: VisionTask[] = ["detect"]) => {
    setSubmitting(true);
    setError(null);
    try {
      // Convertir files a base64
      const requests = await Promise.all(
        files.map(async (f) => ({
          image: { format: "base64" as const, data: await fileToBase64(f) },
          tasks,
        }))
      );

      const { job_id } = await client.submitBatch({ requests, job_name: `batch-${Date.now()}` });

      // Polling automático cada 2s
      const status = await client.waitForBatch(job_id, {
        pollIntervalMs: 2_000,
        timeoutMs: 120_000,
        // Puedes mostrar progreso intermedio si quieres:
        // ver implementación custom abajo
      });
      setJobStatus(status);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Batch error");
    } finally {
      setSubmitting(false);
    }
  }, []);

  return { jobStatus, submitting, error, submitBatch };
}

// ─────────────────────────────────────────────
//  Hook: useStreamVision
//  WebSocket real-time para cámara
// ─────────────────────────────────────────────

interface UseStreamReturn {
  result:      VisionResponse | null;
  connected:   boolean;
  connect:     () => void;
  disconnect:  () => void;
  sendFrame:   (base64: string) => void;
}

export function useStreamVision(tasks: VisionTask[] = ["detect"]): UseStreamReturn {
  const [result,    setResult]    = useState<VisionResponse | null>(null);
  const [connected, setConnected] = useState(false);
  const sessionRef = useRef<StreamSession | null>(null);

  const connect = useCallback(() => {
    const session = client.createStreamSession(tasks, { confidence_threshold: 0.5 });
    session.onConnect    = () => setConnected(true);
    session.onDisconnect = () => setConnected(false);
    session.onResult     = (r) => setResult(r);
    session.connect();
    sessionRef.current = session;
  }, [tasks]);

  const disconnect = useCallback(() => {
    sessionRef.current?.close();
    sessionRef.current = null;
    setConnected(false);
  }, []);

  const sendFrame = useCallback((base64: string) => {
    sessionRef.current?.sendFrame(base64);
  }, []);

  // Cleanup al desmontar
  useEffect(() => () => { sessionRef.current?.close(); }, []);

  return { result, connected, connect, disconnect, sendFrame };
}

// ─────────────────────────────────────────────
//  Componente: DetectionViewer
//  Muestra imagen con bounding boxes dibujados
// ─────────────────────────────────────────────

interface DetectionViewerProps {
  imageUrl:    string;
  detections:  VisionResponse["detection"];
}

export function DetectionViewer({ imageUrl, detections }: DetectionViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !detections) return;
    const canvas  = canvasRef.current;
    const ctx     = canvas.getContext("2d")!;
    const img     = new Image();
    img.src       = imageUrl;
    img.crossOrigin = "anonymous";

    img.onload = () => {
      canvas.width  = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Colores por label (hash simple)
      const colorFor = (label: string) => {
        let h = 0;
        for (const c of label) h = (h * 31 + c.charCodeAt(0)) & 0xFFFFFF;
        return `#${h.toString(16).padStart(6, "0")}`;
      };

      detections.boxes.forEach(box => {
        const color = colorFor(box.label);

        // Caja
        ctx.strokeStyle = color;
        ctx.lineWidth   = 2;
        ctx.strokeRect(box.x1, box.y1, box.width, box.height);

        // Label con fondo
        const label = `${box.label} ${(box.confidence * 100).toFixed(0)}%`;
        ctx.font      = "14px Inter, sans-serif";
        const tw      = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(box.x1, box.y1 - 20, tw + 8, 20);
        ctx.fillStyle = "#000";
        ctx.fillText(label, box.x1 + 4, box.y1 - 5);
      });
    };
  }, [imageUrl, detections]);

  return (
    <canvas
      ref={canvasRef}
      style={{ maxWidth: "100%", borderRadius: 8, border: "1px solid #e2e8f0" }}
    />
  );
}

// ─────────────────────────────────────────────
//  Componente: ImageAnalyzer
//  Upload → análisis completo → muestra resultados
// ─────────────────────────────────────────────

export function ImageAnalyzer() {
  const { result, loading, error, analyze } = useVisionAnalyze();
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedTasks, setSelectedTasks] = useState<VisionTask[]>(["detect"]);

  const AVAILABLE_TASKS: VisionTask[] = ["detect", "classify", "ocr", "face", "embed"];

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setPreviewUrl(URL.createObjectURL(file));
    await analyze(file, selectedTasks, { confidence_threshold: 0.5 });
  };

  const toggleTask = (task: VisionTask) => {
    setSelectedTasks(prev =>
      prev.includes(task) ? prev.filter(t => t !== task) : [...prev, task]
    );
  };

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", fontFamily: "Inter, sans-serif" }}>
      <h2>ApexVision Analyzer</h2>

      {/* Task selector */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
        {AVAILABLE_TASKS.map(task => (
          <button
            key={task}
            onClick={() => toggleTask(task)}
            style={{
              padding: "4px 12px",
              borderRadius: 20,
              border: "1px solid",
              borderColor: selectedTasks.includes(task) ? "#6366f1" : "#cbd5e1",
              background:  selectedTasks.includes(task) ? "#6366f1" : "transparent",
              color:       selectedTasks.includes(task) ? "#fff" : "#64748b",
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            {task}
          </button>
        ))}
      </div>

      {/* File input */}
      <input
        type="file"
        accept="image/*"
        onChange={handleFile}
        style={{ marginBottom: 16 }}
      />

      {loading && <p style={{ color: "#6366f1" }}>Analizando...</p>}
      {error   && <p style={{ color: "#ef4444" }}>{error}</p>}

      {/* Preview con boxes */}
      {previewUrl && result?.detection && (
        <DetectionViewer imageUrl={previewUrl} detections={result.detection} />
      )}

      {/* Resultados */}
      {result && (
        <div style={{ marginTop: 16, display: "grid", gap: 12 }}>

          {/* Detection */}
          {result.detection && (
            <ResultCard title={`Detección — ${result.detection.count} objetos`} color="#6366f1">
              {result.detection.boxes.slice(0, 5).map((b, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
                  <span>{b.label}</span>
                  <span style={{ color: "#64748b" }}>{(b.confidence * 100).toFixed(1)}%</span>
                </div>
              ))}
            </ResultCard>
          )}

          {/* Classification */}
          {result.classification && (
            <ResultCard title="Clasificación" color="#0ea5e9">
              {result.classification.predictions.slice(0, 3).map((p, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
                  <span>{p.label}</span>
                  <ConfidenceBar value={p.confidence} />
                </div>
              ))}
            </ResultCard>
          )}

          {/* OCR */}
          {result.ocr?.text && (
            <ResultCard title={`OCR — ${result.ocr.language_detected}`} color="#10b981">
              <p style={{ fontSize: 13, margin: 0, whiteSpace: "pre-wrap" }}>
                {result.ocr.text}
              </p>
            </ResultCard>
          )}

          {/* Face */}
          {result.face && (
            <ResultCard title={`Rostros — ${result.face.count} detectados`} color="#f59e0b">
              {result.face.faces.slice(0, 3).map((f, i) => (
                <div key={i} style={{ fontSize: 13 }}>
                  {f.attributes?.age && <span>Edad: {f.attributes.age}  </span>}
                  {f.attributes?.gender && <span>Género: {f.attributes.gender}  </span>}
                  {f.attributes?.emotion && <span>Emoción: {f.attributes.emotion}</span>}
                </div>
              ))}
            </ResultCard>
          )}

          {/* Timing */}
          <p style={{ fontSize: 12, color: "#94a3b8", textAlign: "right" }}>
            {result.image_width}×{result.image_height}px · {result.total_inference_ms.toFixed(1)}ms
          </p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────
//  Sub-componentes UI
// ─────────────────────────────────────────────

function ResultCard({ title, color, children }: {
  title: string; color: string; children: React.ReactNode;
}) {
  return (
    <div style={{
      border: `1px solid ${color}33`,
      borderLeft: `4px solid ${color}`,
      borderRadius: 8,
      padding: "10px 14px",
      background: `${color}08`,
    }}>
      <p style={{ fontWeight: 600, fontSize: 13, marginBottom: 8, color }}>{title}</p>
      {children}
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{
        width: 80, height: 6, background: "#e2e8f0", borderRadius: 3, overflow: "hidden",
      }}>
        <div style={{
          width: `${value * 100}%`, height: "100%",
          background: value > 0.7 ? "#10b981" : value > 0.4 ? "#f59e0b" : "#ef4444",
        }} />
      </div>
      <span style={{ fontSize: 12, color: "#64748b" }}>{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

// ─────────────────────────────────────────────
//  Uso rápido (copy-paste)
// ─────────────────────────────────────────────

/*
// En cualquier componente React:

import { useVisionAnalyze } from "./apexvision-react";

function MyComponent() {
  const { analyze, result, loading } = useVisionAnalyze();

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    await analyze(file, ["detect", "ocr"]);
  };

  return (
    <div>
      <input type="file" onChange={handleUpload} />
      {loading && <p>Procesando...</p>}
      {result?.detection && (
        <p>{result.detection.count} objetos detectados</p>
      )}
      {result?.ocr?.text && (
        <p>Texto: {result.ocr.text}</p>
      )}
    </div>
  );
}
*/
