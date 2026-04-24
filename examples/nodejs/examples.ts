/**
 * ApexVision-Core — Node.js / Bun Examples
 * Scripts listos para correr desde terminal
 *
 * Prerrequisito: servidor corriendo en localhost:8000
 *   npm run dev
 *
 * Correr:
 *   npx tsx examples/nodejs/examples.ts
 */

import { readFileSync, writeFileSync } from "fs";
import { join } from "path";
import ApexVisionClient, {
  cosineSimilarity,
  topKSimilar,
  bufferToBase64,
  ApexVisionError,
} from "../../src/sdk/apexvision";

// ─────────────────────────────────────────────
//  Setup
// ─────────────────────────────────────────────

const client = new ApexVisionClient({
  baseUrl: "http://localhost:8000",
  apiKey:  "apexvision-master-dev-key",
  timeout: 60_000,
  retries: 3,
});

const log = {
  ok:    (msg: string) => console.log(`  ✅ ${msg}`),
  info:  (msg: string) => console.log(`  ℹ  ${msg}`),
  err:   (msg: string) => console.log(`  ❌ ${msg}`),
  title: (msg: string) => console.log(`\n${"─".repeat(50)}\n  ${msg}\n${"─".repeat(50)}`),
};

// ─────────────────────────────────────────────
//  Ejemplo 1: ping + health
// ─────────────────────────────────────────────

async function example_health() {
  log.title("1. Health check");

  const alive = await client.ping();
  log.ok(`Server alive: ${alive}`);

  if (!alive) {
    log.err("Servidor no disponible. Corre: npm run dev");
    process.exit(1);
  }

  const health = await client.health();
  log.info(`Servicio: ${health.service} v${health.version}`);
}

// ─────────────────────────────────────────────
//  Ejemplo 2: detección desde URL
// ─────────────────────────────────────────────

async function example_detect_from_url() {
  log.title("2. Detección de objetos desde URL");

  const result = await client.fromUrl(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
    ["detect"],
    { confidence_threshold: 0.4 },
  );

  log.ok(`${result.detection?.count ?? 0} objetos detectados en ${result.total_inference_ms.toFixed(1)}ms`);
  result.detection?.boxes.slice(0, 5).forEach(box => {
    log.info(`  ${box.label.padEnd(15)} conf=${(box.confidence * 100).toFixed(1)}%  bbox=[${box.x1.toFixed(0)},${box.y1.toFixed(0)},${box.x2.toFixed(0)},${box.y2.toFixed(0)}]`);
  });
}

// ─────────────────────────────────────────────
//  Ejemplo 3: multi-task en un solo request
// ─────────────────────────────────────────────

async function example_multitask() {
  log.title("3. Multi-task: detect + classify + ocr en un request");

  const imageUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg";

  const result = await client.analyze({
    image:   { format: "url", url: imageUrl },
    tasks:   ["detect", "classify", "embed"],
    options: { confidence_threshold: 0.3, top_k: 3 },
  });

  log.ok(`Tiempo total: ${result.total_inference_ms.toFixed(1)}ms`);
  log.info(`Imagen: ${result.image_width}×${result.image_height}px`);

  if (result.detection) {
    log.info(`Detección: ${result.detection.count} objetos (${result.detection.model_used})`);
  }
  if (result.classification) {
    const top = result.classification.predictions[0];
    log.info(`Clasificación: ${top?.label} (${((top?.confidence ?? 0) * 100).toFixed(1)}%)`);
  }
  if (result.embedding) {
    log.info(`Embedding: ${result.embedding.dimensions}d (${result.embedding.model_used})`);
    log.info(`  Norma L2: ${Math.sqrt(result.embedding.embedding.reduce((s, v) => s + v * v, 0)).toFixed(4)} (debe ser ~1.0)`);
  }
}

// ─────────────────────────────────────────────
//  Ejemplo 4: OCR desde archivo local
// ─────────────────────────────────────────────

async function example_ocr_local() {
  log.title("4. OCR desde archivo local");

  // Crea una imagen de prueba con texto (si no tienes una a mano)
  const samplePath = join(process.cwd(), "examples", "nodejs", "sample.jpg");

  let imageData: string;
  try {
    const buf = readFileSync(samplePath);
    imageData = bufferToBase64(buf);
    log.info(`Archivo cargado: ${buf.length / 1024} KB`);
  } catch {
    // Si no existe el archivo, usa una URL de demo
    log.info("No hay sample.jpg local, usando URL de demo...");
    const result = await client.fromUrl(
      "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/JPEG_example_JPG_RIP_100.jpg/320px-JPEG_example_JPG_RIP_100.jpg",
      ["ocr"],
    );
    log.ok(`OCR: "${result.ocr?.text ?? "(sin texto)"}" | idioma=${result.ocr?.language_detected}`);
    return;
  }

  const result = await client.fromBase64(imageData, ["ocr"]);
  log.ok(`Texto extraído: "${result.ocr?.text.slice(0, 100)}..."`);
  log.info(`Idioma detectado: ${result.ocr?.language_detected}`);
  log.info(`Bloques: ${result.ocr?.blocks.length}`);
}

// ─────────────────────────────────────────────
//  Ejemplo 5: Similarity search con embeddings
// ─────────────────────────────────────────────

async function example_similarity_search() {
  log.title("5. Image-to-text similarity (CLIP zero-shot)");

  const imageUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg";

  const labels = ["a cat", "a dog", "a car", "a person", "a tree", "a building"];

  const similarities = await client.imageTextSimilarity(
    { format: "url", url: imageUrl },
    labels,
  );

  log.ok("Rankings:");
  similarities.forEach((s, i) => {
    const bar = "█".repeat(Math.round(s.similarity * 30));
    log.info(`  ${(i+1)}. ${s.text.padEnd(15)} ${(s.similarity * 100).toFixed(1)}% ${bar}`);
  });
}

// ─────────────────────────────────────────────
//  Ejemplo 6: Gallery search con cosine similarity
// ─────────────────────────────────────────────

async function example_gallery_search() {
  log.title("6. Gallery search: encontrar imágenes similares");

  const IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/160px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/160px-Camponotus_flavomarginatus_ant.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/160px-Sunflower_from_Silesia2.jpg",
  ];

  log.info("Generando embeddings para galería...");

  // Genera embeddings para todas las imágenes
  const gallery: number[][] = [];
  for (const url of IMAGE_URLS) {
    try {
      const r = await client.fromUrl(url, ["embed"]);
      if (r.embedding) gallery.push(r.embedding.embedding);
    } catch {
      log.err(`  No se pudo procesar: ${url}`);
    }
  }

  log.info(`Galería: ${gallery.length} embeddings de ${gallery[0]?.length}d`);

  // Busca la imagen más similar a la primera
  if (gallery.length >= 2) {
    const query = gallery[0];
    const results = topKSimilar(query, gallery.slice(1), 2);

    log.ok("Imágenes más similares a gallery[0]:");
    results.forEach(r => {
      log.info(`  index=${r.index + 1}  similarity=${(r.similarity * 100).toFixed(1)}%`);
    });

    log.info(`Auto-similarity (debe ser ~100%): ${(cosineSimilarity(query, query) * 100).toFixed(2)}%`);
  }
}

// ─────────────────────────────────────────────
//  Ejemplo 7: Batch processing
// ─────────────────────────────────────────────

async function example_batch() {
  log.title("7. Batch processing (async)");

  const IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/160px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/160px-Camponotus_flavomarginatus_ant.jpg",
  ];

  const requests = IMAGE_URLS.map(url => ({
    image:   { format: "url" as const, url },
    tasks:   ["detect"] as ["detect"],
    options: { confidence_threshold: 0.4 },
  }));

  log.info(`Enviando batch de ${requests.length} imágenes...`);

  const { job_id } = await client.submitBatch({
    requests,
    job_name: `batch-example-${Date.now()}`,
  });

  log.ok(`Job ID: ${job_id}`);
  log.info("Esperando resultados...");

  const status = await client.waitForBatch(job_id, {
    pollIntervalMs: 1_000,
    timeoutMs: 60_000,
  });

  log.ok(`Batch terminado: ${status.completed}/${status.total} completados, ${status.failed} fallidos`);
  if (status.result_path) log.info(`Resultados en: ${status.result_path}`);
}

// ─────────────────────────────────────────────
//  Ejemplo 8: Error handling
// ─────────────────────────────────────────────

async function example_error_handling() {
  log.title("8. Error handling");

  // API key inválida
  const badClient = new ApexVisionClient({
    baseUrl: "http://localhost:8000",
    apiKey:  "invalid-key",
    retries: 0,
  });

  try {
    await badClient.fromUrl("https://example.com/img.jpg", ["detect"]);
  } catch (e) {
    if (e instanceof ApexVisionError) {
      log.ok(`Error capturado correctamente: ${e.statusCode} — ${e.message}`);
      log.info(`isClientError=${e.isClientError} | isForbidden=${e.isForbidden}`);
    }
  }

  // Imagen inválida
  try {
    await client.fromBase64("not_valid_base64_!@#$", ["detect"]);
  } catch (e) {
    log.ok(`Error de validación capturado: ${e instanceof Error ? e.message.slice(0, 80) : e}`);
  }
}

// ─────────────────────────────────────────────
//  Ejemplo 9: WebSocket streaming (Node.js)
// ─────────────────────────────────────────────

async function example_websocket() {
  log.title("9. WebSocket streaming (demo 3s)");

  // En Node.js necesitas el paquete 'ws':
  //   npm install ws
  //   y asignar globalThis.WebSocket = require('ws')
  // En browser y Bun funciona nativo.

  try {
    // @ts-ignore — solo disponible si tienes 'ws' instalado
    const { WebSocket } = await import("ws");
    (globalThis as Record<string, unknown>).WebSocket = WebSocket;
  } catch {
    log.info("Paquete 'ws' no instalado. Skipping WebSocket demo.");
    log.info("  Para habilitarlo: npm install ws");
    return;
  }

  const session = client.createStreamSession(["detect"], { confidence_threshold: 0.5 });

  let frameCount = 0;

  session.onConnect    = () => log.ok("WebSocket conectado");
  session.onDisconnect = () => log.info("WebSocket desconectado");
  session.onResult     = (r) => {
    frameCount++;
    if (r.detection) {
      log.info(`Frame ${frameCount}: ${r.detection.count} objetos en ${r.total_inference_ms.toFixed(1)}ms`);
    }
  };

  session.connect();

  // Simula envío de frames durante 3s
  const MOCK_FRAME = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AJQAB/9k="; // 1x1 JPEG

  const interval = setInterval(() => {
    if (session.isConnected) {
      session.sendFrame(MOCK_FRAME);
    }
  }, 500);

  await new Promise(r => setTimeout(r, 3_000));
  clearInterval(interval);
  session.close();
  log.ok(`Demo finalizado. ${frameCount} frames procesados.`);
}

// ─────────────────────────────────────────────
//  Runner
// ─────────────────────────────────────────────

async function main() {
  console.log("\n🚀 ApexVision-Core SDK — Node.js Examples\n");

  try {
    await example_health();
    await example_detect_from_url();
    await example_multitask();
    await example_ocr_local();
    await example_similarity_search();
    await example_gallery_search();
    await example_batch();
    await example_error_handling();
    await example_websocket();

    console.log("\n✅ Todos los ejemplos completados.\n");
  } catch (err) {
    console.error("\n❌ Error fatal:", err);
    process.exit(1);
  }
}

main();
