# ApexVision-Core — SDK & Ejemplos de Integración

## Estructura

```
src/sdk/
└── apexvision.ts          ← SDK TypeScript principal

examples/
├── react/
│   └── apexvision-react.tsx    ← Hooks + Componentes React
├── flutter/
│   └── apexvision_client.dart  ← Cliente Dart completo
├── kotlin/
│   └── ApexVisionClient.kt     ← Cliente Kotlin/Android + ViewModel
└── nodejs/
    └── examples.ts             ← Scripts de ejemplo Node.js
```

---

## Inicio rápido

### 1. Levanta el servidor
```bash
npm run dev        # desde la raíz del proyecto
```

El servidor queda en `http://localhost:8000`.
Docs interactivos: `http://localhost:8000/docs`

---

## TypeScript / Node.js

```typescript
import ApexVisionClient from "./src/sdk/apexvision";

const client = new ApexVisionClient({
  baseUrl: "http://localhost:8000",
  apiKey:  "apexvision-master-dev-key",
});

// Detección desde URL
const result = await client.fromUrl(
  "https://example.com/photo.jpg",
  ["detect"]
);
console.log(result.detection?.boxes);

// Multi-task en un solo request
const full = await client.analyze({
  image: { format: "url", url: "https://example.com/photo.jpg" },
  tasks: ["detect", "ocr", "classify"],
  options: { confidence_threshold: 0.5, top_k: 3 },
});

// Desde archivo local (Node.js)
import { readFileSync } from "fs";
import { bufferToBase64 } from "./src/sdk/apexvision";
const buf = readFileSync("./photo.jpg");
const r = await client.fromBase64(bufferToBase64(buf), ["detect", "ocr"]);
```

Correr los ejemplos:
```bash
npx tsx examples/nodejs/examples.ts
```

---

## React / Next.js

```tsx
import { useVisionAnalyze } from "./examples/react/apexvision-react";

function MyComponent() {
  const { analyze, result, loading, error } = useVisionAnalyze();

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) analyze(file, ["detect", "ocr"]);
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleFile} />
      {loading && <p>Analizando...</p>}
      {result?.detection && <p>{result.detection.count} objetos</p>}
      {result?.ocr?.text && <p>{result.ocr.text}</p>}
    </div>
  );
}
```

Variables de entorno (`.env.local`):
```
NEXT_PUBLIC_APEX_URL=http://localhost:8000
NEXT_PUBLIC_APEX_KEY=apexvision-master-dev-key
```

---

## Flutter

`pubspec.yaml`:
```yaml
dependencies:
  http: ^1.2.0
  web_socket_channel: ^2.4.0
```

```dart
import 'apexvision_client.dart';

final client = ApexVisionClient(
  baseUrl: "http://192.168.1.100:8000",  // IP de tu PC en la red local
  apiKey:  "apexvision-master-dev-key",
);

// Analizar imagen desde galería
final bytes = await file.readAsBytes();
final result = await client.analyzeBytes(
  bytes,
  tasks: ['detect', 'ocr'],
  options: VisionOptions(confidenceThreshold: 0.5),
);

print('${result.detection?.count} objetos');
print(result.ocr?.text);
```

> **IP local**: para conectar desde Android/iOS al servidor en tu PC,
> usa la IP de tu PC en la red WiFi (ej: `192.168.1.x`), no `localhost`.

---

## Kotlin / Android

`build.gradle.kts`:
```kotlin
implementation("com.squareup.retrofit2:retrofit:2.11.0")
implementation("com.squareup.retrofit2:converter-gson:2.11.0")
implementation("com.squareup.okhttp3:okhttp:4.12.0")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")
implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.0")
```

```kotlin
// En tu ViewModel:
val client = ApexVisionClient(
    baseUrl = "http://192.168.1.100:8000",
    apiKey  = "apexvision-master-dev-key",
)

viewModelScope.launch {
    val result = client.detect(bitmap, confidence = 0.5f)
    result.detection?.boxes?.forEach { box ->
        Log.d("APEX", "${box.label}: ${box.confidence}")
    }
}
```

---

## Tabla de tasks disponibles

| Task       | Descripción                                    | Modelo              |
|------------|------------------------------------------------|---------------------|
| `detect`   | Detección de objetos con bboxes                | YOLOv11             |
| `classify` | Clasificación top-K                            | ViT / CLIP          |
| `ocr`      | Extracción de texto                            | EasyOCR / PaddleOCR |
| `face`     | Detección + landmarks + atributos              | InsightFace         |
| `embed`    | Embedding semántico 512-d L2-normalizado       | CLIP                |
| `depth`    | Mapa de profundidad monocular                  | DPT / MiDaS         |
| `segment`  | Segmentación de instancias                     | SAM / SegFormer     |

Puedes combinar múltiples tasks en un solo request:
```typescript
tasks: ["detect", "ocr", "embed"]
```

---

## Endpoints principales

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/api/v1/vision/analyze` | Multi-task análisis |
| `POST` | `/api/v1/vision/analyze/upload` | Upload multipart |
| `POST` | `/api/v1/vision/detect` | Solo detección |
| `POST` | `/api/v1/vision/detect/preview` | Detección + imagen anotada |
| `POST` | `/api/v1/batch/submit` | Submit batch async |
| `GET`  | `/api/v1/batch/{job_id}` | Estado del batch |
| `WS`   | `/api/v1/stream/ws` | WebSocket streaming |
| `GET`  | `/health` | Health check |
| `GET`  | `/docs` | Swagger UI |
