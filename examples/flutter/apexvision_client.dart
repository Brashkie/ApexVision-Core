// ─────────────────────────────────────────────────────────────────────────────
//  ApexVision-Core — Flutter / Dart Client
//  Compatible con: Flutter Mobile (Android/iOS), Flutter Web, Flutter Desktop
//
//  Dependencias (pubspec.yaml):
//    dependencies:
//      http: ^1.2.0
//      web_socket_channel: ^2.4.0
//
//  Uso:
//    final client = ApexVisionClient(
//      baseUrl: "http://192.168.1.x:8000",  // IP de tu servidor
//      apiKey:  "apexvision-master-dev-key",
//    );
// ─────────────────────────────────────────────────────────────────────────────

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';

// ─────────────────────────────────────────────
//  Tipos
// ─────────────────────────────────────────────

class BoundingBox {
  final double x1, y1, x2, y2, width, height, confidence;
  final String label;
  final int labelId;

  const BoundingBox({
    required this.x1, required this.y1,
    required this.x2, required this.y2,
    required this.width, required this.height,
    required this.confidence,
    required this.label, required this.labelId,
  });

  factory BoundingBox.fromJson(Map<String, dynamic> j) => BoundingBox(
    x1: (j['x1'] as num).toDouble(), y1: (j['y1'] as num).toDouble(),
    x2: (j['x2'] as num).toDouble(), y2: (j['y2'] as num).toDouble(),
    width:  (j['width']  as num).toDouble(),
    height: (j['height'] as num).toDouble(),
    confidence: (j['confidence'] as num).toDouble(),
    label:   j['label']    as String,
    labelId: j['label_id'] as int,
  );
}

class DetectionResult {
  final List<BoundingBox> boxes;
  final int count;
  final String modelUsed;
  final double inferenceMs;

  const DetectionResult({
    required this.boxes, required this.count,
    required this.modelUsed, required this.inferenceMs,
  });

  factory DetectionResult.fromJson(Map<String, dynamic> j) => DetectionResult(
    boxes:       (j['boxes'] as List).map((b) => BoundingBox.fromJson(b as Map<String, dynamic>)).toList(),
    count:       j['count'] as int,
    modelUsed:   j['model_used'] as String,
    inferenceMs: (j['inference_ms'] as num).toDouble(),
  );
}

class OCRResult {
  final String text;
  final String languageDetected;
  final double inferenceMs;
  final List<Map<String, dynamic>> blocks;

  const OCRResult({
    required this.text, required this.languageDetected,
    required this.inferenceMs, required this.blocks,
  });

  factory OCRResult.fromJson(Map<String, dynamic> j) => OCRResult(
    text:              j['text'] as String,
    languageDetected:  j['language_detected'] as String,
    inferenceMs:       (j['inference_ms'] as num).toDouble(),
    blocks:            List<Map<String, dynamic>>.from(j['blocks'] as List),
  );
}

class FaceResult {
  final List<Map<String, dynamic>> faces;
  final int count;
  final double inferenceMs;

  const FaceResult({required this.faces, required this.count, required this.inferenceMs});

  factory FaceResult.fromJson(Map<String, dynamic> j) => FaceResult(
    faces:       List<Map<String, dynamic>>.from(j['faces'] as List),
    count:       j['count'] as int,
    inferenceMs: (j['inference_ms'] as num).toDouble(),
  );
}

class ClassificationResult {
  final List<Map<String, dynamic>> predictions;
  final String modelUsed;
  final double inferenceMs;

  const ClassificationResult({required this.predictions, required this.modelUsed, required this.inferenceMs});

  factory ClassificationResult.fromJson(Map<String, dynamic> j) => ClassificationResult(
    predictions: List<Map<String, dynamic>>.from(j['predictions'] as List),
    modelUsed:   j['model_used'] as String,
    inferenceMs: (j['inference_ms'] as num).toDouble(),
  );
}

class EmbeddingResult {
  final List<double> embedding;
  final int dimensions;
  final String modelUsed;
  final double inferenceMs;

  const EmbeddingResult({
    required this.embedding, required this.dimensions,
    required this.modelUsed, required this.inferenceMs,
  });

  factory EmbeddingResult.fromJson(Map<String, dynamic> j) => EmbeddingResult(
    embedding:   List<double>.from((j['embedding'] as List).map((e) => (e as num).toDouble())),
    dimensions:  j['dimensions'] as int,
    modelUsed:   j['model_used'] as String,
    inferenceMs: (j['inference_ms'] as num).toDouble(),
  );
}

class VisionResponse {
  final String requestId;
  final String status;
  final List<String> tasksRan;
  final DetectionResult?     detection;
  final ClassificationResult? classification;
  final OCRResult?            ocr;
  final FaceResult?           face;
  final EmbeddingResult?      embedding;
  final int imageWidth, imageHeight;
  final double totalInferenceMs;

  const VisionResponse({
    required this.requestId, required this.status,
    required this.tasksRan,
    this.detection, this.classification, this.ocr,
    this.face, this.embedding,
    required this.imageWidth, required this.imageHeight,
    required this.totalInferenceMs,
  });

  factory VisionResponse.fromJson(Map<String, dynamic> j) => VisionResponse(
    requestId:        j['request_id'] as String,
    status:           j['status'] as String,
    tasksRan:         List<String>.from(j['tasks_ran'] as List),
    detection:        j['detection'] != null
        ? DetectionResult.fromJson(j['detection'] as Map<String, dynamic>) : null,
    classification:   j['classification'] != null
        ? ClassificationResult.fromJson(j['classification'] as Map<String, dynamic>) : null,
    ocr:              j['ocr'] != null
        ? OCRResult.fromJson(j['ocr'] as Map<String, dynamic>) : null,
    face:             j['face'] != null
        ? FaceResult.fromJson(j['face'] as Map<String, dynamic>) : null,
    embedding:        j['embedding'] != null
        ? EmbeddingResult.fromJson(j['embedding'] as Map<String, dynamic>) : null,
    imageWidth:       j['image_width'] as int,
    imageHeight:      j['image_height'] as int,
    totalInferenceMs: (j['total_inference_ms'] as num).toDouble(),
  );
}

class BatchJobStatus {
  final String jobId, status;
  final int total, completed, failed;
  final double progressPct;
  final String? resultPath;

  const BatchJobStatus({
    required this.jobId, required this.status,
    required this.total, required this.completed, required this.failed,
    required this.progressPct, this.resultPath,
  });

  factory BatchJobStatus.fromJson(Map<String, dynamic> j) => BatchJobStatus(
    jobId:       j['job_id'] as String,
    status:      j['status'] as String,
    total:       j['total'] as int,
    completed:   j['completed'] as int,
    failed:      j['failed'] as int,
    progressPct: (j['progress_pct'] as num).toDouble(),
    resultPath:  j['result_path'] as String?,
  );
}

// ─────────────────────────────────────────────
//  Cliente principal
// ─────────────────────────────────────────────

class ApexVisionClient {
  final String baseUrl;
  final String apiKey;
  final Duration timeout;
  final http.Client _http;

  ApexVisionClient({
    required this.baseUrl,
    required this.apiKey,
    this.timeout = const Duration(seconds: 60),
  }) : _http = http.Client();

  Map<String, String> get _headers => {
    'Content-Type':       'application/json',
    'X-ApexVision-Key':   apiKey,
  };

  String get _prefix => '$baseUrl/api/v1';

  // ── HTTP helper ──────────────────────────────────────────────────

  Future<Map<String, dynamic>> _post(String path, Map<String, dynamic> body) async {
    final res = await _http
        .post(
          Uri.parse('$_prefix$path'),
          headers: _headers,
          body: jsonEncode(body),
        )
        .timeout(timeout);

    final json = jsonDecode(res.body) as Map<String, dynamic>;
    if (res.statusCode >= 400) {
      throw ApexVisionException(
        message:    json['message'] as String? ?? 'HTTP ${res.statusCode}',
        statusCode: res.statusCode,
      );
    }
    return json;
  }

  Future<Map<String, dynamic>> _get(String path) async {
    final res = await _http
        .get(Uri.parse('$_prefix$path'), headers: _headers)
        .timeout(timeout);
    final json = jsonDecode(res.body) as Map<String, dynamic>;
    if (res.statusCode >= 400) {
      throw ApexVisionException(message: json['message'] as String? ?? 'HTTP ${res.statusCode}', statusCode: res.statusCode);
    }
    return json;
  }

  // ── Análisis desde bytes ─────────────────────────────────────────

  Future<VisionResponse> analyzeBytes(
    Uint8List bytes, {
    List<String> tasks = const ['detect'],
    double confidenceThreshold = 0.5,
    bool useCache = true,
  }) async {
    final b64 = base64Encode(bytes);
    final json = await _post('/vision/analyze', {
      'image': {'format': 'base64', 'data': b64},
      'tasks': tasks,
      'options': {
        'confidence_threshold': confidenceThreshold,
        'use_cache': useCache,
      },
    });
    return VisionResponse.fromJson(json);
  }

  // ── Análisis desde File ──────────────────────────────────────────

  Future<VisionResponse> analyzeFile(
    File file, {
    List<String> tasks = const ['detect'],
    double confidenceThreshold = 0.5,
  }) async {
    final bytes = await file.readAsBytes();
    return analyzeBytes(bytes, tasks: tasks, confidenceThreshold: confidenceThreshold);
  }

  // ── Multipart upload (más eficiente para archivos grandes) ────────

  Future<VisionResponse> uploadFile(
    File file, {
    List<String> tasks = const ['detect'],
    double confidence = 0.5,
  }) async {
    final request = http.MultipartRequest(
      'POST', Uri.parse('$_prefix/vision/analyze/upload'),
    )
      ..headers.addAll({'X-ApexVision-Key': apiKey})
      ..fields['tasks']      = tasks.join(',')
      ..fields['confidence'] = confidence.toString()
      ..files.add(await http.MultipartFile.fromPath('file', file.path));

    final streamed = await request.send().timeout(timeout);
    final body     = await streamed.stream.bytesToString();
    final json     = jsonDecode(body) as Map<String, dynamic>;

    if (streamed.statusCode >= 400) {
      throw ApexVisionException(message: json['message'] as String? ?? 'Upload failed', statusCode: streamed.statusCode);
    }
    return VisionResponse.fromJson(json);
  }

  // ── Análisis desde URL ───────────────────────────────────────────

  Future<VisionResponse> analyzeUrl(
    String url, {
    List<String> tasks = const ['detect'],
    double confidenceThreshold = 0.5,
  }) async {
    final json = await _post('/vision/analyze', {
      'image': {'format': 'url', 'url': url},
      'tasks': tasks,
      'options': {'confidence_threshold': confidenceThreshold},
    });
    return VisionResponse.fromJson(json);
  }

  // ── Shortcuts ────────────────────────────────────────────────────

  Future<VisionResponse> detect(Uint8List bytes, {double confidence = 0.5}) =>
      analyzeBytes(bytes, tasks: ['detect'], confidenceThreshold: confidence);

  Future<VisionResponse> ocr(Uint8List bytes) =>
      analyzeBytes(bytes, tasks: ['ocr']);

  Future<VisionResponse> face(Uint8List bytes, {bool embeddings = false}) =>
      analyzeBytes(bytes, tasks: ['face'], confidenceThreshold: 0.5);

  Future<VisionResponse> classify(Uint8List bytes, {int topK = 5}) =>
      analyzeBytes(bytes, tasks: ['classify']);

  Future<VisionResponse> embed(Uint8List bytes) =>
      analyzeBytes(bytes, tasks: ['embed']);

  // ── Batch ─────────────────────────────────────────────────────────

  Future<String> submitBatch(
    List<Uint8List> images, {
    List<String> tasks = const ['detect'],
    String? jobName,
  }) async {
    final requests = images.map((bytes) => {
      'image': {'format': 'base64', 'data': base64Encode(bytes)},
      'tasks': tasks,
    }).toList();

    final json = await _post('/batch/submit', {
      'requests': requests,
      if (jobName != null) 'job_name': jobName,
    });
    return json['job_id'] as String;
  }

  Future<BatchJobStatus> getBatchStatus(String jobId) async {
    final json = await _get('/batch/$jobId');
    return BatchJobStatus.fromJson(json);
  }

  Future<BatchJobStatus> waitForBatch(
    String jobId, {
    Duration pollInterval = const Duration(seconds: 2),
    Duration timeout = const Duration(minutes: 5),
  }) async {
    final deadline = DateTime.now().add(timeout);
    while (DateTime.now().isBefore(deadline)) {
      final status = await getBatchStatus(jobId);
      if (['done', 'done_with_errors', 'failed'].contains(status.status)) {
        return status;
      }
      await Future<void>.delayed(pollInterval);
    }
    throw ApexVisionException(message: 'Batch $jobId timed out', statusCode: 408);
  }

  // ── Health ────────────────────────────────────────────────────────

  Future<bool> ping() async {
    try {
      final res = await _http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      final json = jsonDecode(res.body) as Map<String, dynamic>;
      return json['status'] == 'ok';
    } catch (_) { return false; }
  }

  // ── WebSocket streaming ───────────────────────────────────────────

  ApexVisionStream createStream({
    List<String> tasks = const ['detect'],
    double confidence  = 0.5,
  }) {
    final wsUrl = baseUrl.replaceFirst(RegExp(r'^http'), 'ws') + '/api/v1/stream/ws';
    return ApexVisionStream(wsUrl: wsUrl, tasks: tasks, confidence: confidence);
  }

  void dispose() => _http.close();
}

// ─────────────────────────────────────────────
//  WebSocket stream
// ─────────────────────────────────────────────

class ApexVisionStream {
  final String wsUrl;
  final List<String> tasks;
  final double confidence;

  WebSocketChannel? _channel;

  ApexVisionStream({required this.wsUrl, required this.tasks, required this.confidence});

  Stream<VisionResponse>? _resultStream;

  void connect() {
    _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
    _resultStream = _channel!.stream.map((data) {
      final json = jsonDecode(data as String) as Map<String, dynamic>;
      return VisionResponse.fromJson(json);
    });
  }

  Stream<VisionResponse> get results {
    if (_resultStream == null) throw StateError('Not connected. Call connect() first.');
    return _resultStream!;
  }

  void sendFrame(Uint8List frameBytes) {
    if (_channel == null) throw StateError('Not connected.');
    _channel!.sink.add(jsonEncode({
      'image':      base64Encode(frameBytes),
      'tasks':      tasks,
      'confidence': confidence,
    }));
  }

  void close() {
    _channel?.sink.close();
    _channel = null;
  }
}

// ─────────────────────────────────────────────
//  Error
// ─────────────────────────────────────────────

class ApexVisionException implements Exception {
  final String message;
  final int statusCode;

  const ApexVisionException({required this.message, required this.statusCode});

  @override
  String toString() => 'ApexVisionException($statusCode): $message';
}

// ─────────────────────────────────────────────
//  Ejemplo de uso en un Widget Flutter
// ─────────────────────────────────────────────

/*
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'apexvision_client.dart';

class VisionPage extends StatefulWidget {
  const VisionPage({super.key});
  @override State<VisionPage> createState() => _VisionPageState();
}

class _VisionPageState extends State<VisionPage> {
  final client = ApexVisionClient(
    baseUrl: 'http://192.168.1.100:8000',
    apiKey:  'apexvision-master-dev-key',
  );

  VisionResponse? _result;
  bool _loading = false;
  String? _error;

  Future<void> _pickAndAnalyze() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() { _loading = true; _error = null; });
    try {
      final bytes = await picked.readAsBytes();
      final result = await client.analyzeBytes(
        bytes,
        tasks: ['detect', 'ocr'],
        confidenceThreshold: 0.5,
      );
      setState(() { _result = result; });
    } on ApexVisionException catch (e) {
      setState(() { _error = e.message; });
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
              icon: const Icon(Icons.image),
              label: Text(_loading ? 'Analizando...' : 'Seleccionar imagen'),
            ),

            if (_error != null)
              Text(_error!, style: const TextStyle(color: Colors.red)),

            if (_result != null) ...[
              const SizedBox(height: 16),

              if (_result!.detection != null)
                Card(child: ListTile(
                  leading: const Icon(Icons.search, color: Colors.indigo),
                  title: Text('${_result!.detection!.count} objetos detectados'),
                  subtitle: Text(
                    _result!.detection!.boxes.take(3).map((b) =>
                      '${b.label} ${(b.confidence * 100).toStringAsFixed(0)}%'
                    ).join(', '),
                  ),
                )),

              if (_result!.ocr?.text.isNotEmpty == true)
                Card(child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Texto extraído', style: TextStyle(fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      Text(_result!.ocr!.text),
                    ],
                  ),
                )),

              Text(
                '${_result!.imageWidth}×${_result!.imageHeight} · ${_result!.totalInferenceMs.toStringAsFixed(1)}ms',
                style: const TextStyle(fontSize: 12, color: Colors.grey),
              ),
            ],
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    client.dispose();
    super.dispose();
  }
}
*/
