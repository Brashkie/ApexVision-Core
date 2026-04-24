/**
 * ApexVision-Core — Kotlin / Android Client
 * Retrofit 2 + OkHttp + Coroutines + Gson
 *
 * Dependencias (build.gradle.kts):
 *   implementation("com.squareup.retrofit2:retrofit:2.11.0")
 *   implementation("com.squareup.retrofit2:converter-gson:2.11.0")
 *   implementation("com.squareup.okhttp3:okhttp:4.12.0")
 *   implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")
 *   implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")
 *   implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.0")
 *
 * AndroidManifest.xml:
 *   <uses-permission android:name="android.permission.INTERNET"/>
 *
 * Para desarrollo local (HTTP plano):
 *   res/xml/network_security_config.xml → allowCleartextTraffic
 */

package com.hepein.apexvision

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.content.Context
import android.util.Base64
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import okhttp3.*
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.*
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

// ─────────────────────────────────────────────
//  Modelos de datos
// ─────────────────────────────────────────────

data class ImageInput(
    val format: String = "base64",
    val data: String? = null,
    val url: String? = null,
)

data class VisionOptions(
    @SerializedName("confidence_threshold") val confidenceThreshold: Float = 0.5f,
    @SerializedName("iou_threshold")        val iouThreshold: Float = 0.45f,
    @SerializedName("max_detections")       val maxDetections: Int = 100,
    @SerializedName("top_k")               val topK: Int = 5,
    @SerializedName("ocr_language")        val ocrLanguage: String = "en",
    @SerializedName("face_landmarks")      val faceLandmarks: Boolean = true,
    @SerializedName("face_attributes")     val faceAttributes: Boolean = true,
    @SerializedName("face_embeddings")     val faceEmbeddings: Boolean = false,
    @SerializedName("use_cache")           val useCache: Boolean = true,
    @SerializedName("classes_filter")      val classesFilter: List<String>? = null,
)

data class VisionRequest(
    val image: ImageInput,
    val tasks: List<String> = listOf("detect"),
    val options: VisionOptions = VisionOptions(),
    @SerializedName("store_result") val storeResult: Boolean = false,
)

data class BoundingBox(
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    val width: Float, val height: Float,
    val confidence: Float,
    val label: String,
    @SerializedName("label_id") val labelId: Int,
)

data class DetectionResult(
    val boxes: List<BoundingBox>,
    val count: Int,
    @SerializedName("model_used")    val modelUsed: String,
    @SerializedName("inference_ms") val inferenceMs: Float,
)

data class ClassificationPrediction(
    val label: String,
    val confidence: Float,
    @SerializedName("label_id") val labelId: Int,
)

data class ClassificationResult(
    val predictions: List<ClassificationPrediction>,
    @SerializedName("model_used")    val modelUsed: String,
    @SerializedName("inference_ms") val inferenceMs: Float,
)

data class OCRResult(
    val text: String,
    @SerializedName("language_detected") val languageDetected: String,
    @SerializedName("inference_ms")      val inferenceMs: Float,
    val blocks: List<Map<String, Any>>,
)

data class FaceAttributes(
    val age: Int?,
    val gender: String?,
    val emotion: String?,
    @SerializedName("emotion_scores") val emotionScores: Map<String, Float>?,
)

data class FaceEntry(
    val bbox: BoundingBox,
    val attributes: FaceAttributes?,
    val landmarks: Map<String, Map<String, Float>>?,
    val embedding: List<Float>?,
)

data class FaceResult(
    val faces: List<FaceEntry>,
    val count: Int,
    @SerializedName("inference_ms") val inferenceMs: Float,
)

data class EmbeddingResult(
    val embedding: List<Float>,
    val dimensions: Int,
    @SerializedName("model_used")    val modelUsed: String,
    @SerializedName("inference_ms") val inferenceMs: Float,
)

data class VisionResponse(
    @SerializedName("request_id")          val requestId: String,
    val status: String,
    @SerializedName("tasks_ran")           val tasksRan: List<String>,
    val detection: DetectionResult?,
    val classification: ClassificationResult?,
    val ocr: OCRResult?,
    val face: FaceResult?,
    val embedding: EmbeddingResult?,
    @SerializedName("image_width")         val imageWidth: Int,
    @SerializedName("image_height")        val imageHeight: Int,
    @SerializedName("total_inference_ms") val totalInferenceMs: Float,
    @SerializedName("stored_at")           val storedAt: String?,
)

data class BatchSubmitResponse(@SerializedName("job_id") val jobId: String)

data class BatchJobStatus(
    @SerializedName("job_id")       val jobId: String,
    val status: String,
    val total: Int,
    val completed: Int,
    val failed: Int,
    @SerializedName("progress_pct") val progressPct: Float,
    @SerializedName("result_path") val resultPath: String?,
    @SerializedName("elapsed_ms")  val elapsedMs: Float?,
)

data class BatchRequest(
    val requests: List<VisionRequest>,
    @SerializedName("job_name") val jobName: String? = null,
    @SerializedName("webhook_url") val webhookUrl: String? = null,
)

data class HealthResponse(val status: String, val service: String, val version: String)

// ─────────────────────────────────────────────
//  Retrofit API interface
// ─────────────────────────────────────────────

interface ApexVisionApi {
    @POST("api/v1/vision/analyze")
    suspend fun analyze(@Body request: VisionRequest): VisionResponse

    @Multipart
    @POST("api/v1/vision/analyze/upload")
    suspend fun uploadFile(
        @Part file: MultipartBody.Part,
        @Part("tasks") tasks: RequestBody,
        @Part("confidence") confidence: RequestBody,
    ): VisionResponse

    @POST("api/v1/batch/submit")
    suspend fun submitBatch(@Body request: BatchRequest): BatchSubmitResponse

    @GET("api/v1/batch/{jobId}")
    suspend fun getBatchStatus(@Path("jobId") jobId: String): BatchJobStatus

    @DELETE("api/v1/batch/{jobId}")
    suspend fun cancelBatch(@Path("jobId") jobId: String): Map<String, String>

    @GET("health")
    suspend fun health(): HealthResponse
}

// ─────────────────────────────────────────────
//  Cliente principal
// ─────────────────────────────────────────────

class ApexVisionClient(
    private val baseUrl: String,
    private val apiKey: String,
    private val timeoutSeconds: Long = 60L,
    private val debug: Boolean = false,
) {
    private val api: ApexVisionApi

    init {
        val logging = HttpLoggingInterceptor().apply {
            level = if (debug) HttpLoggingInterceptor.Level.BODY
                    else       HttpLoggingInterceptor.Level.NONE
        }

        val okHttp = OkHttpClient.Builder()
            .connectTimeout(timeoutSeconds, TimeUnit.SECONDS)
            .readTimeout(timeoutSeconds, TimeUnit.SECONDS)
            .writeTimeout(timeoutSeconds, TimeUnit.SECONDS)
            .addInterceptor(logging)
            .addInterceptor { chain ->
                // Inyecta API key en todos los requests
                val req = chain.request().newBuilder()
                    .addHeader("X-ApexVision-Key", apiKey)
                    .build()
                chain.proceed(req)
            }
            .build()

        api = Retrofit.Builder()
            .baseUrl(baseUrl.trimEnd('/') + "/")
            .client(okHttp)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ApexVisionApi::class.java)
    }

    // ── Análisis desde Bitmap ─────────────────────────────────────────

    suspend fun analyzeBitmap(
        bitmap: Bitmap,
        tasks: List<String> = listOf("detect"),
        options: VisionOptions = VisionOptions(),
    ): VisionResponse {
        val b64 = bitmapToBase64(bitmap)
        return api.analyze(
            VisionRequest(
                image   = ImageInput(format = "base64", data = b64),
                tasks   = tasks,
                options = options,
            )
        )
    }

    // ── Análisis desde Uri (Android) ───────────────────────────────────

    suspend fun analyzeUri(
        context: Context,
        uri: Uri,
        tasks: List<String> = listOf("detect"),
        options: VisionOptions = VisionOptions(),
    ): VisionResponse {
        val bitmap = withContext(Dispatchers.IO) {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream)
            } ?: throw IllegalArgumentException("Cannot open image: $uri")
        }
        return analyzeBitmap(bitmap, tasks, options)
    }

    // ── Análisis desde URL ─────────────────────────────────────────────

    suspend fun analyzeUrl(
        url: String,
        tasks: List<String> = listOf("detect"),
        options: VisionOptions = VisionOptions(),
    ): VisionResponse = api.analyze(
        VisionRequest(
            image   = ImageInput(format = "url", url = url),
            tasks   = tasks,
            options = options,
        )
    )

    // ── Shortcuts ──────────────────────────────────────────────────────

    suspend fun detect(bitmap: Bitmap, confidence: Float = 0.5f) =
        analyzeBitmap(bitmap, listOf("detect"), VisionOptions(confidenceThreshold = confidence))

    suspend fun ocr(bitmap: Bitmap) =
        analyzeBitmap(bitmap, listOf("ocr"))

    suspend fun face(bitmap: Bitmap, embeddings: Boolean = false) =
        analyzeBitmap(bitmap, listOf("face"), VisionOptions(faceEmbeddings = embeddings))

    suspend fun classify(bitmap: Bitmap, topK: Int = 5) =
        analyzeBitmap(bitmap, listOf("classify"), VisionOptions(topK = topK))

    suspend fun embed(bitmap: Bitmap) =
        analyzeBitmap(bitmap, listOf("embed"))

    // ── Batch ──────────────────────────────────────────────────────────

    suspend fun submitBatch(
        bitmaps: List<Bitmap>,
        tasks: List<String> = listOf("detect"),
        jobName: String? = null,
    ): String {
        val requests = bitmaps.map { bmp ->
            VisionRequest(
                image = ImageInput(format = "base64", data = bitmapToBase64(bmp)),
                tasks = tasks,
            )
        }
        val response = api.submitBatch(BatchRequest(requests = requests, jobName = jobName))
        return response.jobId
    }

    suspend fun getBatchStatus(jobId: String): BatchJobStatus =
        api.getBatchStatus(jobId)

    suspend fun waitForBatch(
        jobId: String,
        pollIntervalMs: Long = 2_000L,
        timeoutMs: Long = 300_000L,
    ): BatchJobStatus {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            val status = getBatchStatus(jobId)
            if (status.status in listOf("done", "done_with_errors", "failed")) return status
            delay(pollIntervalMs)
        }
        throw ApexVisionException("Batch $jobId timed out", 408)
    }

    // ── Health ─────────────────────────────────────────────────────────

    suspend fun ping(): Boolean = runCatching { api.health().status == "ok" }.getOrDefault(false)

    // ── Util ───────────────────────────────────────────────────────────

    private fun bitmapToBase64(bitmap: Bitmap): String {
        val out = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
        return Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP)
    }
}

// ─────────────────────────────────────────────
//  Error
// ─────────────────────────────────────────────

class ApexVisionException(message: String, val statusCode: Int) : Exception(message)

// ─────────────────────────────────────────────
//  ViewModel de ejemplo
// ─────────────────────────────────────────────

sealed class VisionUiState {
    data object Idle    : VisionUiState()
    data object Loading : VisionUiState()
    data class  Success(val response: VisionResponse) : VisionUiState()
    data class  Error(val message: String)            : VisionUiState()
}

class VisionViewModel : ViewModel() {

    private val client = ApexVisionClient(
        baseUrl = "http://192.168.1.100:8000",  // IP local del servidor
        apiKey  = "apexvision-master-dev-key",
        debug   = true,
    )

    private val _state = MutableStateFlow<VisionUiState>(VisionUiState.Idle)
    val state: StateFlow<VisionUiState> = _state.asStateFlow()

    /** Detecta objetos en un Bitmap desde la galería o cámara */
    fun detect(bitmap: Bitmap) {
        viewModelScope.launch {
            _state.value = VisionUiState.Loading
            _state.value = try {
                val result = client.detect(bitmap, confidence = 0.5f)
                VisionUiState.Success(result)
            } catch (e: ApexVisionException) {
                VisionUiState.Error("API Error (${e.statusCode}): ${e.message}")
            } catch (e: Exception) {
                VisionUiState.Error("Network error: ${e.message}")
            }
        }
    }

    /** Análisis multi-task: detección + OCR */
    fun analyzeMulti(bitmap: Bitmap) {
        viewModelScope.launch {
            _state.value = VisionUiState.Loading
            _state.value = try {
                val result = client.analyzeBitmap(
                    bitmap = bitmap,
                    tasks  = listOf("detect", "ocr", "classify"),
                    options = VisionOptions(
                        confidenceThreshold = 0.6f,
                        topK = 3,
                    ),
                )
                VisionUiState.Success(result)
            } catch (e: Exception) {
                VisionUiState.Error(e.message ?: "Unknown error")
            }
        }
    }

    /** Batch: analiza múltiples fotos seleccionadas */
    fun analyzeBatch(context: Context, uris: List<Uri>) {
        viewModelScope.launch {
            _state.value = VisionUiState.Loading
            try {
                val bitmaps = uris.mapNotNull { uri ->
                    withContext(Dispatchers.IO) {
                        context.contentResolver.openInputStream(uri)?.use {
                            BitmapFactory.decodeStream(it)
                        }
                    }
                }

                val jobId  = client.submitBatch(bitmaps, tasks = listOf("detect"), jobName = "gallery-batch")
                val status = client.waitForBatch(jobId)

                // status.resultPath tiene la ruta al .parquet con todos los resultados
                _state.value = VisionUiState.Error("Batch done: ${status.completed}/${status.total} imágenes")
            } catch (e: Exception) {
                _state.value = VisionUiState.Error(e.message ?: "Batch error")
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        // client no necesita close() explícito — OkHttp maneja el pool
    }
}

/*
// ─────────────────────────────────────────────
//  Uso en un Fragment / Composable
// ─────────────────────────────────────────────

// Fragment tradicional:
class VisionFragment : Fragment() {
    private val vm: VisionViewModel by viewModels()

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        // Observar estado
        viewLifecycleOwner.lifecycleScope.launch {
            vm.state.collect { state ->
                when (state) {
                    is VisionUiState.Loading -> showLoading()
                    is VisionUiState.Success -> showResult(state.response)
                    is VisionUiState.Error   -> showError(state.message)
                    is VisionUiState.Idle    -> Unit
                }
            }
        }

        // Analizar imagen
        btnAnalyze.setOnClickListener {
            val bitmap = getSelectedBitmap()  // tu lógica de selección
            vm.detect(bitmap)
        }
    }
}

// Jetpack Compose:
@Composable
fun VisionScreen(vm: VisionViewModel = viewModel()) {
    val state by vm.state.collectAsState()

    Column(modifier = Modifier.padding(16.dp)) {
        when (val s = state) {
            is VisionUiState.Loading -> CircularProgressIndicator()
            is VisionUiState.Error   -> Text(s.message, color = MaterialTheme.colorScheme.error)
            is VisionUiState.Success -> {
                s.response.detection?.let { det ->
                    Text("${det.count} objetos detectados")
                    det.boxes.take(5).forEach { box ->
                        Text("${box.label}: ${(box.confidence * 100).toInt()}%")
                    }
                }
                s.response.ocr?.text?.let { text ->
                    if (text.isNotBlank()) Text("Texto: $text")
                }
            }
            else -> Unit
        }
    }
}
*/
