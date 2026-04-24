"""ApexVision-Core — Pydantic v2 Schemas"""
from __future__ import annotations
import base64
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator, model_validator


class VisionTask(StrEnum):
    DETECT     = "detect"
    CLASSIFY   = "classify"
    SEGMENT    = "segment"
    OCR        = "ocr"
    FACE       = "face"
    DEPTH      = "depth"
    EMBED      = "embed"
    CAPTION    = "caption"
    SIMILARITY = "similarity"
    CUSTOM     = "custom"


class OutputFormat(StrEnum):
    JSON    = "json"
    PARQUET = "parquet"
    DELTA   = "delta"


class ImageFormat(StrEnum):
    BASE64 = "base64"
    URL    = "url"
    PATH   = "path"


class ApexBaseModel(BaseModel):
    model_config = {"populate_by_name": True, "use_enum_values": True}


class ImageInput(ApexBaseModel):
    format: ImageFormat = ImageFormat.BASE64
    data: str | None = Field(None, description="Base64 image data")
    url: str | None = Field(None, description="Public image URL")

    @model_validator(mode="after")
    def validate_source(self) -> "ImageInput":
        if self.format == ImageFormat.BASE64 and not self.data:
            raise ValueError("data is required when format='base64'")
        if self.format == ImageFormat.URL and not self.url:
            raise ValueError("url is required when format='url'")
        return self

    @field_validator("data", mode="before")
    @classmethod
    def strip_data_uri(cls, v: str | None) -> str | None:
        if v and v.startswith("data:"):
            return v.split(",", 1)[-1]
        return v

    def decode_bytes(self) -> bytes:
        if self.format == ImageFormat.BASE64 and self.data:
            return base64.b64decode(self.data)
        raise ValueError("Cannot decode non-base64 image directly")


class VisionOptions(ApexBaseModel):
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    iou_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.45
    max_detections: Annotated[int, Field(ge=1, le=1000)] = 100
    classes_filter: list[str] | None = None
    top_k: Annotated[int, Field(ge=1, le=1000)] = 5
    segment_mask_format: str = "rle"
    ocr_language: str = "eng"
    ocr_mode: str = "full"
    face_landmarks: bool = True
    face_attributes: bool = True
    face_embeddings: bool = False
    custom_model_id: str | None = None
    use_cache: bool = True
    device_override: str | None = None


class VisionRequest(ApexBaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    image: ImageInput
    tasks: list[VisionTask] = Field(default=[VisionTask.DETECT], min_length=1, max_length=10)
    options: VisionOptions = Field(default_factory=lambda: VisionOptions())
    output_format: OutputFormat = OutputFormat.JSON
    store_result: bool = False


class BoundingBox(ApexBaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    confidence: float
    label: str
    label_id: int


class DetectionResult(ApexBaseModel):
    boxes: list[BoundingBox] = []
    count: int = 0
    model_used: str = ""
    inference_ms: float = 0.0


class ClassificationResult(ApexBaseModel):
    predictions: list[dict[str, Any]] = []
    model_used: str = ""
    inference_ms: float = 0.0


class OCRResult(ApexBaseModel):
    text: str = ""
    blocks: list[dict[str, Any]] = []
    language_detected: str = ""
    inference_ms: float = 0.0


class FaceResult(ApexBaseModel):
    faces: list[dict[str, Any]] = []
    count: int = 0
    inference_ms: float = 0.0


class EmbeddingResult(ApexBaseModel):
    embedding: list[float] = []
    dimensions: int = 0
    model_used: str = ""
    inference_ms: float = 0.0


class DepthResult(ApexBaseModel):
    depth_map_base64: str = ""
    min_depth: float = 0.0
    max_depth: float = 0.0
    inference_ms: float = 0.0


class SegmentationResult(ApexBaseModel):
    masks: list[dict[str, Any]] = []
    count: int = 0
    inference_ms: float = 0.0


class VisionResponse(ApexBaseModel):
    request_id: UUID
    status: str = "success"
    tasks_ran: list[VisionTask] = []
    detection: DetectionResult | None = None
    classification: ClassificationResult | None = None
    ocr: OCRResult | None = None
    face: FaceResult | None = None
    embedding: EmbeddingResult | None = None
    depth: DepthResult | None = None
    segmentation: SegmentationResult | None = None
    image_width: int = 0
    image_height: int = 0
    total_inference_ms: float = 0.0
    stored_at: str | None = None


class BatchRequest(ApexBaseModel):
    requests: list[VisionRequest] = Field(min_length=1, max_length=500)
    job_name: str | None = None
    webhook_url: str | None = None
    output_format: OutputFormat = OutputFormat.PARQUET


class BatchJobStatus(ApexBaseModel):
    job_id: str
    status: str
    total: int
    completed: int
    failed: int
    progress_pct: float
    result_path: str | None = None
    created_at: str
    updated_at: str


class ErrorResponse(ApexBaseModel):
    error: str
    message: str
    request_id: str | None = None
    detail: Any | None = None
