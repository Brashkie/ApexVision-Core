"""ApexVision-Core — YOLOv11 Detector Tests (16 tests)"""
import base64
import numpy as np
import pytest
from unittest.mock import MagicMock
from python.schemas.vision import BoundingBox, DetectionResult, VisionOptions
from python.core.detector import YOLODetector

def make_image(w=640, h=480): return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
def make_fake_box(label="person", conf=0.9, x1=10, y1=10, x2=200, y2=400):
    box = MagicMock()
    box.xyxy = [MagicMock()]; box.xyxy[0].tolist.return_value = [x1, y1, x2, y2]
    box.conf = [conf]; box.cls = [0]; return box
def make_fake_result(boxes_data, names={0: "person"}):
    result = MagicMock(); result.names = names
    result.boxes = [make_fake_box(**b) for b in boxes_data]; return [result]

def make_detector():
    d = YOLODetector.__new__(YOLODetector); d.model_name = "yolov11n.pt"; return d

def test_parse_results_basic():
    d = make_detector(); opts = VisionOptions(confidence_threshold=0.5)
    fake = make_fake_result([{"label":"person","conf":0.92,"x1":10,"y1":10,"x2":200,"y2":400},
                              {"label":"car","conf":0.75,"x1":300,"y1":100,"x2":500,"y2":350}], names={0:"person",1:"car"})
    fake[0].boxes[1].cls = [1]
    boxes = d._parse_results(fake, opts); assert len(boxes) == 2; assert boxes[0].label == "person"

def test_parse_results_confidence_filter():
    d = make_detector(); opts = VisionOptions(confidence_threshold=0.8)
    fake = make_fake_result([{"label":"person","conf":0.92},{"label":"person","conf":0.60},{"label":"person","conf":0.45}])
    boxes = d._parse_results(fake, opts); assert len(boxes) == 1

def test_parse_results_class_filter():
    d = make_detector(); opts = VisionOptions(confidence_threshold=0.3, classes_filter=["car"])
    fake = make_fake_result([{"label":"person","conf":0.9},{"label":"car","conf":0.8}], names={0:"person",1:"car"})
    fake[0].boxes[1].cls = [1]
    boxes = d._parse_results(fake, opts); assert len(boxes) == 1; assert boxes[0].label == "car"

def test_parse_results_sorted_by_confidence():
    d = make_detector(); opts = VisionOptions(confidence_threshold=0.1)
    fake = make_fake_result([{"label":"p","conf":0.55},{"label":"p","conf":0.99},{"label":"p","conf":0.72}])
    boxes = d._parse_results(fake, opts)
    confs = [b.confidence for b in boxes]; assert confs == sorted(confs, reverse=True)

def test_parse_results_max_detections():
    d = make_detector(); opts = VisionOptions(confidence_threshold=0.1, max_detections=2)
    fake = make_fake_result([{"label":"p","conf":c} for c in [0.9,0.8,0.7,0.6]])
    assert len(d._parse_results(fake, opts)) == 2

def test_parse_results_empty():
    d = make_detector(); result = MagicMock(); result.boxes = []
    assert d._parse_results([result], VisionOptions()) == []

def test_draw_boxes_returns_same_shape():
    img = make_image()
    boxes = [BoundingBox(x1=10,y1=10,x2=200,y2=300,width=190,height=290,confidence=0.9,label="person",label_id=0)]
    out = YOLODetector.draw_boxes(img, boxes); assert out.shape == img.shape; assert not np.array_equal(out, img)

def test_draw_boxes_empty_list():
    img = make_image(); out = YOLODetector.draw_boxes(img, []); assert np.array_equal(out, img)

def test_encode_preview_is_valid_base64_jpeg():
    b64 = YOLODetector.encode_preview(make_image())
    assert base64.b64decode(b64)[:2] == b"\xff\xd8"

def test_from_variant_valid():
    assert YOLODetector.from_variant("nano").model_name == "yolov11n.pt"

def test_from_variant_xlarge():
    assert YOLODetector.from_variant("xlarge").model_name == "yolov11x.pt"

def test_from_variant_invalid():
    with pytest.raises(ValueError, match="Unknown variant"): YOLODetector.from_variant("nonexistent")

def test_clear_cache():
    YOLODetector._cache["fake_cpu"] = object(); YOLODetector.clear_cache(); assert YOLODetector._cache == {}

def test_loaded_models_empty_after_clear():
    YOLODetector.clear_cache(); assert YOLODetector.loaded_models() == []

@pytest.mark.asyncio
async def test_run_with_mocked_model():
    fake_results = make_fake_result([
        {"label":"person","conf":0.95,"x1":50,"y1":50,"x2":300,"y2":500},
        {"label":"person","conf":0.81,"x1":350,"y1":60,"x2":580,"y2":480},
    ])
    detector = YOLODetector(model_name="yolov11n.pt"); detector.device = "cpu"
    YOLODetector._cache[detector._cache_key] = MagicMock(return_value=fake_results)
    result = await detector.run(make_image(), VisionOptions(confidence_threshold=0.5))
    assert isinstance(result, DetectionResult); assert result.count == 2
    assert result.boxes[0].confidence >= result.boxes[1].confidence
    del YOLODetector._cache[detector._cache_key]

@pytest.mark.asyncio
async def test_run_returns_empty_on_no_detections():
    no_result = MagicMock(); no_result.boxes = []
    detector = YOLODetector(model_name="yolov11n.pt")
    YOLODetector._cache[detector._cache_key] = MagicMock(return_value=[no_result])
    result = await detector.run(make_image(), VisionOptions())
    assert result.count == 0; assert result.boxes == []
    del YOLODetector._cache[detector._cache_key]
