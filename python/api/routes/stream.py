"""ApexVision-Core — WebSocket Streaming Router"""
import json
import base64
import numpy as np
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()

@router.websocket("/ws")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time frame analysis via WebSocket.
    Client sends: {"image": "<base64>", "tasks": ["detect"], "confidence": 0.5}
    Server sends: VisionResponse JSON
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")
    from python.core.pipeline import VisionPipeline
    from python.schemas.vision import VisionRequest, ImageInput, ImageFormat, VisionOptions, VisionTask
    pipeline = VisionPipeline()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            req = VisionRequest(
                image=ImageInput(format=ImageFormat.BASE64, data=payload.get("image")),
                tasks=[VisionTask(t) for t in payload.get("tasks", ["detect"])],
                options=VisionOptions(confidence_threshold=payload.get("confidence", 0.5)),
            )
            result = await pipeline.run(req)
            await websocket.send_text(result.model_dump_json())
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011)
