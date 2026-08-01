from __future__ import annotations

from ultralytics import YOLO
import cv2
import numpy as np

from src.qiming.models import DetectionResult, DetectedObject, ImageFrame


class YoloDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.yolo_model = YOLO(model_path)

    def detect_by_class(self, yolo_class: str, frame) -> DetectionResult:
        if frame is None:
            return DetectionResult(
                success=False,
                detected=False,
                message="输入帧无效"
            )

        class_idx = None
        for idx, name in self.yolo_model.names.items():
            if name == yolo_class:
                class_idx = idx
                break
        if class_idx is None:
            return DetectionResult(
                success=False,
                detected=False,
                message=f"YOLO不支持该类别：{yolo_class}"
            )

        results = self.yolo_model(frame, classes=[class_idx], conf=0.5)

        detected_objects = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                detected_objects.append(DetectedObject(
                    class_name=yolo_class,
                    coordinates=(x1, y1, x2, y2),
                    confidence=confidence,
                    center=((x1 + x2) / 2, (y1 + y2) / 2)
                ))

        return DetectionResult(
            success=True,
            detected=len(detected_objects) > 0,
            objects=detected_objects
        )

    def detect(self, image_frame: ImageFrame, target_class: str = "cup") -> DetectionResult:
        frame = image_frame.color_frame if image_frame.color_frame is not None else image_frame.data
        return self.detect_by_class(target_class, frame)