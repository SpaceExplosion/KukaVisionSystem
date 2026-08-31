"""Синтетический тест всей цепочки ТЗ без камеры и робота.

Генерируется кадр с двумя «кубиками» в известных пиксельных позициях,
создается известная матрица поля (масштаб + поворот), проверяются:
    1. нормализация угла в диапазон [-45, +45];
    2. детекция деталей (центр, угол);
    3. пересчет пиксель -> база робота и пересчет угла;
    4. расчет гомографии по реперным точкам;
    5. полный обмен по TCP: <Trigger>Capture</Trigger> -> XML-ответ.

Запуск: python test_pipeline.py
"""

from __future__ import annotations

import socket
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2 as cv
import numpy as np

from field_calibration import (
    FieldCalibration,
    compute_field_matrix,
    transform_angle_deg,
    transform_pixel_to_robot,
)
from server import VisionServer, build_vision_xml
from vision import PartDetector, VisionParams, normalize_angle

# Известная матрица поля: масштаб 0.5 мм/пиксель, поворот 10 град., сдвиг.
SCALE_MM_PER_PX = 0.5
ROTATION_DEG = 10.0
TX_MM, TY_MM = 100.0, -60.0

SQUARE_SIDE_PX = 60
IMAGE_SIZE = (640, 480)  # ширина, высота

# Эталонные детали: (угол в изображении, град).
GROUND_TRUTH = [
    {"angle_img_deg": 20.0, "center_px": (180.0, 160.0)},
    {"angle_img_deg": -35.0, "center_px": (430.0, 320.0)},
]


def field_matrix() -> np.ndarray:
    theta = np.radians(ROTATION_DEG)
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [SCALE_MM_PER_PX * cos, -SCALE_MM_PER_PX * sin, TX_MM],
            [SCALE_MM_PER_PX * sin, SCALE_MM_PER_PX * cos, TY_MM],
            [0.0, 0.0, 1.0],
        ]
    )


def make_field_calibration() -> FieldCalibration:
    """Калибровка поля, восстановленная по 4 реперным точкам."""
    pixel = np.array([[80, 80], [560, 80], [560, 400], [80, 400]], np.float64)
    robot = np.array(
        [transform_pixel_to_robot(field_matrix(), u, v) for u, v in pixel]
    )
    matrix = compute_field_matrix(pixel, robot)
    return FieldCalibration(
        matrix=matrix,
        pixel_points=pixel,
        robot_points=robot,
        residual_mm=np.linalg.norm(
            cv.perspectiveTransform(pixel.reshape(-1, 1, 2), matrix).reshape(-1, 2)
            - robot,
            axis=1,
        ),
    )


def make_frame() -> np.ndarray:
    frame = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 205, dtype=np.uint8)
    for part in GROUND_TRUTH:
        cx, cy = part["center_px"]
        box = cv.boxPoints(
            ((cx, cy), (SQUARE_SIDE_PX, SQUARE_SIDE_PX), part["angle_img_deg"])
        ).astype(np.int32)
        cv.fillConvexPoly(frame, box, (45, 45, 45))
    cv.circle(frame, (80, 80), 4, (0, 0, 0), -1)  # мелкий «мусор» — должен отсеяться
    return frame


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"Тест не пройден: {name} {detail}")


def test_normalize_angle() -> None:
    check("угол 90 -> 0", abs(normalize_angle(90.0)) < 1e-9)
    check("угол 70 -> -20", abs(normalize_angle(70.0) - (-20.0)) < 1e-9)
    check("угол -80 -> 10", abs(normalize_angle(-80.0) - 10.0) < 1e-9)


def test_detection(detector: PartDetector) -> list:
    parts = detector.detect(make_frame())
    check("найдено 2 детали", len(parts) == 2, f"найдено {len(parts)}")
    for part, truth in zip(parts, GROUND_TRUTH):
        du = abs(part.center[0] - truth["center_px"][0])
        dv = abs(part.center[1] - truth["center_px"][1])
        check(
            f"центр {truth['center_px']}",
            du <= 2.0 and dv <= 2.0,
            f"смещение ({du:.1f}, {dv:.1f}) px",
        )
        expected = normalize_angle(truth["angle_img_deg"])
        check(
            f"угол {truth['angle_img_deg']:>5} град",
            abs(part.angle_deg - expected) <= 2.0,
            f"получено {part.angle_deg:+.1f}",
        )
    return parts


def test_transform(parts: list, field: FieldCalibration) -> None:
    for part, truth in zip(parts, GROUND_TRUTH):
        u, v = part.center
        x, y = transform_pixel_to_robot(field.matrix, u, v)
        expected_x, expected_y = transform_pixel_to_robot(
            field_matrix(), *truth["center_px"]
        )
        check(
            f"координаты робота ({expected_x:.1f}, {expected_y:.1f})",
            abs(x - expected_x) <= 2.0 and abs(y - expected_y) <= 2.0,
            f"получено ({x:.1f}, {y:.1f}) мм",
        )
        angle = transform_angle_deg(field.matrix, u, v, part.angle_deg)
        expected = normalize_angle(truth["angle_img_deg"] + ROTATION_DEG)
        check(
            f"угол в базе робота {expected:+.1f}",
            abs(angle - expected) <= 2.0,
            f"получено {angle:+.1f}",
        )


def test_tcp(detector: PartDetector, field: FieldCalibration) -> None:
    image_path = Path(__file__).with_name("test_frame.png")
    cv.imwrite(str(image_path), make_frame())

    server = VisionServer(
        source=str(image_path),
        detector=detector,
        field=field,
        host="127.0.0.1",
        port=0,
        part_height_mm=15.0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    with socket.create_connection(server.address, timeout=5.0) as client:
        client.sendall(b"<Trigger>Capture</Trigger>")
        chunks = b""
        while b"</VisionResult>" not in chunks:
            chunk = client.recv(4096)
            if not chunk:
                break
            chunks += chunk

    server.stop()
    image_path.unlink(missing_ok=True)

    root = ET.fromstring(chunks.decode("ascii"))
    items = root.find("Item")
    check("XML: Item.Count = 2", items.get("Count") == "2",
          f"получено {items.get('Count')}")

    for element, truth in zip(root.findall("Item/Part"), GROUND_TRUTH):
        expected_x, expected_y = transform_pixel_to_robot(
            field_matrix(), *truth["center_px"]
        )
        expected_a = normalize_angle(truth["angle_img_deg"] + ROTATION_DEG)
        check(
            f"XML Part X/Y/A {truth['center_px']}",
            abs(float(element.get("X")) - expected_x) <= 2.0
            and abs(float(element.get("Y")) - expected_y) <= 2.0
            and abs(float(element.get("A")) - expected_a) <= 2.0,
            f"получено X={element.get('X')} Y={element.get('Y')} "
            f"A={element.get('A')}",
        )
        check(
            f"XML Part Z {truth['center_px']}",
            float(element.get("Z")) == 15.0,
        )


def main() -> int:
    print("=== Синтетический тест пайплайна Pick-and-Place ===\n")

    test_normalize_angle()
    print()

    field = make_field_calibration()
    check(
        "гомография по 4 точкам восстановлена",
        float(field.residual_mm.max()) < 0.01,
        f"максимальная невязка {field.residual_mm.max():.4f} мм",
    )
    print()

    detector = PartDetector(params=VisionParams(min_area_px=900.0))
    parts = test_detection(detector)
    print()

    test_transform(parts, field)
    print()

    check(
        "XML с ошибкой",
        'Error="x"' in build_vision_xml([], error="x"),
    )
    test_tcp(detector, field)

    print("\nВсе тесты пройдены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
