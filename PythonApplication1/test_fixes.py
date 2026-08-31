"""Регрессионные тесты для исправленных дефектов.

Каждый тест закрывает конкретную ошибку, найденную при разборе проекта,
и падает, если ошибка вернется. Запуск: python test_fixes.py
"""

from __future__ import annotations

import argparse
import socket
import threading
import xml.etree.ElementTree as ET

import cv2 as cv
import numpy as np

from field_calibration import compute_field_matrix
from server import FrameSource, RobotPart, VisionServer, build_vision_xml
from vision import (
    PartDetector,
    VisionParams,
    add_vision_arguments,
    detect_parts,
    params_from_args,
)

PASSED = 0
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"[OK] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAILED.append(f"{name} {detail}".strip())
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


def _dark_object_frame() -> np.ndarray:
    """Темный кубик на светлом фоне: центр (130, 130), площадь 3600 px."""
    frame = np.full((480, 640, 3), 205, np.uint8)
    cv.rectangle(frame, (100, 100), (160, 160), (40, 40, 40), -1)
    return frame


def _bright_object_frame() -> np.ndarray:
    """Светлый кубик на темном фоне: центр (130, 130), площадь 3600 px."""
    frame = np.full((480, 640, 3), 40, np.uint8)
    cv.rectangle(frame, (100, 100), (160, 160), (210, 210, 210), -1)
    return frame


# --------------------------------------------------------------------------
# 1. vision.py: полярность бинаризации была инвертирована
# --------------------------------------------------------------------------

def test_polarity() -> None:
    print("\n--- polarity: объект, а не фон ---")
    cases = (
        ("dark-obj  polarity=dark", _dark_object_frame(), "dark"),
        ("dark-obj  polarity=auto", _dark_object_frame(), "auto"),
        ("bright-obj polarity=bright", _bright_object_frame(), "bright"),
        ("bright-obj polarity=auto", _bright_object_frame(), "auto"),
    )
    for name, frame, polarity in cases:
        parts = detect_parts(
            frame, VisionParams(min_area_px=900.0, polarity=polarity)
        )
        ok = len(parts) == 1
        detail = f"найдено {len(parts)}"
        if ok:
            cx, cy = parts[0].center
            # Фон дал бы центр кадра (319, 239) и площадь ~306000.
            ok = abs(cx - 130) < 3 and abs(cy - 130) < 3 and parts[0].area_px < 5000
            detail = f"центр ({cx:.0f}, {cy:.0f}), площадь {parts[0].area_px:.0f}"
        check(name, ok, detail)


def test_polarity_forced_is_not_auto() -> None:
    """Принудительная полярность должна отличаться от автоматической."""
    frame = _dark_object_frame()
    wrong = detect_parts(frame, VisionParams(min_area_px=900.0, polarity="bright"))
    ok = len(wrong) == 1 and wrong[0].area_px > 100000
    check(
        "polarity=bright на темном объекте берет фон (ожидаемо)",
        ok,
        f"площадь {wrong[0].area_px:.0f}" if wrong else "ничего не найдено",
    )


# --------------------------------------------------------------------------
# 2. vision.py: CLI-параметры
# --------------------------------------------------------------------------

def test_cli_params() -> None:
    print("\n--- CLI параметры пайплайна ---")
    parser = argparse.ArgumentParser()
    add_vision_arguments(parser)

    only_high = params_from_args(parser.parse_args(["--hsv-high", "10,20,30"]))
    check("--hsv-high в одиночку включает HSV", only_high.use_hsv is True)
    check("--hsv-high применяется", only_high.hsv_high == (10, 20, 30))

    only_low = params_from_args(parser.parse_args(["--hsv-low", "5,6,7"]))
    check("--hsv-low в одиночку включает HSV", only_low.use_hsv is True)

    no_hsv = params_from_args(parser.parse_args([]))
    check("без границ HSV выключен", no_hsv.use_hsv is False)

    aspect = params_from_args(parser.parse_args(["--aspect-max", "0.8"]))
    check("--aspect-max существует и применяется", aspect.aspect_max == 0.8)


def test_aspect_swapped() -> None:
    """Перепутанные границы аспекта не должны отбрасывать все детали."""
    params = VisionParams(min_area_px=900.0, aspect_min=1.0, aspect_max=0.3)
    parts = detect_parts(_dark_object_frame(), params)
    check(
        "aspect_min > aspect_max не ломает детекцию",
        len(parts) == 1,
        f"найдено {len(parts)}",
    )


def test_hsv_blur_applied() -> None:
    """В HSV-ветке сглаживание должно учитываться (раньше игнорировалось)."""
    frame = _bright_object_frame()
    noisy = frame.copy()
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 60, noisy.shape, dtype=np.int16)
    noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    params = VisionParams(
        min_area_px=900.0, use_hsv=True, hsv_low=(0, 0, 150),
        hsv_high=(179, 60, 255), blur_ksize=9,
    )
    parts = detect_parts(noisy, params)
    check(
        "HSV-ветка использует GaussianBlur",
        len(parts) >= 1,
        f"найдено {len(parts)}",
    )


# --------------------------------------------------------------------------
# 3. field_calibration.py: рассогласование числа точек
# --------------------------------------------------------------------------

def test_field_point_mismatch() -> None:
    print("\n--- калибровка поля: несовпадение числа точек ---")
    pixels = [(0, 0), (100, 0), (100, 100), (0, 100), (50, 50)]
    robots = [(0, 0), (50, 0), (50, 50), (0, 50), (25, 25), (10, 10)]
    try:
        compute_field_matrix(pixels, robots)
        check("несовпадение числа точек -> ValueError", False, "исключения не было")
    except ValueError as exc:
        # Сообщение должно называть обе величины, а не «нужно минимум 3».
        text = str(exc)
        check(
            "несовпадение числа точек -> понятный ValueError",
            "5" in text and "6" in text,
            text,
        )

    try:
        compute_field_matrix([(0, 0), (1, 1)], [(0, 0), (1, 1)])
        check("двух точек недостаточно -> ValueError", False, "исключения не было")
    except ValueError as exc:
        check("двух точек недостаточно -> ValueError", True, str(exc))


def test_field_matrix_valid() -> None:
    """Корректный набор точек по-прежнему дает точную гомографию."""
    pixels = np.array([[80, 80], [560, 80], [560, 400], [80, 400]], np.float64)
    robots = pixels * 0.5 + np.array([100.0, -60.0])
    matrix = compute_field_matrix(pixels, robots)
    mapped = cv.perspectiveTransform(pixels.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    error = float(np.abs(mapped - robots).max())
    check("гомография по 4 точкам точна", error < 1e-6, f"ошибка {error:.2e} мм")


# --------------------------------------------------------------------------
# 4. server.py: XML-ответ
# --------------------------------------------------------------------------

def test_xml_escaping() -> None:
    print("\n--- XML-ответ роботу ---")
    xml = build_vision_xml([], error='sock "x" & <y>')
    try:
        item = ET.fromstring(xml).find("Item")
        check("XML с кавычками/& разбирается", True)
        check(
            "текст ошибки не искажен",
            item.get("Error") == 'sock "x" & <y>',
            repr(item.get("Error")),
        )
    except ET.ParseError as exc:
        check("XML с кавычками/& разбирается", False, str(exc))
        check("текст ошибки не искажен", False, "XML не разобран")


def test_xml_non_finite() -> None:
    for label, part in (
        ("nan", RobotPart(float("nan"), 1.0, 2.0, 3.0)),
        ("inf", RobotPart(1.0, float("inf"), 2.0, 3.0)),
    ):
        xml = build_vision_xml([part])
        item = ET.fromstring(xml).find("Item")
        check(
            f"{label} не уходит роботу",
            item.get("Count") == "0" and item.get("Error") == "non-finite-coordinates",
            f'Count={item.get("Count")}, Error={item.get("Error")}',
        )


def test_xml_valid_parts() -> None:
    xml = build_vision_xml([
        RobotPart(450.2, -120.5, 15.0, 24.8),
        RobotPart(510.0, -80.3, 15.0, -12.1),
    ])
    root = ET.fromstring(xml)
    parts = root.findall("Item/Part")
    check("две корректные детали в XML", len(parts) == 2, f"найдено {len(parts)}")
    check('Count="2"', root.find("Item").get("Count") == "2")
    check("Index нумеруется с 1", [p.get("Index") for p in parts] == ["1", "2"])


def test_xml_ascii_encodable() -> None:
    """Ответ должен кодироваться в ASCII даже с русским текстом ошибки."""
    xml = build_vision_xml([], error="камера не отвечает")
    try:
        xml.encode("ascii", errors="replace")
        check("не-ASCII ошибка кодируется без исключения", True)
    except UnicodeEncodeError as exc:
        check("не-ASCII ошибка кодируется без исключения", False, str(exc))


# --------------------------------------------------------------------------
# 5. server.py: порт не должен утекать при отказе источника
# --------------------------------------------------------------------------

def test_port_released_on_source_failure() -> None:
    print("\n--- сервер: освобождение порта ---")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    try:
        VisionServer(
            detector=PartDetector(),
            field=None,
            source="/nonexistent/definitely-missing-source.mp4",
            host="127.0.0.1",
            port=port,
        )
        check("отказ источника -> исключение", False, "исключения не было")
        return
    except Exception as exc:
        check("отказ источника -> исключение", True, type(exc).__name__)

    # Порт обязан быть свободен: слушатель закрыт в except-ветке конструктора.
    retry = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        retry.bind(("127.0.0.1", port))
        check("порт освобожден после отказа источника", True, f"порт {port}")
    except OSError as exc:
        check("порт освобожден после отказа источника", False, str(exc))
    finally:
        retry.close()


# --------------------------------------------------------------------------
# 6. server.py: FrameSource на статичном снимке и полный обмен
# --------------------------------------------------------------------------

def test_frame_source_release_idempotent() -> None:
    print("\n--- FrameSource ---")
    from pathlib import Path
    path = Path(__file__).with_name("_test_fixes_frame.png")
    cv.imwrite(str(path), _dark_object_frame())
    try:
        source = FrameSource(str(path))
        check("статичный снимок распознан", source.is_static is True)
        check("кадр читается", source.read().shape == (480, 640, 3))
        source.release()
        source.release()  # повторный release не должен падать
        check("повторный release безопасен", True)
    finally:
        path.unlink(missing_ok=True)


def test_tcp_roundtrip_error_response() -> None:
    """Без калибровки поля сервер отвечает корректным XML с ошибкой."""
    from pathlib import Path
    path = Path(__file__).with_name("_test_fixes_tcp.png")
    cv.imwrite(str(path), _dark_object_frame())
    server = None
    try:
        server = VisionServer(
            detector=PartDetector(params=VisionParams(min_area_px=900.0)),
            field=None,
            source=str(path),
            host="127.0.0.1",
            port=0,
            log=lambda _m: None,
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with socket.create_connection(server.address, timeout=5.0) as client:
            client.sendall(b"<Trigger>Capture</Trigger>")
            data = b""
            while b"</VisionResult>" not in data:
                chunk = client.recv(4096)
                if not chunk:
                    break
                data += chunk
        item = ET.fromstring(data.decode("ascii")).find("Item")
        check(
            "без калибровки поля -> field-calibration-missing",
            item.get("Error") == "field-calibration-missing",
            f'Error={item.get("Error")}',
        )
    finally:
        if server is not None:
            server.stop()
        path.unlink(missing_ok=True)


def main() -> int:
    print("=== Регрессионные тесты исправлений ===")
    test_polarity()
    test_polarity_forced_is_not_auto()
    test_cli_params()
    test_aspect_swapped()
    test_hsv_blur_applied()
    test_field_point_mismatch()
    test_field_matrix_valid()
    test_xml_escaping()
    test_xml_non_finite()
    test_xml_valid_parts()
    test_xml_ascii_encodable()
    test_port_released_on_source_failure()
    test_frame_source_release_idempotent()
    test_tcp_roundtrip_error_response()

    print(f"\nПройдено проверок: {PASSED}, провалено: {len(FAILED)}")
    if FAILED:
        for item in FAILED:
            print(f"  ПРОВАЛ: {item}")
        return 1
    print("Все регрессионные тесты пройдены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
