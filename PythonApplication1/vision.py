"""Vision pipeline: detection of cube-like parts and their orientation.

Pipeline per frame (раздел 2 ТЗ):
    1. Исправление дисторсии по внутренней калибровке (camera_calibration.npz).
    2. Предобработка: градации серого или HSV, фильтрация, бинаризация.
    3. Морфологическое закрытие.
    4. Поиск контуров и фильтрация по площади и «квадратности».
    5. cv2.minAreaRect -> центр (u, v) и угол в диапазоне [-45, +45] град.

Запуск живого просмотра для настройки параметров:
    python vision.py --camera 0 --calibration camera_calibration.npz
    python vision.py --image frame.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class Part:
    """Обнаруженная деталь в пиксельных координатах исправленного кадра."""

    center: tuple[float, float]  # (u, v) в пикселях
    angle_deg: float             # угол поворота в пределах симметрии захвата
    area_px: float               # площадь контура в пикселях
    box: np.ndarray              # 4x2 вершины minAreaRect


@dataclass
class VisionParams:
    """Настраиваемые параметры пайплайна."""

    min_area_px: float = 400.0    # отсечение мусора по минимальной площади
    max_area_px: float = 0.0      # 0 = без ограничения сверху
    aspect_min: float = 0.55      # min(w, h) / max(w, h) прямоугольника
    aspect_max: float = 1.0       # 1.0 = строго квадратные объекты
    fill_min: float = 0.55        # заливка: площадь контура / площадь прямоугольника
    angle_symmetry_deg: float = 90.0  # симметрия кубика и двухпальцевого захвата
    use_hsv: bool = False         # цветная сегментация вместо яркостной
    hsv_low: tuple[int, int, int] = (0, 60, 60)
    hsv_high: tuple[int, int, int] = (179, 255, 255)
    blur_ksize: int = 5           # сглаживание, нечетное >= 3 (1 = выключено)
    close_ksize: int = 5          # морфологическое закрытие (1 = выключено)
    adaptive: bool = False        # адаптивный порог вместо порога Оцу
    polarity: str = "auto"        # auto | dark | bright — что считать фоном


def load_camera_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Загружает матрицу камеры и коэффициенты дисторсии из .npz."""
    with np.load(path) as data:
        return np.asarray(data["camera_matrix"]), np.asarray(
            data["distortion_coefficients"]
        )


def undistort_frame(
    frame: np.ndarray,
    camera_matrix: np.ndarray | None,
    distortion: np.ndarray | None,
) -> np.ndarray:
    """Устраняет оптические искажения кадра (cv2.undistort)."""
    if camera_matrix is None or distortion is None:
        return frame
    return cv.undistort(frame, camera_matrix, distortion)


def normalize_angle(angle_deg: float, symmetry_deg: float = 90.0) -> float:
    """Приводит угол к диапазону [-sym/2, +sym/2] с учетом симметрии детали.

    Для квадратного кубика и двухпальцевого захвата симметрия 90 град.,
    результат лежит в [-45, +45].
    """
    half = abs(symmetry_deg) / 2.0
    if half < 1e-9:
        return float(angle_deg)
    return float((angle_deg + half) % symmetry_deg - half)


def _preprocess(frame: np.ndarray, params: VisionParams) -> np.ndarray:
    """Предобработка кадра: фильтрация, бинаризация, морфология."""
    blur_k = max(1, params.blur_ksize) | 1  # нечетное
    if params.use_hsv:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        low = np.asarray(params.hsv_low, dtype=np.uint8)
        high = np.asarray(params.hsv_high, dtype=np.uint8)
        binary = cv.inRange(hsv, low, high)
    else:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if blur_k >= 3:
            gray = cv.GaussianBlur(gray, (blur_k, blur_k), 0)
        if params.adaptive:
            binary = cv.adaptiveThreshold(
                gray,
                255,
                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY,
                31,
                10,
            )
        else:
            _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        foreground = np.count_nonzero(binary)
        is_minority = foreground < binary.size / 2
        invert = (
            params.polarity == "bright"
            if params.polarity in ("dark", "bright")
            else not is_minority  # auto: объекты занимают меньшую часть кадра
        )
        if invert:
            binary = cv.bitwise_not(binary)

    if params.close_ksize >= 3:
        kernel = cv.getStructuringElement(
            cv.MORPH_RECT, (params.close_ksize, params.close_ksize)
        )
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=2)
    return binary


def detect_parts(frame: np.ndarray, params: VisionParams) -> list[Part]:
    """Ищет детали на (исправленном) кадре и возвращает список Part."""
    binary = _preprocess(frame, params)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    parts: list[Part] = []
    for contour in contours:
        area = float(cv.contourArea(contour))
        if area < params.min_area_px:
            continue
        if params.max_area_px > 0 and area > params.max_area_px:
            continue

        (cx, cy), (width, height), angle = cv.minAreaRect(contour)
        long_side = max(width, height)
        short_side = min(width, height)
        if long_side <= 0:
            continue

        aspect = short_side / long_side
        if not (params.aspect_min <= aspect <= params.aspect_max):
            continue

        fill = area / (width * height) if width * height > 0 else 0.0
        if fill < params.fill_min:
            continue

        parts.append(
            Part(
                center=(float(cx), float(cy)),
                angle_deg=normalize_angle(angle, params.angle_symmetry_deg),
                area_px=area,
                box=cv.boxPoints(((cx, cy), (width, height), angle)),
            )
        )

    # Стабильный порядок: сверху вниз, слева направо.
    parts.sort(key=lambda p: (round(p.center[1], -1), p.center[0]))
    return parts


def draw_parts(frame: np.ndarray, parts: list[Part]) -> np.ndarray:
    """Рисует рамки, центры и углы деталей поверх копии кадра."""
    display = frame.copy()
    for index, part in enumerate(parts, start=1):
        box = part.box.astype(np.int32)
        cv.polylines(display, [box], True, (50, 220, 50), 2, cv.LINE_AA)
        u, v = (int(part.center[0]), int(part.center[1]))
        cv.circle(display, (u, v), 4, (0, 0, 255), -1, cv.LINE_AA)
        cv.putText(
            display,
            f"{index}: {part.angle_deg:+.1f} deg",
            (u + 8, v - 8),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv.LINE_AA,
        )
        cv.putText(
            display,
            f"{index}: {part.angle_deg:+.1f} deg",
            (u + 8, v - 8),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 220, 50),
            1,
            cv.LINE_AA,
        )
    return display


class PartDetector:
    """Полная цепочка: дисторсия -> предобработка -> контуры -> minAreaRect."""

    def __init__(
        self,
        camera_matrix: np.ndarray | None = None,
        distortion: np.ndarray | None = None,
        params: VisionParams | None = None,
    ) -> None:
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.params = params or VisionParams()

    def detect(self, frame: np.ndarray) -> list[Part]:
        corrected = undistort_frame(frame, self.camera_matrix, self.distortion)
        return detect_parts(corrected, self.params)


def parse_hsv(text: str) -> tuple[int, int, int]:
    values = [int(v.strip()) for v in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("ожидается три числа через запятую: H,S,V")
    return (values[0], values[1], values[2])


def add_vision_arguments(parser: argparse.ArgumentParser) -> None:
    """Общие параметры пайплайна, используются также в server.py."""
    parser.add_argument("--min-area", type=float, default=400.0,
                        help="минимальная площадь детали в пикселях")
    parser.add_argument("--max-area", type=float, default=0.0,
                        help="максимальная площадь в пикселях (0 = без лимита)")
    parser.add_argument("--aspect-min", type=float, default=0.55,
                        help="минимальное отношение сторон minAreaRect")
    parser.add_argument("--fill-min", type=float, default=0.55,
                        help="минимальная заливка контуром прямоугольника")
    parser.add_argument("--hsv-low", type=parse_hsv, default=None, metavar="H,S,V",
                        help="нижняя граница HSV, включает цветную сегментацию")
    parser.add_argument("--hsv-high", type=parse_hsv, default=None, metavar="H,S,V",
                        help="верхняя граница HSV")
    parser.add_argument("--blur", type=int, default=5, help="ядро GaussianBlur")
    parser.add_argument("--close", type=int, default=5, help="ядро морфозакрытия")
    parser.add_argument("--adaptive", action="store_true",
                        help="адаптивный порог вместо Оцу")
    parser.add_argument("--polarity", choices=("auto", "dark", "bright"),
                        default="auto", help="полярность бинаризации")
    parser.add_argument("--symmetry", type=float, default=90.0,
                        help="симметрия захвата в градусах (90 -> диапазон +-45)")


def params_from_args(args: argparse.Namespace) -> VisionParams:
    return VisionParams(
        min_area_px=args.min_area,
        max_area_px=args.max_area,
        aspect_min=args.aspect_min,
        fill_min=args.fill_min,
        use_hsv=args.hsv_low is not None,
        hsv_low=args.hsv_low or (0, 60, 60),
        hsv_high=args.hsv_high or (179, 255, 255),
        blur_ksize=args.blur,
        close_ksize=args.close,
        adaptive=args.adaptive,
        polarity=args.polarity,
        angle_symmetry_deg=args.symmetry,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="номер камеры")
    source.add_argument("--image", type=Path, help="обработать один снимок")
    parser.add_argument("--calibration", type=Path,
                        default=Path(__file__).with_name("camera_calibration.npz"),
                        help="файл внутренней калибровки камеры (.npz)")
    add_vision_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    camera_matrix = distortion = None
    if args.calibration.exists():
        camera_matrix, distortion = load_camera_calibration(args.calibration)
        print(f"Загружена калибровка камеры: {args.calibration}")
    else:
        print(f"Калибровка {args.calibration} не найдена — дисторсия не корректируется.")

    detector = PartDetector(camera_matrix, distortion, params_from_args(args))

    if args.image is not None:
        frame = cv.imread(str(args.image))
        if frame is None:
            print(f"Не удалось прочитать изображение {args.image}.")
            return 1
        parts = detector.detect(frame)
        for index, part in enumerate(parts, start=1):
            print(f"Деталь {index}: центр ({part.center[0]:.1f}, "
                  f"{part.center[1]:.1f}) px, угол {part.angle_deg:+.1f} град, "
                  f"площадь {part.area_px:.0f} px")
        cv.imshow("Vision detection", draw_parts(frame, parts))
        cv.waitKey(0)
        cv.destroyAllWindows()
        return 0

    camera = cv.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Не удалось открыть камеру {args.camera}.")
        return 1

    print("Просмотр детекции: Q/ESC — выход.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Не удалось получить кадр с камеры.")
                break
            parts = detector.detect(frame)
            display = draw_parts(frame, parts)
            cv.putText(display, f"Parts: {len(parts)}", (12, 28),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(display, f"Parts: {len(parts)}", (12, 28),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2, cv.LINE_AA)
            cv.imshow("Vision detection", display)
            if cv.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
