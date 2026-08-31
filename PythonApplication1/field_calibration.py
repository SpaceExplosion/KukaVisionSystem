"""Калибровка поля: преобразование пикселей в миллиметры базы робота.

Раздел 3 ТЗ. Метод:
    1. На рабочем столе робота размечаются минимум 4 реперные точки.
    2. Острием инструмента робота физически касаются каждой точки
       и считывают координаты (X, Y) в базе робота с пульта.
    3. Камерой фиксируются координаты тех же точек в пикселях (u, v):
       в этом инструменте — кликами мыши по живому (исправленному) кадру.
    4. Рассчитывается матрица проективного преобразования
       (cv2.findHomography / cv2.estimateAffine2D) и сохраняется в файл.

Запуск:
    python field_calibration.py --robot 350.2,-120.4 --robot 520.1,-118.9 \
        --robot 515.0,180.2 --robot 345.8,175.6
    Если координаты робота не переданы аргументами, они запрашиваются
    в консоли после каждого клика.

Управление в окне:
    ЛКМ   добавить точку (в том же порядке, что и координаты робота)
    D     удалить последнюю точку
    V     режим проверки: координаты мыши пересчитываются в мм
    C     рассчитать и сохранить калибровку (нужно >= 4 точек)
    Q/ESC выход
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from vision import load_camera_calibration, undistort_frame


@dataclass(frozen=True)
class FieldCalibration:
    """Матрица преобразования пиксель -> миллиметры базы робота."""

    matrix: np.ndarray        # 3x3, применяется к однородным пиксельным координатам
    pixel_points: np.ndarray  # Nx2 реперные точки в пикселях
    robot_points: np.ndarray  # Nx2 реперные точки в мм базы робота
    residual_mm: np.ndarray   # Nx1 погрешность аппроксимации в каждой точке


def compute_field_matrix(
    pixel_points: np.ndarray,
    robot_points: np.ndarray,
    method: str = "homography",
) -> np.ndarray:
    """Строит матрицу 3x3, отображающую (u, v) -> (X, Y) базы робота."""
    pixels = np.asarray(pixel_points, dtype=np.float64).reshape(-1, 2)
    robots = np.asarray(robot_points, dtype=np.float64).reshape(-1, 2)
    if len(pixels) != len(robots) or len(pixels) < 3:
        raise ValueError("Нужно минимум 3 пары точек (по ТЗ — 4).")

    if method == "affine":
        if len(pixels) == 3:
            affine = cv.getAffineTransform(
                pixels.astype(np.float32), robots.astype(np.float32)
            )
            matrix = np.vstack([affine, [0.0, 0.0, 1.0]])
        else:
            affine, _ = cv.estimateAffine2D(pixels, robots, method=cv.LMEDS)
            if affine is None:
                raise RuntimeError("Не удалось оценить аффинное преобразование.")
            matrix = np.vstack([affine, [0.0, 0.0, 1.0]])
    else:
        matrix, _ = cv.findHomography(pixels, robots, method=0)
        if matrix is None:
            raise RuntimeError("Не удалось рассчитать гомографию.")
    return matrix


def transform_pixel_to_robot(
    matrix: np.ndarray, u: float, v: float
) -> tuple[float, float]:
    """Переводит пиксельные координаты в миллиметры базы робота."""
    point = np.array([[[u, v]]], dtype=np.float64)
    mapped = cv.perspectiveTransform(point, matrix)[0, 0]
    return float(mapped[0]), float(mapped[1])


def transform_angle_deg(
    matrix: np.ndarray, u: float, v: float, angle_deg: float
) -> float:
    """Пересчитывает угол поворота детали в систему координат базы.

    Направление оси детали в пикселях поворачивается линейной частью
    матрицы преобразования, поэтому корректно учитывается разворот
    камеры относительно базы робота.
    """
    direction = np.array(
        [np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))],
        dtype=np.float64,
    )
    base = cv.perspectiveTransform(
        np.array([[[u, v]]], dtype=np.float64), matrix
    )[0, 0]
    tip = cv.perspectiveTransform(
        np.array([[[u + direction[0], v + direction[1]]]], dtype=np.float64),
        matrix,
    )[0, 0]
    robot_vector = tip - base
    return float(np.degrees(np.arctan2(robot_vector[1], robot_vector[0])))


def calibration_residuals(
    matrix: np.ndarray, pixel_points: np.ndarray, robot_points: np.ndarray
) -> np.ndarray:
    """Погрешность аппроксимации каждой реперной точки в мм."""
    mapped = cv.perspectiveTransform(
        np.asarray(pixel_points, dtype=np.float64).reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    return np.linalg.norm(mapped - np.asarray(robot_points), axis=1)


def save_field_calibration(path: Path, calibration: FieldCalibration) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        matrix=calibration.matrix,
        pixel_points=calibration.pixel_points,
        robot_points=calibration.robot_points,
    )
    print(f"Калибровка поля сохранена: {path}")


def load_field_calibration(path: Path) -> FieldCalibration:
    with np.load(path) as data:
        matrix = np.asarray(data["matrix"])
        pixel_points = np.asarray(data["pixel_points"])
        robot_points = np.asarray(data["robot_points"])
    return FieldCalibration(
        matrix=matrix,
        pixel_points=pixel_points,
        robot_points=robot_points,
        residual_mm=calibration_residuals(matrix, pixel_points, robot_points),
    )


def parse_point(text: str) -> tuple[float, float]:
    values = [float(v.strip().replace(",", ".")) for v in text.replace(";", ",").split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("ожидается пара чисел через запятую: X,Y")
    return (values[0], values[1])


def draw_overlay(
    frame: np.ndarray,
    pixel_points: np.ndarray,
    robot_points: list[tuple[float, float]],
    status: str,
    verify: bool,
    matrix: np.ndarray | None,
    mouse_xy: tuple[int, int] | None,
) -> np.ndarray:
    display = frame.copy()

    for index, (u, v) in enumerate(np.asarray(pixel_points).reshape(-1, 2)):
        position = (int(u), int(v))
        cv.circle(display, position, 6, (0, 0, 255), -1, cv.LINE_AA)
        label = str(index + 1)
        if index < len(robot_points):
            label += f": {robot_points[index][0]:.1f}, {robot_points[index][1]:.1f}"
        cv.putText(display, label, (position[0] + 10, position[1] - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(display, label, (position[0] + 10, position[1] - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv.LINE_AA)

    if verify and matrix is not None and mouse_xy is not None:
        x, y = transform_pixel_to_robot(matrix, mouse_xy[0], mouse_xy[1])
        cv.putText(display, f"Robot: X={x:.1f}  Y={y:.1f} mm", (12, 28),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(display, f"Robot: X={x:.1f}  Y={y:.1f} mm", (12, 28),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 60), 2, cv.LINE_AA)

    lines = [
        status,
        "LMB add point | D undo | V verify | C calibrate | Q quit",
    ]
    for line, row in zip(lines, (52, 76)):
        cv.putText(display, line, (12, row), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(display, line, (12, row), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 1, cv.LINE_AA)
    return display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Калибровка поля пиксель->робот")
    parser.add_argument("--camera", type=int, default=0, help="номер камеры")
    parser.add_argument("--calibration", type=Path,
                        default=Path(__file__).with_name("camera_calibration.npz"),
                        help="файл внутренней калибровки камеры (.npz)")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("field_calibration.npz"),
                        help="выходной файл калибровки поля")
    parser.add_argument("--method", choices=("homography", "affine"),
                        default="homography",
                        help="тип преобразования (по ТЗ — гомография)")
    parser.add_argument("--robot", type=parse_point, action="append", metavar="X,Y",
                        help="координаты реперных точек в базе робота; "
                             "повторять по числу точек в том же порядке, "
                             "что и клики мышью")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    camera_matrix = distortion = None
    if args.calibration.exists():
        camera_matrix, distortion = load_camera_calibration(args.calibration)
        print(f"Загружена калибровка камеры: {args.calibration}")
    else:
        print(f"Калибровка {args.calibration} не найдена — кадр не исправляется.")

    camera = cv.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Не удалось открыть камеру {args.camera}.")
        return 1

    pixel_points: list[tuple[float, float]] = []
    robot_points: list[tuple[float, float]] = list(args.robot or [])
    field: FieldCalibration | None = None
    verify = False
    mouse_xy: tuple[int, int] | None = None
    status = f"Кликните точку 1 (всего нужно >= 4)"

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal mouse_xy
        mouse_xy = (x, y)
        if event != cv.EVENT_LBUTTONDOWN:
            return
        nonlocal pixel_points, robot_points, status
        index = len(pixel_points)
        if index < len(robot_points):
            robot = robot_points[index]
        else:
            raw = input(f"Введите координаты робота X,Y для точки {index + 1}: ")
            robot = parse_point(raw)
            robot_points.append(robot)
        pixel_points.append((float(x), float(y)))
        print(f"Точка {index + 1}: пиксель ({x}, {y}) -> робот "
              f"({robot[0]:.1f}, {robot[1]:.1f}) мм")
        status = f"Точек: {len(pixel_points)} (нужно >= 4, затем C)"

    window = "Field calibration"
    cv.namedWindow(window)
    cv.setMouseCallback(window, on_mouse)

    print("Калибровка поля запущена.")
    print("ЛКМ — добавить точку; D — отменить; V — проверка; "
          "C — рассчитать; Q/ESC — выход.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Не удалось получить кадр с камеры.")
                return 1
            frame = undistort_frame(frame, camera_matrix, distortion)

            display = draw_overlay(
                frame,
                np.asarray(pixel_points, dtype=np.float64).reshape(-1, 2),
                robot_points,
                status,
                verify,
                field.matrix if field is not None else None,
                mouse_xy,
            )
            cv.imshow(window, display)

            key = cv.waitKey(20) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("d"), ord("D")) and pixel_points:
                pixel_points.pop()
                status = f"Точка удалена, осталось: {len(pixel_points)}"
            elif key in (ord("v"), ord("V")):
                if field is None:
                    status = "Сначала рассчитайте калибровку (C)"
                else:
                    verify = not verify
                    status = f"Режим проверки {'вкл' if verify else 'выкл'}"
            elif key in (ord("c"), ord("C")):
                if len(pixel_points) < 4:
                    status = f"Нужно минимум 4 точки, есть {len(pixel_points)}"
                    continue
                if len(pixel_points) > len(robot_points):
                    status = "Количество кликов и координат робота не совпадает"
                    continue
                matrix = compute_field_matrix(pixel_points, robot_points,
                                              args.method)
                field = FieldCalibration(
                    matrix=matrix,
                    pixel_points=np.asarray(pixel_points),
                    robot_points=np.asarray(robot_points),
                    residual_mm=calibration_residuals(matrix, pixel_points,
                                                      robot_points),
                )
                save_field_calibration(args.output, field)
                residuals = field.residual_mm
                print(f"Тип преобразования: {args.method}")
                print(f"Погрешность: средняя {residuals.mean():.2f} мм, "
                      f"максимальная {residuals.max():.2f} мм")
                for index, error in enumerate(residuals, start=1):
                    print(f"  точка {index}: {error:.2f} мм")
                status = (f"Сохранено. Ошибка max {residuals.max():.2f} mm "
                          f"(V — проверка)")
    finally:
        camera.release()
        cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
