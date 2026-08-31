"""Interactive camera calibration for a 12 x 18 ArUco GridBoard.

Board parameters:
    dictionary:        DICT_6X6_250
    columns x rows:    12 x 18 markers
    marker size:       8 mm
    marker separation: 6 mm
    first marker ID:   0

Controls:
    SPACE  save the current board view
    C      calculate and save calibration
    U      enable/disable the undistorted preview
    R      remove all collected views
    Q/ESC  exit
"""

from __future__ import annotations


import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

# Parameters of the printed board.
BOARD_COLUMNS = 12
BOARD_ROWS = 18
MARKER_SIZE_MM = 8.0
MARKER_SEPARATION_MM = 6.0
DICTIONARY_NAME = "DICT_6X6_250"
START_ID = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a camera using an ArUco GridBoard"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="camera number (default: 0)"
    )
    parser.add_argument(
        "--samples", type=int, default=15, help="minimum number of board views"
    )
    parser.add_argument(
        "--min-markers",
        type=int,
        default=6,
        help="minimum visible markers required to save a view",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("camera_calibration"),
        help="output path without an extension",
    )
    return parser.parse_args()


def require_aruco() -> object:
    if not hasattr(cv, "aruco"):
        raise RuntimeError(
            "В установленной сборке OpenCV нет модуля aruco. "
            "Установите пакет opencv-contrib-python."
        )
    return cv.aruco


def create_board(aruco: object, dictionary: object) -> object:
    board_size = (BOARD_COLUMNS, BOARD_ROWS)

    # OpenCV 4.7+ API.
    if hasattr(aruco, "GridBoard"):
        return aruco.GridBoard(
            board_size,
            MARKER_SIZE_MM,
            MARKER_SEPARATION_MM,
            dictionary,
        )

    # Compatibility with older OpenCV versions.
    if hasattr(aruco, "GridBoard_create"):
        return aruco.GridBoard_create(
            BOARD_COLUMNS,
            BOARD_ROWS,
            MARKER_SIZE_MM,
            MARKER_SEPARATION_MM,
            dictionary,
            START_ID,
        )

    raise RuntimeError("Эта версия OpenCV не поддерживает ArUco GridBoard.")


def create_detector(aruco: object, dictionary: object) -> tuple[object | None, object]:
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    if hasattr(aruco, "CORNER_REFINE_SUBPIX") and hasattr(
        parameters, "cornerRefinementMethod"
    ):
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary, parameters), parameters
    return None, parameters


def detect_markers(
    aruco: object,
    detector: object | None,
    parameters: object,
    dictionary: object,
    gray: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return list(corners), ids


def board_point_map(board: object) -> dict[int, np.ndarray]:
    if hasattr(board, "getIds"):
        board_ids = np.asarray(board.getIds()).reshape(-1)
        board_points = board.getObjPoints()
    else:
        board_ids = np.asarray(board.ids).reshape(-1)
        board_points = board.objPoints

    return {
        int(marker_id): np.asarray(marker_points, dtype=np.float32).reshape(4, 3)
        for marker_id, marker_points in zip(board_ids, board_points)
    }


def select_board_markers(
    corners: list[np.ndarray],
    ids: np.ndarray | None,
    point_map: dict[int, np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray | None]:
    if ids is None:
        return [], None

    selected_corners: list[np.ndarray] = []
    selected_ids: list[int] = []
    for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
        marker_id = int(marker_id)
        if marker_id in point_map:
            selected_corners.append(marker_corners)
            selected_ids.append(marker_id)

    if not selected_ids:
        return [], None
    return selected_corners, np.asarray(selected_ids, dtype=np.int32).reshape(-1, 1)


def make_calibration_points(
    corners: list[np.ndarray],
    ids: np.ndarray,
    point_map: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    object_points = []
    image_points = []
    for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
        object_points.append(point_map[int(marker_id)])
        image_points.append(np.asarray(marker_corners, dtype=np.float32).reshape(4, 2))

    return (
        np.concatenate(object_points).astype(np.float32),
        np.concatenate(image_points).astype(np.float32),
    )


def mean_reprojection_error(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rotation_vectors: tuple[np.ndarray, ...],
    translation_vectors: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    total_squared_error = 0.0
    total_points = 0
    for object_set, image_set, rvec, tvec in zip(
        object_points, image_points, rotation_vectors, translation_vectors
    ):
        projected, _ = cv.projectPoints(
            object_set, rvec, tvec, camera_matrix, distortion
        )
        projected = projected.reshape(-1, 2)
        measured = image_set.reshape(-1, 2)
        total_squared_error += cv.norm(measured, projected, cv.NORM_L2) ** 2
        total_points += len(object_set)
    return float(np.sqrt(total_squared_error / total_points))


def save_calibration(
    output: Path,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    reprojection_error: float,
) -> tuple[Path, Path]:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output.with_suffix(".npz")
    yaml_path = output.with_suffix(".yaml")

    metadata = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "reprojection_error": reprojection_error,
        "board_columns": BOARD_COLUMNS,
        "board_rows": BOARD_ROWS,
        "marker_size_mm": MARKER_SIZE_MM,
        "marker_separation_mm": MARKER_SEPARATION_MM,
        "start_id": START_ID,
    }
    np.savez(
        npz_path,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        dictionary=DICTIONARY_NAME,
        **metadata,
    )

    storage = cv.FileStorage(str(yaml_path), cv.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise OSError(f"Cannot create {yaml_path}")
    storage.write("camera_matrix", camera_matrix)
    storage.write("distortion_coefficients", distortion)
    storage.write("dictionary", DICTIONARY_NAME)
    for name, value in metadata.items():
        storage.write(name, value)
    storage.release()
    return npz_path, yaml_path


def draw_text(
    frame: np.ndarray, text: str, line: int, color: tuple[int, int, int]
) -> None:
    position = (12, 28 + line * 28)
    cv.putText(
        frame, text, position, cv.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv.LINE_AA
    )
    cv.putText(
        frame, text, position, cv.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv.LINE_AA
    )


def main() -> int:
    args = parse_args()
    if args.samples < 3:
        raise ValueError("Для калибровки требуется минимум 3 ракурса.")
    if args.min_markers < 1:
        raise ValueError("Количество видимых маркеров должно быть положительным.")

    aruco = require_aruco()
    dictionary_id = getattr(aruco, DICTIONARY_NAME, None)
    if dictionary_id is None:
        raise RuntimeError(f"OpenCV не поддерживает словарь {DICTIONARY_NAME}.")

    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    board = create_board(aruco, dictionary)
    detector, detector_parameters = create_detector(aruco, dictionary)
    point_map = board_point_map(board)

    expected_ids = set(range(START_ID, START_ID + BOARD_COLUMNS * BOARD_ROWS))
    if set(point_map) != expected_ids:
        raise RuntimeError(
            "Идентификаторы созданной GridBoard не соответствуют диапазону 0...215."
        )

    captured_object_points: list[np.ndarray] = []
    captured_image_points: list[np.ndarray] = []
    camera_matrix: np.ndarray | None = None
    distortion: np.ndarray | None = None
    use_undistortion = False
    status = "Show the ArUco board to the camera"

    camera = cv.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Не удалось открыть камеру {args.camera}.")
        print(
            "Закройте другие программы, использующие камеру, или выберите другую: --camera 1"
        )
        return 1

    print("Калибровка камеры по ArUco GridBoard запущена.")
    print(
        f"Доска: {BOARD_COLUMNS} x {BOARD_ROWS}, маркер {MARKER_SIZE_MM:g} мм, "
        f"промежуток {MARKER_SEPARATION_MM:g} мм, {DICTIONARY_NAME}, ID {START_ID}...215"
    )
    print(
        "SPACE — добавить ракурс; C — рассчитать; U — коррекция; R — сброс; Q/ESC — выход"
    )

    image_size: tuple[int, int] | None = None
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Не удалось получить кадр с камеры.")
                return 1

            image_size = (frame.shape[1], frame.shape[0])
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            detected_corners, detected_ids = detect_markers(
                aruco, detector, detector_parameters, dictionary, gray
            )
            board_corners, board_ids = select_board_markers(
                detected_corners, detected_ids, point_map
            )
            marker_count = 0 if board_ids is None else len(board_ids)

            if board_ids is not None:
                aruco.drawDetectedMarkers(frame, board_corners, board_ids)

            display = frame
            if (
                use_undistortion
                and camera_matrix is not None
                and distortion is not None
            ):
                display = cv.undistort(frame, camera_matrix, distortion)

            enough_markers = marker_count >= args.min_markers
            marker_color = (50, 220, 50) if enough_markers else (40, 180, 255)
            draw_text(
                display,
                f"Board markers: {marker_count} (need {args.min_markers})",
                0,
                marker_color,
            )
            draw_text(
                display,
                f"Saved views: {len(captured_image_points)}/{args.samples}",
                1,
                (255, 255, 255),
            )
            draw_text(display, status, 2, (255, 255, 255))
            draw_text(
                display,
                "SPACE capture | C calibrate | U undistort | R reset | Q exit",
                3,
                (220, 220, 220),
            )
            cv.imshow("ArUco camera calibration", display)

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                if board_ids is None or not enough_markers:
                    status = f"Need at least {args.min_markers} board markers"
                    print(
                        f"Нужно видеть минимум {args.min_markers} маркеров — ракурс не добавлен."
                    )
                else:
                    object_set, image_set = make_calibration_points(
                        board_corners, board_ids, point_map
                    )
                    captured_object_points.append(object_set)
                    captured_image_points.append(image_set)
                    status = f"Captured view {len(captured_image_points)}"
                    print(
                        f"Добавлен ракурс {len(captured_image_points)}/{args.samples} "
                        f"({marker_count} маркеров)"
                    )
            elif key in (ord("r"), ord("R")):
                captured_object_points.clear()
                captured_image_points.clear()
                camera_matrix = None
                distortion = None
                use_undistortion = False
                status = "All views removed"
                print("Все сохранённые ракурсы удалены.")
            elif key in (ord("u"), ord("U")):
                if camera_matrix is None:
                    status = "Calibrate first by pressing C"
                else:
                    use_undistortion = not use_undistortion
                    status = f"Undistortion {'ON' if use_undistortion else 'OFF'}"
            elif key in (ord("c"), ord("C")):
                if len(captured_image_points) < args.samples:
                    missing = args.samples - len(captured_image_points)
                    status = f"Need {missing} more views"
                    print(f"Нужно добавить ещё {missing} ракурсов.")
                    continue

                rms, camera_matrix, distortion, rvecs, tvecs = cv.calibrateCamera(
                    captured_object_points,
                    captured_image_points,
                    image_size,
                    None,
                    None,
                )
                reprojection_error = mean_reprojection_error(
                    captured_object_points,
                    captured_image_points,
                    rvecs,
                    tvecs,
                    camera_matrix,
                    distortion,
                )
                npz_path, yaml_path = save_calibration(
                    args.output,
                    camera_matrix,
                    distortion,
                    image_size,
                    reprojection_error,
                )
                use_undistortion = True
                status = f"Saved, error: {reprojection_error:.3f} px"
                print("\nКалибровка завершена.")
                print(f"RMS OpenCV: {rms:.6f}")
                print(f"Средняя ошибка репроекции: {reprojection_error:.4f} px")
                print("Матрица камеры:\n", camera_matrix)
                print("Коэффициенты дисторсии:\n", distortion)
                print(f"Сохранено: {npz_path}")
                print(f"Сохранено: {yaml_path}")
    finally:
        camera.release()
        cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
