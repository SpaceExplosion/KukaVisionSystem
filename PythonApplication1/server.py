"""TCP/IP сервер координат для контроллера KUKA (KUKA.Ethernet KRL).

Раздел 4 ТЗ. Протокол обмена:
    1. Робот (EKI-клиент) подключается к серверу и отправляет команду
       <Trigger>Capture</Trigger>
    2. ПК захватывает кадр, исправляет дисторсию, детектирует детали,
       пересчитывает пиксели в миллиметры базы робота и отвечает XML:
       <VisionResult>
         <Item Count="2">
           <Part Index="1" X="450.2" Y="-120.5" Z="15.0" A="24.8" />
           <Part Index="2" X="510.0" Y="-80.3" Z="15.0" A="-12.1" />
         </Item>
       </VisionResult>
    3. Соединение остается открытым — робот может запрашивать кадры
       многократно.

Запуск:
    python server.py --camera 0 --port 59152 --part-height 15.0
"""

from __future__ import annotations

import argparse
import math
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import cv2 as cv
import numpy as np

from field_calibration import (
    FieldCalibration,
    load_field_calibration,
    transform_angle_deg,
    transform_pixel_to_robot,
)
from vision import (
    Part,
    PartDetector,
    add_vision_arguments,
    draw_parts,
    load_camera_calibration,
    normalize_angle,
    params_from_args,
)

MAX_BUFFER_BYTES = 4096
WARMUP_FRAMES = 5


@dataclass(frozen=True)
class RobotPart:
    """Деталь в координатах базы робота (мм, градусы)."""

    x_mm: float
    y_mm: float
    z_mm: float
    angle_deg: float


class FrameSource:
    """Захват кадров: индекс камеры, файл видео или одиночный снимок.

    Для одиночного снимка VideoCapture возвращает кадр один раз,
    поэтому источник открывается заново при каждом неудачном чтении —
    это же делает сервер тестируемым на статичной картинке.
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, source: int | str) -> None:
        self.source = source
        path = Path(source) if isinstance(source, str) else None
        self._static_image = (
            path is not None
            and path.exists()
            and path.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        self._capture = None if self._static_image else self._open()

    @property
    def is_static(self) -> bool:
        """True для одиночного снимка (кадр не меняется от чтения к чтению)."""
        return self._static_image

    def _open(self) -> cv.VideoCapture:
        capture = cv.VideoCapture(self.source)
        if not capture.isOpened():
            raise OSError(f"Не удалось открыть источник кадров: {self.source}")
        return capture

    def read(self) -> np.ndarray:
        if self._static_image:
            frame = cv.imread(str(self.source))
            if frame is None:
                raise RuntimeError(f"Не удалось прочитать снимок {self.source}.")
            return frame
        last_error: Exception | None = None
        for _ in range(10):
            if self._capture is not None:
                ok, frame = self._capture.read()
                if ok:
                    return frame
                self._capture.release()
                self._capture = None
            time.sleep(0.05)
            try:
                self._capture = self._open()
            except OSError as exc:
                # Камеру могли выдернуть: запоминаем причину и пробуем снова,
                # не оставляя объект в нерабочем состоянии.
                last_error = exc
        detail = f" ({last_error})" if last_error is not None else ""
        raise RuntimeError(f"Источник кадров не отдает кадры{detail}.")

    def warm_up(self, frames: int = WARMUP_FRAMES) -> None:
        if self._static_image:
            return
        for _ in range(frames):
            self.read()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def build_vision_xml(parts: list[RobotPart], error: str | None = None) -> str:
    """Формирует XML-ответ в формате KUKA.Ethernet KRL."""
    if error is not None:
        # Текст ошибки экранируется: & < > и кавычки в сообщении иначе рвут
        # XML, и парсер контроллера отбрасывает весь ответ.
        safe_error = escape(str(error), {'"': "&quot;", "'": "&apos;"})
        return (
            '<VisionResult>\n'
            f'  <Item Count="0" Error="{safe_error}"/>\n'
            "</VisionResult>\n"
        )
    # Деталь с nan/inf (вырожденная гомография) отдавать роботу нельзя:
    # KRL примет её как число и уведет инструмент в непредсказуемую точку.
    valid = [
        part for part in parts
        if all(
            math.isfinite(value)
            for value in (part.x_mm, part.y_mm, part.z_mm, part.angle_deg)
        )
    ]
    if len(valid) != len(parts):
        return build_vision_xml([], error="non-finite-coordinates")

    lines = ["<VisionResult>", f'  <Item Count="{len(valid)}">']
    for index, part in enumerate(valid, start=1):
        lines.append(
            f'    <Part Index="{index}" X="{part.x_mm:.1f}" '
            f'Y="{part.y_mm:.1f}" Z="{part.z_mm:.1f}" '
            f'A="{part.angle_deg:.1f}" />'
        )
    lines.append("  </Item>")
    lines.append("</VisionResult>\n")
    return "\n".join(lines)


def parts_to_robot(
    parts: list[Part],
    field: FieldCalibration,
    z_mm: float,
    symmetry_deg: float = 90.0,
) -> list[RobotPart]:
    """Переводит пиксельные координаты деталей в базу робота."""
    robot_parts: list[RobotPart] = []
    for part in parts:
        u, v = part.center
        x, y = transform_pixel_to_robot(field.matrix, u, v)
        angle = transform_angle_deg(field.matrix, u, v, part.angle_deg)
        robot_parts.append(
            RobotPart(x, y, z_mm, normalize_angle(angle, symmetry_deg))
        )
    return robot_parts


class VisionServer:
    """Однопоточный TCP-сервер: один контроллер KUKA в каждый момент.

    frames позволяет внедрить внешний источник кадров (например,
    общий с GUI), log — перенаправить вывод в интерфейс, а
    capture_callback(frame, parts, response) вызывается после каждого
    успешного ответа роботу (используется GUI для показа результата).
    """

    def __init__(
        self,
        detector: PartDetector,
        field: FieldCalibration | None = None,
        source: int | str = 0,
        host: str = "0.0.0.0",
        port: int = 59152,
        part_height_mm: float = 15.0,
        show_window: bool = False,
        frames: FrameSource | None = None,
        log=None,
        capture_callback=None,
    ) -> None:
        self.detector = detector
        self.field = field
        self.part_height_mm = part_height_mm
        self.show_window = show_window
        self.capture_callback = capture_callback
        self._log_fn = log or print
        self._owns_frames = frames is None
        self._lock = threading.Lock()
        self._connections: set[socket.socket] = set()
        self._stop = False
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._listener.bind((host, port))
            self._listener.listen(1)
            self._listener.settimeout(0.5)
            self.address = self._listener.getsockname()
            # Источник открывается после слушателя, но его отказ не должен
            # оставлять порт занятым «повисшим» сокетом.
            self.frames = frames if frames is not None else FrameSource(source)
            if self._owns_frames:
                self.frames.warm_up()
        except Exception:
            self._listener.close()
            raise

    def _log(self, message: str) -> None:
        self._log_fn(message)

    def process_capture(self) -> tuple[str, np.ndarray, list]:
        """Захват кадра -> детекция -> база робота -> XML."""
        with self._lock:
            frame = self.frames.read()
            parts = self.detector.detect(frame)
            if self.field is None:
                response = build_vision_xml(
                    parts=[], error="field-calibration-missing"
                )
                return response, frame, parts
            robot_parts = parts_to_robot(
                parts,
                self.field,
                self.part_height_mm,
                self.detector.params.angle_symmetry_deg,
            )
            return build_vision_xml(robot_parts), frame, parts

    def handle_connection(self, conn: socket.socket, addr) -> None:
        self._log(f"Робот подключился: {addr}")
        self._connections.add(conn)
        buffer = b""
        try:
            conn.settimeout(600.0)
            while not self._stop:
                try:
                    chunk = conn.recv(1024)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buffer = (buffer + chunk)[-MAX_BUFFER_BYTES:]
                if b"capture" not in buffer.lower():
                    continue
                buffer = b""
                try:
                    response, frame, parts = self.process_capture()
                except Exception as exc:  # детекция не должна рвать соединение
                    self._log(f"Ошибка обработки кадра: {exc}")
                    response = build_vision_xml([], error="capture-failed")
                    frame = None
                    parts = []
                # errors="replace": сообщение об ошибке может содержать
                # не-ASCII символы (путь, текст исключения) — обрыв соединения
                # из-за UnicodeEncodeError недопустим.
                conn.sendall(response.encode("ascii", errors="replace"))
                self._log("Отправлен ответ:\n" + response.rstrip())
                if self.capture_callback is not None and frame is not None:
                    try:
                        self.capture_callback(frame, parts, response)
                    except Exception as exc:
                        self._log(f"Ошибка обработчика захвата: {exc}")
                if self.show_window and frame is not None:
                    cv.imshow("Vision server", draw_parts(frame, parts))
                    cv.waitKey(1)
        except (ConnectionError, OSError) as exc:
            self._log(f"Соединение закрыто ({addr}): {exc}")
        finally:
            self._connections.discard(conn)
            conn.close()
            self._log(f"Робот отключился: {addr}")

    def serve_forever(self) -> None:
        host, port = self.address
        self._log(f"Сервер запущен: {host}:{port}. Ожидание подключения робота...")
        try:
            while not self._stop:
                try:
                    conn, addr = self._listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break  # слушатель закрыт через stop()/release()
                self.handle_connection(conn, addr)
        finally:
            self.release()

    def stop(self) -> None:
        """Останавливает цикл и закрывает слушателя и активные соединения."""
        self._stop = True
        try:
            self._listener.close()
        except OSError:
            pass
        for conn in tuple(self._connections):
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
            self._connections.discard(conn)

    def release(self) -> None:
        try:
            self._listener.close()
        except OSError:
            pass
        if self._owns_frames:
            self.frames.release()
        if self.show_window:
            cv.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TCP-сервер координат для KUKA.Ethernet KRL"
    )
    parser.add_argument("--host", default="0.0.0.0", help="адрес сервера")
    parser.add_argument("--port", type=int, default=59152,
                        help="порт сервера (по ТЗ 59152)")
    parser.add_argument("--camera", default="0",
                        help="индекс камеры либо путь к видео/снимку")
    parser.add_argument("--calibration", type=Path,
                        default=Path(__file__).with_name("camera_calibration.npz"),
                        help="файл внутренней калибровки камеры (.npz)")
    parser.add_argument("--field", type=Path,
                        default=Path(__file__).with_name("field_calibration.npz"),
                        help="файл калибровки поля (.npz)")
    parser.add_argument("--part-height", type=float, default=15.0,
                        help="высота детали Z в мм, постоянная для всех деталей")
    parser.add_argument("--show", action="store_true",
                        help="показывать окно отладки с результатом детекции")
    add_vision_arguments(parser)
    return parser.parse_args()


def source_from_text(text: str) -> int | str:
    return int(text) if text.isdigit() else text


def main() -> int:
    args = parse_args()

    camera_matrix = distortion = None
    if args.calibration.exists():
        camera_matrix, distortion = load_camera_calibration(args.calibration)
        print(f"Загружена калибровка камеры: {args.calibration}")
    else:
        print(f"Калибровка {args.calibration} не найдена — дисторсия не "
              "корректируется.")

    field = None
    if args.field.exists():
        field = load_field_calibration(args.field)
        residuals = field.residual_mm
        print(f"Загружена калибровка поля: {args.field} "
              f"(ошибка max {residuals.max():.2f} мм)")
    else:
        print(f"Калибровка поля {args.field} не найдена — без нее координаты "
              "робота выдаваться не будут.")

    detector = PartDetector(camera_matrix, distortion, params_from_args(args))
    server = VisionServer(
        source=source_from_text(args.camera),
        detector=detector,
        field=field,
        host=args.host,
        port=args.port,
        part_height_mm=args.part_height,
        show_window=args.show,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка сервера.")
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
