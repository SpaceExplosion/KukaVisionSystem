# Многострочная строка документации (docstring), описывающая назначение скрипта и параметры доски.
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

# Включаем поддержку новых аннотаций типов из будущих версий Python (полезно для старых версий Python 3)
from __future__ import annotations

# Импортируем модуль для обработки аргументов командной строки, переданных при запуске скрипта
import argparse
# Импортируем класс Path для удобной кроссплатформенной работы с путями к файлам и папкам
from pathlib import Path

# Импортируем библиотеку OpenCV (компьютерное зрение) и задаем ей короткое имя 'cv'
import cv2 as cv
# Импортируем библиотеку NumPy для математических вычислений и работы с матрицами/массивами, короткое имя 'np'
import numpy as np


# Задаем константу: количество столбцов (маркеров) на распечатанной калибровочной доске
BOARD_COLUMNS = 12
# Задаем константу: количество строк на распечатанной калибровочной доске
BOARD_ROWS = 18
# Задаем физический размер стороны одного маркера в миллиметрах
MARKER_SIZE_MM = 8.0
# Задаем физическое расстояние (промежуток) между маркерами в миллиметрах
MARKER_SEPARATION_MM = 6.0
# Указываем название используемого словаря ArUco (размер 6x6, 250 уникальных маркеров)
DICTIONARY_NAME = "DICT_6X6_250"
# Указываем идентификатор (ID), с которого начинается нумерация маркеров на доске
START_ID = 0


# Определяем функцию для парсинга (разбора) аргументов командной строки
def parse_args() -> argparse.Namespace:
    # Создаем объект парсера и задаем описание программы для справки (--help)
    parser = argparse.ArgumentParser(description="Calibrate a camera using an ArUco GridBoard")
    # Добавляем аргумент --camera для выбора индекса камеры (по умолчанию 0 — основная камера)
    parser.add_argument("--camera", type=int, default=0, help="camera number (default: 0)")
    # Добавляем аргумент --samples: сколько минимум кадров (ракурсов) нужно для калибровки
    parser.add_argument("--samples", type=int, default=15, help="minimum number of board views")
    # Добавляем аргумент --min-markers: минимальное количество видимых маркеров в кадре для его сохранения
    parser.add_argument(
        "--min-markers",
        type=int,
        default=6,
        help="minimum visible markers required to save a view",
    )
    # Добавляем аргумент --output: путь и имя файла (без расширения) для сохранения результатов калибровки
    parser.add_argument(
        "--output",
        type=Path,
        # По умолчанию сохраняем рядом со скриптом с именем "camera_calibration"
        default=Path(__file__).with_name("camera_calibration"),
        help="output path without an extension",
    )
    # Парсим переданные аргументы и возвращаем их в виде объекта
    return parser.parse_args()


# Функция для проверки наличия модуля ArUco в установленной версии OpenCV
def require_aruco() -> object:
    # Если в модуле cv нет атрибута "aruco"...
    if not hasattr(cv, "aruco"):
        # Вызываем ошибку с подробным описанием, как её исправить (нужен opencv-contrib-python)
        raise RuntimeError(
            "В установленной сборке OpenCV нет модуля aruco. "
            "Установите пакет opencv-contrib-python."
        )
    # Если модуль есть, возвращаем его для дальнейшего использования
    return cv.aruco


# Функция для создания объекта калибровочной доски (GridBoard)
def create_board(aruco: object, dictionary: object) -> object:
    # Группируем столбцы и строки в один кортеж (размер доски)
    board_size = (BOARD_COLUMNS, BOARD_ROWS)

    # Проверяем, используется ли новый API (OpenCV версии 4.7 и выше)
    if hasattr(aruco, "GridBoard"):
        # Возвращаем созданный объект GridBoard новым способом
        return aruco.GridBoard(
            board_size,
            MARKER_SIZE_MM,
            MARKER_SEPARATION_MM,
            dictionary,
        )

    # Если нового API нет, проверяем наличие старого метода создания доски
    if hasattr(aruco, "GridBoard_create"):
        # Возвращаем объект GridBoard, созданный с помощью старого метода
        return aruco.GridBoard_create(
            BOARD_COLUMNS,
            BOARD_ROWS,
            MARKER_SIZE_MM,
            MARKER_SEPARATION_MM,
            dictionary,
            START_ID,
        )

    # Если ни один из методов не найден, выбрасываем ошибку о несовместимости версий
    raise RuntimeError("Эта версия OpenCV не поддерживает ArUco GridBoard.")


# Функция для настройки и создания детектора маркеров ArUco
def create_detector(aruco: object, dictionary: object) -> tuple[object | None, object]:
    # Если есть новый класс параметров, создаем объект параметров
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    # Иначе используем старый метод создания параметров
    else:
        parameters = aruco.DetectorParameters_create()

    # Включаем субпиксельное уточнение углов для более точной калибровки (если поддерживается)
    if hasattr(aruco, "CORNER_REFINE_SUBPIX") and hasattr(
        parameters, "cornerRefinementMethod"
    ):
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    # Если доступен новый класс ArucoDetector (OpenCV 4.7+), создаем и возвращаем его вместе с параметрами
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary, parameters), parameters
    # В старых версиях детектор как отдельный объект не нужен, возвращаем None и параметры
    return None, parameters


# Функция для поиска (детекции) маркеров на черно-белом изображении
def detect_markers(
    aruco: object,
    detector: object | None,
    parameters: object,
    dictionary: object,
    gray: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    # Если есть новый объект детектора, используем его метод detectMarkers
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    # Если детектора нет (старая версия), вызываем функцию detectMarkers напрямую из модуля aruco
    else:
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    # Возвращаем список найденных углов и массив ID найденных маркеров
    return list(corners), ids


# Функция для создания словаря, связывающего ID маркера с его физическими 3D-координатами на доске
def board_point_map(board: object) -> dict[int, np.ndarray]:
    # Проверяем, используются ли старые getter-методы для получения ID
    if hasattr(board, "getIds"):
        # Извлекаем ID и превращаем их в одномерный массив
        board_ids = np.asarray(board.getIds()).reshape(-1)
        # Извлекаем соответствующие 3D-координаты углов маркеров в физическом пространстве
        board_points = board.getObjPoints()
    # Если getter'ов нет, напрямую обращаемся к свойствам объекта (новый API)
    else:
        board_ids = np.asarray(board.ids).reshape(-1)
        board_points = board.objPoints

    # Создаем и возвращаем словарь (dict comprehension), где ключ - это ID маркера, 
    # а значение - матрица 4x3 (4 угла, 3 координаты: X, Y, Z) типа float32
    return {
        int(marker_id): np.asarray(marker_points, dtype=np.float32).reshape(4, 3)
        for marker_id, marker_points in zip(board_ids, board_points)
    }


# Функция для отсеивания тех найденных маркеров, которые не принадлежат нашей доске
def select_board_markers(
    corners: list[np.ndarray],
    ids: np.ndarray | None,
    point_map: dict[int, np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray | None]:
    # Если на изображении не найдено ни одного маркера (ids == None), возвращаем пустоту
    if ids is None:
        return [], None

    # Создаем пустой список для углов маркеров, относящихся к доске
    selected_corners: list[np.ndarray] = []
    # Создаем пустой список для их ID
    selected_ids: list[int] = []
    # Перебираем все найденные маркеры (их углы и ID)
    for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
        # Приводим ID к целому числу
        marker_id = int(marker_id)
        # Если найденный ID есть в карте координат доски (т.е. маркер принадлежит доске)
        if marker_id in point_map:
            # Добавляем его углы в список отобранных
            selected_corners.append(marker_corners)
            # Добавляем его ID в список отобранных
            selected_ids.append(marker_id)

    # Если ни один маркер с доски не найден, возвращаем пустые значения
    if not selected_ids:
        return [], None
    # Возвращаем отфильтрованный список углов и массив ID в формате столбца (N, 1)
    return selected_corners, np.asarray(selected_ids, dtype=np.int32).reshape(-1, 1)


# Функция для подготовки пар точек (3D объект - 2D изображение) для калибровки
def make_calibration_points(
    corners: list[np.ndarray],
    ids: np.ndarray,
    point_map: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    # Список для 3D координат маркеров в физическом мире
    object_points = []
    # Список для 2D координат углов тех же маркеров на пикселях изображения
    image_points = []
    # Проходимся по всем углам и ID отфильтрованных маркеров доски
    for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
        # Получаем эталонные 3D точки маркера из карты (point_map) и добавляем в список
        object_points.append(point_map[int(marker_id)])
        # Форматируем 2D точки в матрицу 4x2 и добавляем в список
        image_points.append(np.asarray(marker_corners, dtype=np.float32).reshape(4, 2))

    # Объединяем списки массивов в единые массивы NumPy и возвращаем их (3D точки и 2D точки)
    return (
        np.concatenate(object_points).astype(np.float32),
        np.concatenate(image_points).astype(np.float32),
    )


# Функция для вычисления средней ошибки репроекции (насколько точно калибровка описывает реальность)
def mean_reprojection_error(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rotation_vectors: tuple[np.ndarray, ...],
    translation_vectors: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    # Переменная для накопления квадрата ошибки
    total_squared_error = 0.0
    # Переменная для подсчета общего количества точек
    total_points = 0
    # Цикл по всем сохраненным ракурсам (3D точки, 2D точки, векторы вращения и смещения)
    for object_set, image_set, rvec, tvec in zip(
        object_points, image_points, rotation_vectors, translation_vectors
    ):
        # Проецируем идеальные 3D точки на 2D плоскость, используя найденные параметры камеры (матрицу, дисторсию и т.д.)
        projected, _ = cv.projectPoints(object_set, rvec, tvec, camera_matrix, distortion)
        # Приводим спроецированные точки к плоскому виду списка 2D координат
        projected = projected.reshape(-1, 2)
        # Приводим измеренные (реальные пиксельные) точки к такому же виду
        measured = image_set.reshape(-1, 2)
        # Вычисляем L2-норму (расстояние) между измеренными и спроецированными точками, возводим в квадрат и суммируем
        total_squared_error += cv.norm(measured, projected, cv.NORM_L2) ** 2
        # Увеличиваем счетчик точек на количество точек в текущем ракурсе
        total_points += len(object_set)
    # Возвращаем корень из среднеквадратичной ошибки (RMSE)
    return float(np.sqrt(total_squared_error / total_points))


# Функция для сохранения результатов калибровки в файлы .npz и .yaml
def save_calibration(
    output: Path,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    reprojection_error: float,
) -> tuple[Path, Path]:
    # Преобразуем путь в абсолютный и раскрываем символ ~, если он есть (путь пользователя)
    output = output.expanduser().resolve()
    # Создаем директорию для файла, если ее нет (включая родительские)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Формируем путь для сохранения массива numpy (.npz)
    npz_path = output.with_suffix(".npz")
    # Формируем путь для сохранения файла OpenCV FileStorage (.yaml)
    yaml_path = output.with_suffix(".yaml")

    # Собираем словарь с дополнительной полезной информацией (метаданными) о калибровке
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
    # Сохраняем матрицу камеры, дисторсию и метаданные в сжатый архив numpy (.npz)
    np.savez(
        npz_path,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        dictionary=DICTIONARY_NAME,
        **metadata, # Распаковываем словарь метаданных в аргументы функции
    )

    # Открываем .yaml файл для записи встроенным инструментом OpenCV (FileStorage)
    storage = cv.FileStorage(str(yaml_path), cv.FILE_STORAGE_WRITE)
    # Проверяем, успешно ли открылся файл
    if not storage.isOpened():
        raise OSError(f"Cannot create {yaml_path}")
    # Записываем матрицу камеры
    storage.write("camera_matrix", camera_matrix)
    # Записываем коэффициенты искажений (дисторсии)
    storage.write("distortion_coefficients", distortion)
    # Записываем имя словаря ArUco
    storage.write("dictionary", DICTIONARY_NAME)
    # В цикле записываем все остальные метаданные из нашего словаря
    for name, value in metadata.items():
        storage.write(name, value)
    # Закрываем файл
    storage.release()
    # Возвращаем пути к сохраненным файлам
    return npz_path, yaml_path


# Функция-утилита для удобного вывода текста поверх видеокадра
def draw_text(frame: np.ndarray, text: str, line: int, color: tuple[int, int, int]) -> None:
    # Рассчитываем координаты текста в зависимости от номера строки (line)
    position = (12, 28 + line * 28)
    # Сначала рисуем толстый черный текст (обводку) для читаемости на любом фоне
    cv.putText(frame, text, position, cv.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv.LINE_AA)
    # Затем поверх рисуем сам текст выбранным цветом, более тонким шрифтом
    cv.putText(frame, text, position, cv.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv.LINE_AA)


# Основная функция, точка входа в программу
def main() -> int:
    # Получаем аргументы командной строки
    args = parse_args()
    # Проверяем, что запрошено минимум 3 ракурса (математический минимум для калибровки камеры)
    if args.samples < 3:
        raise ValueError("Для калибровки требуется минимум 3 ракурса.")
    # Проверяем, что минимальное количество маркеров для захвата > 0
    if args.min_markers < 1:
        raise ValueError("Количество видимых маркеров должно быть положительным.")

    # Получаем модуль ArUco (выбрасывает ошибку, если его нет)
    aruco = require_aruco()
    # Получаем числовой ID выбранного словаря (например, DICT_6X6_250)
    dictionary_id = getattr(aruco, DICTIONARY_NAME, None)
    # Если такого словаря в OpenCV нет, выбрасываем ошибку
    if dictionary_id is None:
        raise RuntimeError(f"OpenCV не поддерживает словарь {DICTIONARY_NAME}.")

    # Создаем объект словаря ArUco на основе его ID
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    # Создаем объект доски
    board = create_board(aruco, dictionary)
    # Создаем детектор маркеров и параметры к нему
    detector, detector_parameters = create_detector(aruco, dictionary)
    # Формируем карту идеальных 3D точек для маркеров доски
    point_map = board_point_map(board)

    # Вычисляем ожидаемые ID маркеров (от START_ID до количества маркеров на доске)
    expected_ids = set(range(START_ID, START_ID + BOARD_COLUMNS * BOARD_ROWS))
    # Проверяем правильность созданной доски: совпадают ли сгенерированные ID с ожидаемыми
    if set(point_map) != expected_ids:
        raise RuntimeError("Идентификаторы созданной GridBoard не соответствуют диапазону 0...215.")

    # Инициализируем список для сбора 3D координат для всех сохраненных кадров
    captured_object_points: list[np.ndarray] = []
    # Инициализируем список для сбора 2D координат для всех сохраненных кадров
    captured_image_points: list[np.ndarray] = []
    # Задаем начальные пустые значения для параметров матрицы и дисторсии камеры
    camera_matrix: np.ndarray | None = None
    distortion: np.ndarray | None = None
    # Флаг, который показывает, включен ли предпросмотр видео без искажений (undistortion)
    use_undistortion = False
    # Начальный статус (сообщение на экране)
    status = "Show the ArUco board to the camera"

    # Открываем видеопоток с указанной камеры (обычно веб-камера с индексом 0)
    camera = cv.VideoCapture(args.camera)
    # Если камера не открылась (занята или отсутствует)
    if not camera.isOpened():
        # Выводим сообщение об ошибке
        print(f"Не удалось открыть камеру {args.camera}.")
        print("Закройте другие программы, использующие камеру, или выберите другую: --camera 1")
        # Возвращаем код ошибки 1, чтобы завершить скрипт
        return 1

    # Выводим в консоль начальную информацию о процессе калибровки
    print("Калибровка камеры по ArUco GridBoard запущена.")
    print(
        f"Доска: {BOARD_COLUMNS} x {BOARD_ROWS}, маркер {MARKER_SIZE_MM:g} мм, "
        f"промежуток {MARKER_SEPARATION_MM:g} мм, {DICTIONARY_NAME}, ID {START_ID}...215"
    )
    print("SPACE — добавить ракурс; C — рассчитать; U — коррекция; R — сброс; Q/ESC — выход")

    # Инициализируем переменную для хранения размеров изображения
    image_size: tuple[int, int] | None = None
    
    # Блок try-finally гарантирует, что камера и окна закроются даже при ошибке в коде
    try:
        # Основной бесконечный цикл обработки видеокадров
        while True:
            # Читаем один кадр с камеры. ok = True, если кадр прочитан успешно, frame = сам кадр
            ok, frame = camera.read()
            # Если камера не отдала кадр, выходим с ошибкой
            if not ok:
                print("Не удалось получить кадр с камеры.")
                return 1

            # Сохраняем размер кадра (ширина, высота)
            image_size = (frame.shape[1], frame.shape[0])
            # Конвертируем кадр в оттенки серого (для алгоритма детекции)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Ищем маркеры на кадре
            detected_corners, detected_ids = detect_markers(
                aruco, detector, detector_parameters, dictionary, gray
            )
            # Фильтруем маркеры: оставляем только те, что относятся к калибровочной доске
            board_corners, board_ids = select_board_markers(
                detected_corners, detected_ids, point_map
            )
            # Считаем количество найденных маркеров с доски
            marker_count = 0 if board_ids is None else len(board_ids)

            # Если на кадре есть маркеры доски
            if board_ids is not None:
                # Рисуем зеленые рамки вокруг обнаруженных маркеров (прямо поверх кадра)
                aruco.drawDetectedMarkers(frame, board_corners, board_ids)

            # Переменная display содержит кадр, который будет показан пользователю
            display = frame
            # Если включен режим устранения искажений и калибровка уже рассчитана
            if use_undistortion and camera_matrix is not None and distortion is not None:
                # Применяем устранение искажений (выравнивание дисторсии) к кадру
                display = cv.undistort(frame, camera_matrix, distortion)

            # Проверяем, достаточно ли маркеров в кадре (на основе переданного аргумента)
            enough_markers = marker_count >= args.min_markers
            # Выбираем цвет текста: зеленый, если маркеров достаточно, иначе — оранжевый
            marker_color = (50, 220, 50) if enough_markers else (40, 180, 255)
            # Выводим строку 0: Количество маркеров в кадре
            draw_text(display, f"Board markers: {marker_count} (need {args.min_markers})", 0, marker_color)
            # Выводим строку 1: Прогресс сохраненных ракурсов
            draw_text(display, f"Saved views: {len(captured_image_points)}/{args.samples}", 1, (255, 255, 255))
            # Выводим строку 2: Текущий статус приложения
            draw_text(display, status, 2, (255, 255, 255))
            # Выводим строку 3: Подсказка по горячим клавишам управления
            draw_text(display, "SPACE capture | C calibrate | U undistort | R reset | Q exit", 3, (220, 220, 220))
            
            # Показываем получившийся кадр (с рамками и текстом) в отдельном окне
            cv.imshow("ArUco camera calibration", display)

            # Ждем 1 миллисекунду нажатия клавиши. '& 0xFF' маскирует лишние биты для кроссплатформенности.
            key = cv.waitKey(1) & 0xFF
            
            # Если нажата клавиша Esc (код 27), или 'q', или 'Q'
            if key in (27, ord("q"), ord("Q")):
                # Прерываем цикл (завершаем программу)
                break
                
            # Если нажат пробел (сохранить ракурс)
            if key == ord(" "):
                # Если маркеров нет или их слишком мало
                if board_ids is None or not enough_markers:
                    # Обновляем статус и выводим предупреждение в консоль
                    status = f"Need at least {args.min_markers} board markers"
                    print(f"Нужно видеть минимум {args.min_markers} маркеров — ракурс не добавлен.")
                # Если маркеров достаточно
                else:
                    # Получаем пары 3D-2D точек для текущего ракурса
                    object_set, image_set = make_calibration_points(
                        board_corners, board_ids, point_map
                    )
                    # Сохраняем 3D точки в общий список
                    captured_object_points.append(object_set)
                    # Сохраняем 2D точки в общий список
                    captured_image_points.append(image_set)
                    # Обновляем статус
                    status = f"Captured view {len(captured_image_points)}"
                    # Выводим успех в консоль
                    print(
                        f"Добавлен ракурс {len(captured_image_points)}/{args.samples} "
                        f"({marker_count} маркеров)"
                    )
                    
            # Если нажата 'r' или 'R' (сброс, Reset)
            elif key in (ord("r"), ord("R")):
                # Очищаем собранные 3D точки
                captured_object_points.clear()
                # Очищаем собранные 2D точки
                captured_image_points.clear()
                # Обнуляем результаты калибровки
                camera_matrix = None
                distortion = None
                # Выключаем режим устранения искажений
                use_undistortion = False
                # Обновляем статус
                status = "All views removed"
                print("Все сохранённые ракурсы удалены.")
                
            # Если нажата 'u' или 'U' (переключить предпросмотр устранения дисторсии, Undistort)
            elif key in (ord("u"), ord("U")):
                # Если матрица еще не рассчитана (не нажимали 'C')
                if camera_matrix is None:
                    status = "Calibrate first by pressing C"
                else:
                    # Переключаем флаг на противоположный (вкл/выкл)
                    use_undistortion = not use_undistortion
                    # Обновляем статус
                    status = f"Undistortion {'ON' if use_undistortion else 'OFF'}"
                    
            # Если нажата 'c' или 'C' (запуск расчета Калибровки, Calibrate)
            elif key in (ord("c"), ord("C")):
                # Проверяем, собрали ли мы минимально необходимое число ракурсов (samples)
                if len(captured_image_points) < args.samples:
                    # Считаем, сколько не хватает
                    missing = args.samples - len(captured_image_points)
                    # Обновляем статус и выводим в консоль
                    status = f"Need {missing} more views"
                    print(f"Нужно добавить ещё {missing} ракурсов.")
                    # Пропускаем дальнейшие вычисления калибровки
                    continue

                # Запускаем алгоритм калибровки OpenCV (вычисляет матрицу камеры, дисторсию и вектора поворота/смещения)
                rms, camera_matrix, distortion, rvecs, tvecs = cv.calibrateCamera(
                    captured_object_points,
                    captured_image_points,
                    image_size,
                    None,
                    None,
                )
                # Вычисляем нашу собственную среднюю ошибку репроекции (как меру точности)
                reprojection_error = mean_reprojection_error(
                    captured_object_points,
                    captured_image_points,
                    rvecs,
                    tvecs,
                    camera_matrix,
                    distortion,
                )
                # Сохраняем результаты в файлы (.npz и .yaml)
                npz_path, yaml_path = save_calibration(
                    args.output,
                    camera_matrix,
                    distortion,
                    image_size,
                    reprojection_error,
                )
                # Автоматически включаем показ изображения без дисторсии
                use_undistortion = True
                # Обновляем статус
                status = f"Saved, error: {reprojection_error:.3f} px"
                # Выводим подробный лог результатов в консоль
                print("\nКалибровка завершена.")
                print(f"RMS OpenCV: {rms:.6f}")
                print(f"Средняя ошибка репроекции: {reprojection_error:.4f} px")
                print("Матрица камеры:\n", camera_matrix)
                print("Коэффициенты дисторсии:\n", distortion)
                print(f"Сохранено: {npz_path}")
                print(f"Сохранено: {yaml_path}")
    
    # Блок finally выполнится в любом случае (при выходе из цикла или при внезапной ошибке)
    finally:
        # Освобождаем (закрываем) камеру
        camera.release()
        # Закрываем все окна OpenCV
        cv.destroyAllWindows()

    # Возвращаем код 0, сигнализирующий об успешном завершении программы
    return 0


# Стандартная конструкция Python: код ниже выполнится только если этот файл запущен напрямую (а не импортирован)
if __name__ == "__main__":
    # Вызываем main() и передаем ее результат (0 или 1) системе при завершении программы
    raise SystemExit(main())