"""Графический интерфейс оператора для vision-ячейки KUKA + OpenCV.

Материал-дизайн на customtkinter + ttk (единственная внешняя
зависимость — Pillow и customtkinter). Все консольные инструменты
проекта собраны в одном окне:

    * источник кадров: индекс камеры, видеофайл или снимок;
    * живой просмотр детекции: рамки, центры, углы, таблица деталей;
    * параметры пайплайна меняются «на лету», в том числе при
      работающем сервере (детектор общий);
    * калибровка поля: клики по кадру + координаты робота с пульта,
      расчет и сохранение гомографии, проверка курсором мыши;
    * калибровка камеры по ArUco GridBoard (ракурсы, расчет, сохранение);
    * TCP-сервер KUKA.Ethernet KRL: запуск/остановка, журнал обмена XML,
      тестовый захват без робота.

Запуск:
    python gui.py
    python gui.py --camera 0        # сразу подключить камеру
"""

from __future__ import annotations

import argparse
import math
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

import calibration as camcal
from field_calibration import (
    FieldCalibration,
    calibration_residuals,
    compute_field_matrix,
    load_field_calibration,
    save_field_calibration,
    transform_angle_deg,
    transform_pixel_to_robot,
)
from server import FrameSource, VisionServer, build_vision_xml, parts_to_robot
from vision import (
    PartDetector,
    VisionParams,
    draw_parts,
    load_camera_calibration,
    normalize_angle,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CAMERA_CALIBRATION = BASE_DIR / "camera_calibration.npz"
DEFAULT_FIELD_CALIBRATION = BASE_DIR / "field_calibration.npz"

POLL_MS = 40          # период обновления интерфейса
ARUCO_EVERY = 3       # детекция ArUco каждый N-й тик (только в своей вкладке)
LOG_MAX_LINES = 800   # предел журнала событий

FONT_FAMILY = "Segoe UI"

# Палитра Material Design (светлая тема, повышенный контраст).
C = {
    "primary":        "#1565C0",
    "primary_hover":  "#1E78D2",
    "tonal":          "#DCEBFA",
    "tonal_hover":    "#C4DDF5",
    "on_primary":     "#FFFFFF",
    "appbar":         "#0D47A1",
    "bg":             "#DFE6EF",
    "surface":        "#FFFFFF",
    "surface_alt":    "#F1F5FA",
    "head_bg":        "#E3ECF6",   # шапка таблицы — светлее шапки карточки
    "head_text":      "#0D47A1",
    "head_hover":     "#D5E2F1",
    "border":         "#B7C4D4",
    "border_strong":  "#8FA2B8",
    "text":           "#0E1216",
    "text_secondary": "#3D4854",
    "hover_row":      "#E4EFFB",
    "success":        "#156D2C",
    "success_bg":     "#DCF1E2",
    "warning":        "#7E5300",
    "warning_bg":     "#FBECD0",
    "error":          "#B3271B",
    "error_hover":    "#962016",
    "video_bg":       "#0C1015",
    "log_bg":         "#141920",
    "log_text":       "#DCE6F6",
    "row_alt":        "#EEF4FB",
}

# Единые размеры элементов.
BTN_H = 34            # высота всех кнопок в рядах
BTN_H_MAIN = 38       # высота главных действий вкладки
ENTRY_H = 32
SPIN_W = 86           # ширина всех спинбоксов


def parse_xy(text: str) -> tuple[float, float]:
    """Разбирает 'X,Y' / 'X;Y' / 'X Y' (запятая-разделитель или точка)."""
    normalized = text.replace(";", ",").replace("\t", " ")
    tokens = [t for t in (s.strip() for s in normalized.split(",")) if t]
    if len(tokens) == 1 and " " in tokens[0]:
        tokens = tokens[0].split()
    if len(tokens) != 2:
        raise ValueError("Введите два числа: X,Y (например 350.2,-120.4)")
    try:
        return float(tokens[0].replace(",", ".")), float(tokens[1].replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Не числа: {text!r}") from exc


class SharedSource:
    """Потокобезопасная обертка над FrameSource.

    Один источник делится конвейером предпросмотра и TCP-сервером,
    поэтому каждое чтение (включая переоткрытие захвата) под замком.
    """

    def __init__(self, source: int | str) -> None:
        self._inner = FrameSource(source)
        self._lock = threading.Lock()

    @property
    def is_static(self) -> bool:
        return self._inner.is_static

    def read(self) -> np.ndarray:
        with self._lock:
            return self._inner.read()

    def warm_up(self, frames: int = 3) -> None:
        for _ in range(frames):
            self.read()

    def release(self) -> None:
        with self._lock:
            self._inner.release()


class WhiteArrowOptionMenu(ctk.CTkOptionMenu):
    """CTkOptionMenu с белой стрелкой и собственной рамкой.

    Стрелка рисуется движком цветом text_color в несколько сглаженных
    слоев, поэтому рисуем с белым text_color, а тексту значения
    возвращаем темный цвет после отрисовки.

    Рамка дорисовывается поверх штатной отрисовки: левая часть — серая,
    правая — цветом синей кнопки, поэтому серой линии справа от стрелки
    нет и синий блок доходит до края элемента.
    """

    def _draw(self, no_color_updates=False) -> None:
        original = self._text_color
        self._text_color = "#FFFFFF"
        try:
            super()._draw(no_color_updates)
        finally:
            self._text_color = original
        try:
            self._text_label.configure(fg=self._apply_appearance_mode(original))
        except (tk.TclError, AttributeError):
            pass

        try:
            width = self._apply_widget_scaling(self._current_width)
            height = self._apply_widget_scaling(self._current_height)
            radius = self._apply_widget_scaling(self._corner_radius)
            self._draw_engine.draw_rounded_rect_with_border_vertical_split(
                width, height, radius, 2, width - height
            )
            border = self._apply_appearance_mode(
                getattr(self, "_gui_border_color", C["border_strong"])
            )
            right = self._apply_appearance_mode(
                getattr(self, "_gui_arrow_section_color", self._button_color)
            )
            self._canvas.itemconfig("border_parts_left", outline=border, fill=border)
            self._canvas.itemconfig("border_parts_right", outline=right, fill=right)
        except (tk.TclError, AttributeError):
            pass


class Tooltip:
    """Простая всплывающая подсказка (Material caption)."""

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 550) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay_ms
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        if self._tip is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        frame = ctk.CTkFrame(
            self._tip, fg_color="#37474F", corner_radius=6,
        )
        frame.pack(padx=1, pady=1)
        ctk.CTkLabel(
            frame, text=self.text, justify="left",
            font=ctk.CTkFont(FONT_FAMILY, 11), text_color="#FFFFFF",
        ).pack(padx=10, pady=5)

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class RobotPointDialog(ctk.CTkToplevel):
    """Модальный диалог ввода координат робота для реперной точки."""

    def __init__(
        self,
        master: tk.Misc,
        index: int,
        pixel: tuple[float, float],
        initial: str = "",
    ):
        super().__init__(master)
        self.title(f"Точка {index}")
        self.resizable(False, False)
        self.configure(fg_color=C["surface"])
        self.result: tuple[float, float] | None = None
        self.transient(master)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=(18, 16))

        ctk.CTkLabel(
            body, anchor="w",
            text=f"Реперная точка {index}",
            font=ctk.CTkFont(FONT_FAMILY, 15, "bold"), text_color=C["text"],
        ).pack(fill="x")
        ctk.CTkLabel(
            body, anchor="w",
            text=f"Пиксель кадра: u={pixel[0]:.0f}, v={pixel[1]:.0f}",
            font=ctk.CTkFont(FONT_FAMILY, 12), text_color=C["text_secondary"],
        ).pack(fill="x", pady=(2, 12))

        ctk.CTkLabel(
            body, anchor="w", text="Координаты робота X,Y (мм)",
            font=ctk.CTkFont(FONT_FAMILY, 12, "bold"),
            text_color=C["text_secondary"],
        ).pack(fill="x")
        self.entry = ctk.CTkEntry(
            body, height=38, font=ctk.CTkFont(FONT_FAMILY, 14),
            border_color=C["border"], fg_color=C["surface"],
            text_color=C["text"], placeholder_text="350.2,-120.4",
        )
        if initial:
            self.entry.insert(0, initial)
        self.entry.pack(fill="x", pady=(4, 2))
        self.entry.focus_set()
        self.entry.bind("<Enter>", lambda _e: self.entry.configure(border_color=C["primary"]))
        self.entry.bind("<Leave>", lambda _e: self.entry.configure(border_color=C["border"]))

        self.error_var = tk.StringVar()
        ctk.CTkLabel(
            body, anchor="w", textvariable=self.error_var,
            font=ctk.CTkFont(FONT_FAMILY, 11), text_color=C["error"],
        ).pack(fill="x", pady=(0, 8))

        # Кнопки одинаковой ширины: равные колонки grid.
        buttons = ctk.CTkFrame(body, fg_color="transparent")
        buttons.pack(fill="x")
        buttons.columnconfigure((0, 1), weight=1, uniform="btn")
        ctk.CTkButton(
            buttons, text="Добавить точку", height=BTN_H_MAIN,
            command=self.confirm, font=self._btn_font(),
            fg_color=C["primary"], hover_color=C["primary_hover"],
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(
            buttons, text="Отмена", height=BTN_H_MAIN,
            command=self.cancel, font=self._btn_font(),
            fg_color="transparent", border_width=2,
            border_color=C["border"], text_color=C["text_secondary"],
            hover_color=C["bg"],
        ).grid(row=0, column=1, sticky="ew")

        self.entry.bind("<Return>", lambda _e: self.confirm())
        self.bind("<Escape>", lambda _e: self.cancel())
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.update_idletasks()
        x = master.winfo_rootx() + max((master.winfo_width() - self.winfo_width()) // 2, 0)
        y = master.winfo_rooty() + max((master.winfo_height() - self.winfo_height()) // 3, 0)
        self.geometry(f"+{x}+{y}")
        self.grab_set()
        self.wait_window()

    @staticmethod
    def _btn_font() -> ctk.CTkFont:
        return ctk.CTkFont(FONT_FAMILY, 13, "bold")

    def confirm(self, event=None) -> None:
        try:
            self.result = parse_xy(self.entry.get())
        except ValueError as exc:
            self.error_var.set(str(exc))
            return
        self.destroy()

    def cancel(self, event=None) -> None:
        self.result = None
        self.destroy()


class VisionGui:
    """Главное окно оператора."""

    TAB_DETECT = "Детекция"
    TAB_FIELD = "Поле"
    TAB_CAMERA = "Камера"
    TAB_SERVER = "Сервер"

    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        root.title("KUKA Vision · Pick-and-Place")
        width, height = 1520, 900
        root.update_idletasks()
        x = max((root.winfo_screenwidth() - width) // 2, 0)
        y = max((root.winfo_screenheight() - height) // 2, 0)
        root.geometry(f"{width}x{height}+{x}+{y}")
        root.minsize(1200, 720)
        root.configure(fg_color=C["bg"])
        try:
            root.state("zoomed")  # сразу развернуть на весь экран
        except tk.TclError:
            pass
        # повтор после отрисовки окна — на случай, если состояние сбросилось
        root.after(150, lambda: self._ensure_zoomed())

        self.events: queue.Queue = queue.Queue()
        self.fonts = type("Fonts", (), {})()
        self._make_fonts()

        self.detector = PartDetector()
        self.field: FieldCalibration | None = None
        self.camera_calibrated = False
        self._load_default_calibrations()

        self.source: SharedSource | None = None
        self.capture_thread: threading.Thread | None = None
        self.capture_running = False
        self.state_lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.latest_parts: list = []
        self.latest_stamp = 0.0
        self.fps = 0.0
        self.flash_until = 0.0
        self.flash_frame: np.ndarray | None = None

        self.pixel_points: list[tuple[float, float]] = []
        self.robot_points: list[tuple[float, float]] = []
        self.add_points_mode = False

        self.aruco_available = hasattr(cv, "aruco")
        self._board = None
        self._detector_aruco = None
        self._point_map = None
        self.cam_object_points: list[np.ndarray] = []
        self.cam_image_points: list[np.ndarray] = []
        self.board_corners = None
        self.board_ids = None
        self.marker_count = 0

        self.server: VisionServer | None = None
        self.server_thread: threading.Thread | None = None
        self.server_busy_until = 0.0

        self.mouse_frame: tuple[int, int] | None = None
        self.view_params = (1.0, 0.0, 0.0)
        self._photo: ImageTk.PhotoImage | None = None
        self._tick_count = 0
        self._chips_state: tuple = ()
        self._last_tab: str | None = None

        self._style_ttk()
        self._build_appbar()
        self._build_connection_bar()
        # статусбар пакуется до основной области, чтобы закрепиться внизу
        self._build_statusbar()
        self._build_main_area()

        root.bind("<Escape>", lambda _e: self.set_add_points_mode(False))
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(POLL_MS, self._ui_tick)

    def _ensure_zoomed(self) -> None:
        """Повторно разворачивает окно, если состояние сбросилось."""
        try:
            if self.root.state() != "zoomed":
                self.root.state("zoomed")
        except tk.TclError:
            pass

    def _make_fonts(self) -> None:
        f = self.fonts
        f.title = ctk.CTkFont(FONT_FAMILY, 17, "bold")
        f.section = ctk.CTkFont(FONT_FAMILY, 12, "bold")
        f.body = ctk.CTkFont(FONT_FAMILY, 13)
        f.body_bold = ctk.CTkFont(FONT_FAMILY, 13, "bold")
        f.caption = ctk.CTkFont(FONT_FAMILY, 11)
        f.button = ctk.CTkFont(FONT_FAMILY, 13, "bold")

    # ------------------------------------------------------------------
    # Стиль ttk-виджетов (таблицы, спинбоксы, комбобоксы) под Material
    # ------------------------------------------------------------------

    def _style_ttk(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Card.Treeview",
            background=C["surface"],
            fieldbackground=C["surface"],
            foreground=C["text"],
            rowheight=30,
            font=(FONT_FAMILY, 11),
            borderwidth=0,
            relief="flat",
        )
        style.configure(
            "Card.Treeview.Heading",
            background=C["head_bg"],
            foreground=C["head_text"],
            font=(FONT_FAMILY, 11, "bold"),
            relief="flat",
            padding=(8, 7),
        )
        style.map(
            "Card.Treeview",
            background=[
                ("selected", C["primary"]),
                ("active", C["hover_row"]),
            ],
            foreground=[("selected", C["on_primary"])],
        )
        style.map(
            "Card.Treeview.Heading",
            background=[("active", C["head_hover"])],
        )

        style.configure(
            "Material.TSpinbox",
            arrowsize=13,
            padding=(6, 4),
            bordercolor=C["border_strong"],
            lightcolor=C["surface"],
            darkcolor=C["surface"],
            fieldbackground=C["surface"],
            foreground=C["text"],
            borderwidth=1,
        )
        style.map(
            "Material.TSpinbox",
            bordercolor=[("focus", C["primary"]), ("active", C["primary"]), ("hover", C["primary"])],
        )
        style.configure(
            "Material.TCombobox",
            padding=(8, 4),
            bordercolor=C["border_strong"],
            lightcolor=C["surface"],
            darkcolor=C["surface"],
            fieldbackground=C["surface"],
            foreground=C["text"],
            arrowsize=13,
            borderwidth=1,
        )
        style.map(
            "Material.TCombobox",
            bordercolor=[("focus", C["primary"]), ("active", C["primary"])],
            fieldbackground=[("readonly", C["surface"])],
        )
        self.root.option_add("*TCombobox*Listbox.Font", (FONT_FAMILY, 11))

    # ------------------------------------------------------------------
    # Переиспользуемые элементы Material
    # ------------------------------------------------------------------

    def card(self, parent, title: str | None = None, padding: int = 14):
        """Белая карточка с синей шапкой без скруглений; возвращает содержимое."""
        outer = ctk.CTkFrame(
            parent, fg_color=C["surface"], corner_radius=0,
            border_width=1, border_color=C["border_strong"],
        )
        outer.pack(fill="x", pady=(0, 12))
        if title:
            header = ctk.CTkFrame(
                outer, fg_color=C["primary"], corner_radius=0, height=36
            )
            header.pack(fill="x")
            header.pack_propagate(False)
            ctk.CTkLabel(
                header, text=title.upper(), anchor="w",
                font=self.fonts.section, text_color=C["on_primary"],
            ).pack(side="left", padx=14)
            top_pad = (12, 12)  # зазор, чтобы синяя шапка не сливалась с содержимым
        else:
            top_pad = (padding, padding)
        inner = ctk.CTkFrame(outer, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=padding, pady=top_pad)
        return inner

    def button(self, parent, text, command, kind="primary", height=BTN_H):
        """Кнопка одного из видов Material: primary/tonal/outline/ghost/danger."""
        base = dict(
            height=height, corner_radius=8, font=self.fonts.button,
            text=text, command=command, cursor="hand2",
        )
        if kind == "primary":
            return ctk.CTkButton(
                parent, fg_color=C["primary"], hover_color=C["primary_hover"],
                text_color=C["on_primary"], text_color_disabled="#AFCDF2", **base)
        if kind == "danger":
            return ctk.CTkButton(
                parent, fg_color=C["error"], hover_color=C["error_hover"],
                text_color=C["on_primary"], text_color_disabled="#EFC3BD", **base)
        if kind == "outline":
            return ctk.CTkButton(
                parent, fg_color="transparent", hover_color=C["hover_row"],
                border_width=2, border_color=C["border_strong"],
                text_color=C["text"], text_color_disabled="#93A2B4", **base)
        if kind == "ghost":
            # вторичная кнопка с четко очерченной границей
            return ctk.CTkButton(
                parent, fg_color=C["surface"], hover_color=C["hover_row"],
                border_width=2, border_color=C["border_strong"],
                text_color=C["text_secondary"], text_color_disabled="#93A2B4", **base)
        return ctk.CTkButton(
            parent, fg_color=C["tonal"], hover_color=C["tonal_hover"],
            text_color=C["primary"], text_color_disabled="#7FA6CC", **base)

    def set_button_state(self, btn, enabled: bool) -> None:
        """Включает/выключает кнопку (цвет неактивного текста задан в button)."""
        btn.configure(state="normal" if enabled else "disabled")

    def button_row(self, parent, *specs, columns=None) -> None:
        """Ряд кнопок равной ширины: колонки grid с одинаковым весом."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0, 8))
        count = columns or len(specs)
        gap = 8
        for i in range(count):
            row.columnconfigure(i, weight=1, uniform="btnrow")
        for i, spec in enumerate(specs):
            text, command, kind = spec
            btn = self.button(row, text, command, kind)
            btn.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else gap // 2, 0 if i == count - 1 else gap // 2))
        return row

    def chip(self, parent) -> tuple[ctk.CTkFrame, ctk.CTkLabel]:
        """Статус-чип: скругленная плашка с рамкой цвета состояния."""
        frame = ctk.CTkFrame(
            parent, corner_radius=999, fg_color=C["bg"], height=26,
            border_width=1, border_color=C["border_strong"],
        )
        label = ctk.CTkLabel(
            frame, text="", height=26, font=ctk.CTkFont(FONT_FAMILY, 11, "bold"),
            text_color=C["text_secondary"],
        )
        label.pack(padx=12, pady=1)
        return frame, label

    def set_chip(self, pair, text: str, fg: str, bg: str) -> None:
        frame, label = pair
        frame.configure(fg_color=bg, border_color=fg)
        label.configure(
            text=text, text_color=fg, font=ctk.CTkFont(FONT_FAMILY, 11, "bold")
        )

    def param_row(self, parent, row: int, label: str, attr: str,
                  frm: float, to: float, inc: float, as_int: bool = False,
                  tooltip: str | None = None) -> None:
        value = getattr(self.detector.params, attr)
        var = tk.StringVar(value=f"{value:g}")

        def apply(*_args) -> None:
            try:
                parsed = float(var.get().replace(",", "."))
            except ValueError:
                return
            if not math.isfinite(parsed):
                return
            parsed = min(max(parsed, frm), to)
            if as_int:
                parsed = int(round(parsed))
            if parsed != getattr(self.detector.params, attr):
                setattr(self.detector.params, attr, parsed)

        var.trace_add("write", apply)
        lbl = ctk.CTkLabel(
            parent, text=label, anchor="w", font=self.fonts.body,
            text_color=C["text"],
        )
        lbl.grid(row=row, column=0, sticky="ew", pady=3)
        spin = ttk.Spinbox(
            parent, textvariable=var, from_=frm, to=to, increment=inc,
            width=10, style="Material.TSpinbox", font=(FONT_FAMILY, 12),
            justify="right",
        )
        spin.grid(row=row, column=1, sticky="e", pady=3)
        parent.columnconfigure(0, weight=1)
        if tooltip:
            Tooltip(spin, tooltip)

    def switch_row(self, parent, row: int, text: str, variable, command,
                   tooltip: str | None = None):
        widget = ctk.CTkSwitch(
            parent, text=text, variable=variable, command=command,
            font=self.fonts.body, text_color=C["text"],
            progress_color=C["primary"], button_color="#B9C4D0",
            button_hover_color=C["primary"],
        )
        widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=5)
        if tooltip:
            Tooltip(widget, tooltip)
        return widget

    def combo(self, parent, variable, values, width=150):
        """Выпадающий список: скругленная рамка вокруг белого поля,
        справа серой линии нет — синий блок со стрелкой до края."""
        wrap = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=0)
        menu = WhiteArrowOptionMenu(
            wrap,
            variable=variable,
            values=list(values),
            width=width,
            height=ENTRY_H,
            corner_radius=8,
            font=self.fonts.body,
            dropdown_font=ctk.CTkFont(FONT_FAMILY, 12),
            fg_color=C["surface"],
            bg_color=C["surface"],
            text_color=C["text"],
            button_color=C["primary"],
            button_hover_color=C["primary_hover"],
            dropdown_fg_color=C["surface"],
            dropdown_hover_color=C["tonal"],
            dropdown_text_color=C["text"],
        )
        menu._gui_border_color = C["border_strong"]
        menu._gui_arrow_section_color = C["primary"]
        menu.pack(fill="both", expand=True)
        return wrap

    @staticmethod
    def entry_hover(entry) -> None:
        """Приятный ховер: рамка поля подсвечивается при наведении."""
        entry.bind("<Enter>", lambda _e: entry.configure(border_color=C["primary"]))
        entry.bind("<Leave>", lambda _e: entry.configure(border_color=C["border"]))

    # ------------------------------------------------------------------
    # Каркас окна
    # ------------------------------------------------------------------

    def _build_appbar(self) -> None:
        bar = ctk.CTkFrame(self.root, fg_color=C["appbar"], corner_radius=0, height=58)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        ctk.CTkLabel(
            bar, text="KUKA Vision", font=self.fonts.title,
            text_color=C["on_primary"],
        ).pack(side="left", padx=(20, 4), pady=10)
        ctk.CTkLabel(
            bar, text="· Pick-and-Place", font=ctk.CTkFont(FONT_FAMILY, 13),
            text_color="#AECBEF",
        ).pack(side="left", pady=10)

        chips_box = ctk.CTkFrame(bar, fg_color="transparent")
        chips_box.pack(side="right", padx=16)
        self.chip_camera = self.chip(chips_box)
        self.chip_field = self.chip(chips_box)
        self.chip_server = self.chip(chips_box)
        for pair in (self.chip_camera, self.chip_field, self.chip_server):
            pair[0].pack(side="left", padx=(8, 0))

    def _build_connection_bar(self) -> None:
        card_outer = ctk.CTkFrame(
            self.root, fg_color=C["surface"], corner_radius=12,
            border_width=1, border_color=C["border"],
        )
        card_outer.pack(fill="x", padx=16, pady=(12, 0))
        row = ctk.CTkFrame(card_outer, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=10)

        ctk.CTkLabel(
            row, text="Источник кадров", font=self.fonts.body_bold,
            text_color=C["text"],
        ).pack(side="left", padx=(0, 12))

        self.source_var = tk.StringVar(value="0")
        entry = ctk.CTkEntry(
            row, textvariable=self.source_var, width=260, height=ENTRY_H,
            font=self.fonts.body, border_color=C["border"],
            fg_color=C["surface"], text_color=C["text"],
            placeholder_text="индекс камеры / путь к файлу",
        )
        entry.pack(side="left")
        entry.bind("<Return>", lambda _e: self.on_connect_clicked())
        self.entry_hover(entry)
        Tooltip(entry, "Индекс камеры (0, 1…), путь к видео или снимку.\nEnter — подключить.")

        self.btn_connect_holder = ctk.CTkFrame(row, fg_color="transparent")
        self.btn_connect_holder.pack(side="left", padx=(12, 0))
        self.btn_connect = self.button(self.btn_connect_holder, "Подключить", self.on_connect_clicked, "primary")
        self.btn_connect.pack(side="left")
        self.btn_disconnect = self.button(self.btn_connect_holder, "Отключить", self.on_disconnect_clicked, "outline", BTN_H)
        self.set_button_state(self.btn_disconnect, False)
        self.btn_disconnect.pack(side="left", padx=(8, 0))
        # одинаковая ширина пары
        for b in (self.btn_connect, self.btn_disconnect):
            b.configure(width=118)

        self.lbl_source_state = ctk.CTkLabel(
            row, text="Источник отключен", anchor="e",
            font=self.fonts.caption, text_color=C["text_secondary"],
        )
        self.lbl_source_state.pack(side="right", padx=(12, 0))

    def _build_main_area(self) -> None:
        panes = ctk.CTkFrame(self.root, fg_color="transparent")
        panes.pack(fill="both", expand=True, padx=16, pady=(12, 8))
        panes.columnconfigure(0, weight=1, uniform="half")
        panes.columnconfigure(1, weight=1, uniform="half")
        panes.rowconfigure(0, weight=1)

        # --- левая колонка: видео + таблица (только pack) ---
        left = ctk.CTkFrame(panes, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        video_card = ctk.CTkFrame(
            left, fg_color=C["surface"], corner_radius=12,
            border_width=1, border_color=C["border"],
        )
        video_card.pack(fill="both", expand=True, pady=(0, 12))
        video_inner = ctk.CTkFrame(video_card, fg_color="transparent")
        video_inner.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas = tk.Canvas(
            video_inner, bg=C["video_bg"], highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", lambda _e: setattr(self, "mouse_frame", None))
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Control-Button-1>", self._on_canvas_click)
        hint = ("Нет сигнала\nПодключите источник кадров\n(камера, видео или снимок)")
        self.canvas.create_text(
            480, 260, text=hint, fill="#8FA1B3",
            font=("Segoe UI", 14), justify="center", tags="placeholder",
        )

        table_card = self.card(left, "Обнаруженные детали")
        columns = ("n", "u", "v", "x", "y", "a", "s")
        headers = ("№", "u, px", "v, px", "X, мм", "Y, мм", "угол°", "S, px²")
        widths = (50, 95, 95, 115, 115, 95, 135)
        parts_wrap = ctk.CTkFrame(
            table_card, fg_color=C["surface"], corner_radius=8,
            border_width=1, border_color=C["border"],
        )
        parts_wrap.pack(fill="x")
        self.parts_tree = ttk.Treeview(
            parts_wrap, columns=columns, show="headings", height=5,
            style="Card.Treeview",
        )
        for col, header, width in zip(columns, headers, widths):
            anchor = "center" if col in ("n", "a") else "e"
            self.parts_tree.heading(col, text=header)
            self.parts_tree.column(
                col, width=width, anchor=anchor, stretch=True, minwidth=60
            )
        self.parts_tree.pack(fill="x")
        self.parts_tree.tag_configure("odd", background=C["surface"])
        self.parts_tree.tag_configure("even", background=C["row_alt"])
        menu = tk.Menu(self.root, tearoff=0, font=(FONT_FAMILY, 11))
        menu.add_command(label="Копировать X, Y, угол", command=self.copy_selected_part)
        menu.add_command(label="Копировать строку целиком", command=lambda: self.copy_selected_part(full=True))
        self.parts_menu = menu
        self.parts_tree.bind("<Button-3>", self._show_parts_menu)
        self.parts_tree.bind("<Double-Button-1>", lambda _e: self.copy_selected_part())

        # --- правая колонка: вкладки ---
        self.tabview = ctk.CTkTabview(
            panes, width=430, corner_radius=12,
            command=self._on_tab_changed,
            fg_color=C["surface"],
            text_color=C["text_secondary"],
            segmented_button_fg_color=C["surface_alt"],
            segmented_button_selected_color=C["primary"],
            segmented_button_selected_hover_color=C["primary_hover"],
            segmented_button_unselected_color=C["surface_alt"],
            segmented_button_unselected_hover_color=C["tonal"],
            text_color_disabled=C["border"],
        )
        self.tabview.grid(row=0, column=1, sticky="nsew")
        try:
            self.tabview._segmented_button.configure(font=self.fonts.body_bold)
        except (AttributeError, tk.TclError):
            pass
        for name in (self.TAB_DETECT, self.TAB_FIELD, self.TAB_CAMERA, self.TAB_SERVER):
            self.tabview.add(name)
        self.detection_tab = self.tabview.tab(self.TAB_DETECT)
        self.field_tab = self.tabview.tab(self.TAB_FIELD)
        self.camera_tab = self.tabview.tab(self.TAB_CAMERA)
        self.server_tab = self.tabview.tab(self.TAB_SERVER)
        for tab in (self.detection_tab, self.field_tab, self.camera_tab, self.server_tab):
            tab.configure(fg_color=C["surface"])
        self._build_detection_tab(self.detection_tab)
        self._build_field_tab(self.field_tab)
        self._build_camera_tab(self.camera_tab)
        self._build_server_tab(self.server_tab)
        self._on_tab_changed()

    def _on_tab_changed(self, _value: str | None = None) -> None:
        """Активная вкладка — белая надпись на синем, остальные — темные."""
        try:
            current = self.tabview.get()
            if current == self._last_tab:
                return
            self._last_tab = current
            for name, btn in self.tabview._segmented_button._buttons_dict.items():
                btn.configure(
                    text_color=C["on_primary"] if name == current else C["text_secondary"]
                )
        except (AttributeError, KeyError, tk.TclError):
            pass

    def _build_statusbar(self) -> None:
        bar = ctk.CTkFrame(
            self.root, fg_color=C["surface"], corner_radius=0, height=30,
            border_width=1, border_color=C["border"],
        )
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.lbl_view_info = ctk.CTkLabel(
            bar, text="", font=self.fonts.caption, text_color=C["text_secondary"],
        )
        self.lbl_view_info.pack(side="left", padx=16)
        self.lbl_mouse_mm = ctk.CTkLabel(
            bar, text="", font=self.fonts.caption, text_color=C["text_secondary"],
        )
        self.lbl_mouse_mm.pack(side="right", padx=16)
        self.status_chips = (self.chip_camera, self.chip_field, self.chip_server)
        self.refresh_calibration_labels()

    # ------------------------- вкладка «Детекция» -------------------------

    def _build_detection_tab(self, tab) -> None:
        params = self.detector.params
        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=12)

        grid = ctk.CTkFrame(body, fg_color="transparent")
        grid.pack(fill="x")
        self.param_row(grid, 0, "Мин. площадь, px²", "min_area_px", 0, 200000, 50,
                       tooltip="Отсечение мусора: контуры меньше площади отбрасываются")
        self.param_row(grid, 1, "Макс. площадь, px² (0 = нет)", "max_area_px", 0, 500000, 100)
        self.param_row(grid, 2, "Мин. отношение сторон", "aspect_min", 0.05, 1.0, 0.05,
                       tooltip="min(w,h)/max(w,h) прямоугольника; кубик ≈ 1")
        self.param_row(grid, 3, "Макс. отношение сторон", "aspect_max", 0.05, 1.0, 0.05)
        self.param_row(grid, 4, "Мин. заливка контура", "fill_min", 0.05, 1.0, 0.05,
                       tooltip="Площадь контура / площадь minAreaRect")
        self.param_row(grid, 5, "Симметрия угла, °", "angle_symmetry_deg", 1, 360, 5,
                       tooltip="90 → угол детали в диапазоне ±45°")

        sep1 = ctk.CTkFrame(body, height=1, fg_color=C["border"])
        sep1.pack(fill="x", pady=10)

        switches = ctk.CTkFrame(body, fg_color="transparent")
        switches.pack(fill="x")
        switches.columnconfigure(0, weight=1)
        adaptive_var = tk.BooleanVar(value=params.adaptive)
        adaptive_var.trace_add("write", lambda *_a: setattr(params, "adaptive", adaptive_var.get()))
        self.switch_row(switches, 0, "Адаптивный порог (иначе Оцу)", adaptive_var,
                        lambda: setattr(params, "adaptive", adaptive_var.get()))
        polarity_var = tk.StringVar(value=params.polarity)
        polarity_var.trace_add("write", lambda *_a: setattr(params, "polarity", polarity_var.get()))
        pol_row = ctk.CTkFrame(switches, fg_color="transparent")
        pol_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        ctk.CTkLabel(pol_row, text="Что считать объектом", font=self.fonts.body,
                     text_color=C["text"]).pack(side="left")
        self.combo(pol_row, polarity_var, ("auto", "dark", "bright"), 130).pack(side="right")
        Tooltip(pol_row, "auto — меньшая часть кадра; dark/bright — принудительно")

        blur_close = ctk.CTkFrame(body, fg_color="transparent")
        blur_close.pack(fill="x")
        self.param_row(blur_close, 0, "Ядро GaussianBlur", "blur_ksize", 1, 21, 2, as_int=True)
        self.param_row(blur_close, 1, "Ядро морфозакрытия", "close_ksize", 1, 21, 2, as_int=True)

        sep2 = ctk.CTkFrame(body, height=1, fg_color=C["border"])
        sep2.pack(fill="x", pady=10)

        hsv_var = tk.BooleanVar(value=params.use_hsv)

        def apply_hsv(*_a) -> None:
            params.use_hsv = hsv_var.get()

        hsv_switch = ctk.CTkSwitch(
            body, text="Цветная сегментация HSV", variable=hsv_var,
            command=apply_hsv, font=self.fonts.body, text_color=C["text"],
            progress_color=C["primary"], button_color="#B9C4D0",
            button_hover_color=C["primary"],
        )
        hsv_switch.pack(anchor="w", pady=(0, 6))
        Tooltip(hsv_switch, "Диапазон H,S,V вместо яркостной бинаризации Оцу")

        hsv_grid = ctk.CTkFrame(body, fg_color="transparent")
        hsv_grid.pack(fill="x", pady=(2, 0))
        hsv_grid.columnconfigure((1, 2, 3), weight=1, uniform="hsv")
        for col_i, channel in enumerate(("H", "S", "V")):
            ctk.CTkLabel(
                hsv_grid, text=f"{channel} ({'0–179' if channel == 'H' else '0–255'})",
                font=self.fonts.caption, text_color=C["text_secondary"],
            ).grid(row=0, column=1 + col_i, pady=(0, 2))
        limit_map = {"H": 179, "S": 255, "V": 255}
        self._hsv_spins: dict[str, tk.StringVar] = {}
        for row_i, (title, key) in enumerate(
            (("Нижняя граница", "hsv_low"), ("Верхняя граница", "hsv_high")), start=1
        ):
            ctk.CTkLabel(
                hsv_grid, text=title, anchor="w", font=self.fonts.body,
                text_color=C["text"],
            ).grid(row=row_i, column=0, sticky="w", pady=3, padx=(0, 8))
            current = getattr(params, key)
            for col_i, channel in enumerate(("H", "S", "V")):
                var = tk.StringVar(value=str(current[col_i]))
                self._hsv_spins[f"{key}{channel}"] = var

                def make_apply(k=key):
                    def apply(*_a) -> None:
                        try:
                            value = tuple(
                                max(0, min(limit_map[ch], int(var.get())))
                                for ch, var in (
                                    ("H", self._hsv_spins[f"{k}H"]),
                                    ("S", self._hsv_spins[f"{k}S"]),
                                    ("V", self._hsv_spins[f"{k}V"]),
                                )
                            )
                        except ValueError:
                            return
                        setattr(params, k, value)

                    return apply

                spin = ttk.Spinbox(
                    hsv_grid, textvariable=var, from_=0,
                    to=limit_map[channel], width=6,
                    style="Material.TSpinbox", font=(FONT_FAMILY, 12),
                    justify="center",
                )
                spin.grid(row=row_i, column=1 + col_i, sticky="ew", padx=3, pady=3)
                var.trace_add("write", make_apply())

        hint = (
            "Изменения применяются мгновенно и действуют также на работающий "
            "сервер — детектор общий."
        )
        ctk.CTkLabel(
            body, text=hint, wraplength=370, justify="left", anchor="w",
            font=self.fonts.caption, text_color=C["text_secondary"],
        ).pack(anchor="w", fill="x", pady=(12, 6))

        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.pack(fill="x")
        self.button(actions, "Сбросить параметры", self.reset_params, "ghost").pack(anchor="e")

    def reset_params(self) -> None:
        defaults = VisionParams()
        self.detector.params.__dict__.update(vars(defaults))
        for child in self.detection_tab.winfo_children():
            child.destroy()
        self._build_detection_tab(self.detection_tab)
        self.log("Параметры детекции сброшены к значениям по умолчанию.")

    # --------------------------- вкладка «Поле» ---------------------------

    def _build_field_tab(self, tab) -> None:
        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=12)

        info = (
            "1. Разметьте ≥ 4 реперных точек на столе робота.\n"
            "2. Коснитесь каждой острием инструмента, считайте X,Y с пульта.\n"
            "3. Нажмите «Добавить точку», кликните её в кадре и введите координаты.\n"
            "4. «Рассчитать и сохранить» — сервер начнет отдавать мм.\n\n"
            "Ctrl+клик добавляет точку из любой вкладки."
        )
        ctk.CTkLabel(
            body, text=info, wraplength=380, justify="left", anchor="w",
            font=self.fonts.caption, text_color=C["text_secondary"],
        ).pack(fill="x", pady=(0, 10))

        self.verify_var = tk.BooleanVar(value=False)
        verify_sw = ctk.CTkSwitch(
            body, text="Проверка: курсор над кадром → координаты в мм",
            variable=self.verify_var, font=self.fonts.body,
            text_color=C["text"], progress_color=C["primary"],
            button_color="#B9C4D0", button_hover_color=C["primary"],
        )
        verify_sw.pack(anchor="w", pady=(0, 10))

        self.btn_add_point_holder = ctk.CTkFrame(body, fg_color="transparent")
        self.btn_add_point_holder.pack(fill="x")
        self.btn_add_point_holder.columnconfigure(0, weight=1, uniform="fieldbtns")
        self.btn_add_point_holder.columnconfigure(1, weight=1, uniform="fieldbtns")
        self.btn_add_point = self.button(
            self.btn_add_point_holder, "Добавить точку кликом",
            self.toggle_add_points_mode, "primary",
        )
        self.btn_add_point.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        undo_btn = self.button(self.btn_add_point_holder, "Отменить последнюю",
                               self.undo_field_point, "outline")
        undo_btn.grid(row=0, column=1, sticky="ew")

        clear_btn = self.button(body, "Очистить все точки", self.clear_field_points, "ghost")
        clear_btn.pack(anchor="e", pady=(2, 6))

        tree_wrap = ctk.CTkFrame(
            body, fg_color=C["surface"], corner_radius=8,
            border_width=1, border_color=C["border"],
        )
        tree_wrap.pack(fill="both", expand=True, pady=(0, 10))
        columns = ("n", "u", "v", "x", "y")
        headers = ("№", "u, px", "v, px", "X, мм", "Y, мм")
        widths = (50, 100, 100, 150, 150)
        self.points_tree = ttk.Treeview(
            tree_wrap, columns=columns, show="headings", height=7,
            style="Card.Treeview",
        )
        for col, header, width in zip(columns, headers, widths):
            anchor = "center" if col == "n" else "e"
            self.points_tree.heading(col, text=header)
            self.points_tree.column(
                col, width=width, anchor=anchor, stretch=True, minwidth=60
            )
        scroll = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.points_tree.yview)
        self.points_tree.configure(yscrollcommand=scroll.set)
        self.points_tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        scroll.pack(side="right", fill="y", padx=(0, 2), pady=6)
        self.points_tree.tag_configure("odd", background=C["surface"])
        self.points_tree.tag_configure("even", background=C["row_alt"])

        method_row = ctk.CTkFrame(body, fg_color="transparent")
        method_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(method_row, text="Преобразование", font=self.fonts.body,
                     text_color=C["text"]).pack(side="left")
        self.method_var = tk.StringVar(value="Гомография")
        self.combo(method_row, self.method_var, ("Гомография", "Аффинное"), 170).pack(side="right")

        out_row = ctk.CTkFrame(body, fg_color="transparent")
        out_row.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(out_row, text="Файл", font=self.fonts.body,
                     text_color=C["text"]).pack(side="left", padx=(0, 8))
        self.field_output_var = tk.StringVar(value=str(DEFAULT_FIELD_CALIBRATION))
        field_entry = ctk.CTkEntry(out_row, textvariable=self.field_output_var, height=ENTRY_H,
                                   font=self.fonts.caption, border_color=C["border"],
                                   fg_color=C["surface"], text_color=C["text"])
        field_entry.pack(side="left", fill="x", expand=True)
        self.entry_hover(field_entry)
        browse = self.button(out_row, "…", self.browse_field_output, "outline", ENTRY_H)
        browse.configure(width=36)
        browse.pack(side="left", padx=(6, 0))

        self.button(body, "Рассчитать и сохранить", self.compute_field,
                    "primary", BTN_H_MAIN).pack(fill="x")

    def browse_field_output(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".npz", filetypes=[("NumPy archive", "*.npz")]
        )
        if path:
            self.field_output_var.set(path)

    # --------------------- вкладка «Калибровка камеры» --------------------

    def _build_camera_tab(self, tab) -> None:
        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=12)

        if not self.aruco_available:
            warn_card = ctk.CTkFrame(body, fg_color=C["warning_bg"], corner_radius=8)
            warn_card.pack(fill="x")
            ctk.CTkLabel(
                warn_card, justify="left", anchor="w",
                text=("В установленной сборке OpenCV нет модуля aruco.\n"
                      "Установите opencv-contrib-python, чтобы использовать\n"
                      "внутреннюю калибровку из этого окна."),
                wraplength=360, font=self.fonts.body,
                text_color=C["warning"],
            ).pack(padx=12, pady=10)
            return

        board_info = (
            f"Доска {camcal.BOARD_COLUMNS}×{camcal.BOARD_ROWS}, маркер "
            f"{camcal.MARKER_SIZE_MM:g} мм, промежуток {camcal.MARKER_SEPARATION_MM:g} мм,\n"
            f"{camcal.DICTIONARY_NAME}. Показывайте доску под разными углами;\n"
            "для надежности нужно 15–20 положений."
        )
        ctk.CTkLabel(
            body, text=board_info, wraplength=380, justify="left", anchor="w",
            font=self.fonts.caption, text_color=C["text_secondary"],
        ).pack(fill="x", pady=(0, 10))

        spins = ctk.CTkFrame(body, fg_color="transparent")
        spins.pack(fill="x")
        spins.columnconfigure(1, weight=1)
        ctk.CTkLabel(spins, text="Цель по ракурсам", font=self.fonts.body,
                     text_color=C["text"]).grid(row=0, column=0, sticky="w", pady=3)
        self.target_samples_var = tk.IntVar(value=15)
        ttk.Spinbox(spins, from_=3, to=40, textvariable=self.target_samples_var,
                    width=SPIN_W // 8, style="Material.TSpinbox",
                    font=(FONT_FAMILY, 12), justify="right").grid(row=0, column=1, sticky="e", pady=3)
        ctk.CTkLabel(spins, text="Мин. видимых маркеров", font=self.fonts.body,
                     text_color=C["text"]).grid(row=1, column=0, sticky="w", pady=3)
        self.min_markers_var = tk.IntVar(value=6)
        ttk.Spinbox(spins, from_=1, to=100, textvariable=self.min_markers_var,
                    width=SPIN_W // 8, style="Material.TSpinbox",
                    font=(FONT_FAMILY, 12), justify="right").grid(row=1, column=1, sticky="e", pady=3)

        counters = ctk.CTkFrame(body, fg_color="transparent")
        counters.pack(fill="x", pady=(10, 4))
        self.lbl_marker_count = ctk.CTkLabel(
            counters, text="Видимых маркеров доски: –", anchor="w",
            font=self.fonts.body_bold, text_color=C["text_secondary"],
        )
        self.lbl_marker_count.pack(fill="x")
        self.lbl_views = ctk.CTkLabel(
            counters, text="Сохранено ракурсов: 0", anchor="w",
            font=self.fonts.body_bold, text_color=C["text_secondary"],
        )
        self.lbl_views.pack(fill="x")

        self.button_row(
            body,
            ("Добавить ракурс", self.capture_camera_view, "primary"),
            ("Сбросить", self.reset_camera_views, "outline"),
        )

        out_row = ctk.CTkFrame(body, fg_color="transparent")
        out_row.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(out_row, text="Файлы", font=self.fonts.body,
                     text_color=C["text"]).pack(side="left", padx=(0, 8))
        self.cam_output_var = tk.StringVar(value=str(BASE_DIR / "camera_calibration"))
        cam_entry = ctk.CTkEntry(out_row, textvariable=self.cam_output_var, height=ENTRY_H,
                                 font=self.fonts.caption, border_color=C["border"],
                                 fg_color=C["surface"], text_color=C["text"])
        cam_entry.pack(side="left", fill="x", expand=True)
        self.entry_hover(cam_entry)

        self.btn_run_camcal = self.button(
            body, "Рассчитать и сохранить", self.run_camera_calibration,
            "primary", BTN_H_MAIN,
        )
        self.btn_run_camcal.pack(fill="x")
        self.set_button_state(self.btn_run_camcal, False)

    # ------------------------ вкладка «Сервер» ----------------------------

    def _build_server_tab(self, tab) -> None:
        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=12)
        body.rowconfigure(5, weight=1)
        body.columnconfigure(0, weight=1)

        fields = ctk.CTkFrame(body, fg_color="transparent")
        fields.grid(row=0, column=0, sticky="ew")
        fields.columnconfigure(1, weight=1)

        def labeled_entry(row, label, variable, width=170, is_spin=None, **kw):
            ctk.CTkLabel(fields, text=label, anchor="w", font=self.fonts.body,
                         text_color=C["text"]).grid(row=row, column=0, sticky="w", pady=3)
            widget = ctk.CTkEntry(fields, textvariable=variable, width=width,
                                  height=ENTRY_H, justify="right",
                                  border_color=C["border"], fg_color=C["surface"],
                                  text_color=C["text"], font=self.fonts.body, **kw)
            widget.grid(row=row, column=1, sticky="e", pady=3)
            return widget

        self.host_var = tk.StringVar(value="0.0.0.0")
        self.host_entry = labeled_entry(0, "Хост", self.host_var)
        self.port_var = tk.IntVar(value=59152)
        self.port_entry = labeled_entry(1, "Порт", self.port_var)
        self.part_height_var = tk.DoubleVar(value=15.0)
        self.height_entry = labeled_entry(2, "Высота детали Z, мм", self.part_height_var)
        for entry in (self.host_entry, self.port_entry, self.height_entry):
            self.entry_hover(entry)
        Tooltip(self.host_entry, "IP-адрес ПК, который укажете в VisionConfig.xml контроллера")
        Tooltip(self.port_entry, "Порт EKI по ТЗ — 59152")
        Tooltip(self.height_entry, "Высота детали Z: камера дает только X, Y, угол")

        buttons = ctk.CTkFrame(body, fg_color="transparent")
        buttons.grid(row=1, column=0, sticky="ew", pady=10)
        buttons.columnconfigure(0, weight=1, uniform="srv")
        buttons.columnconfigure(1, weight=1, uniform="srv")
        buttons.columnconfigure(2, weight=1, uniform="srv")
        self.btn_server_start = self.button(buttons, "Запустить", self.start_server, "primary")
        self.btn_server_start.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn_server_stop = self.button(buttons, "Остановить", self.stop_server, "outline")
        self.btn_server_stop.configure(state="disabled", width=118)
        self.btn_server_stop.grid(row=0, column=1, sticky="ew", padx=4)
        self.btn_test_capture = self.button(buttons, "Тестовый захват", self.manual_capture, "tonal")
        self.btn_test_capture.grid(row=0, column=2, sticky="ew", padx=(4, 0))
        Tooltip(self.btn_test_capture, "Тот же путь обработки, что и по Trigger робота — без робота")

        self.lbl_server_state = ctk.CTkLabel(
            body, text="Сервер остановлен", anchor="w",
            font=self.fonts.body_bold, text_color=C["text_secondary"],
        )
        self.lbl_server_state.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        journal_header = ctk.CTkFrame(body, fg_color=C["primary"], corner_radius=8, height=32)
        journal_header.grid(row=4, column=0, sticky="ew")
        journal_header.pack_propagate(False)
        ctk.CTkLabel(
            journal_header, text="ЖУРНАЛ СОБЫТИЙ И ОБМЕНА XML".upper(),
            anchor="w", font=self.fonts.section, text_color=C["on_primary"],
        ).pack(side="left", padx=12)
        clear_btn = self.button(journal_header, "Очистить", self.clear_log, "ghost", 26)
        clear_btn.configure(
            fg_color="transparent", border_color="#9EC2EE",
            text_color=C["on_primary"], hover_color=C["primary_hover"],
        )
        clear_btn.pack(side="right", padx=6)

        self.log_text = ctk.CTkTextbox(
            body, fg_color=C["log_bg"], text_color=C["log_text"],
            font=ctk.CTkFont("Consolas", 11), corner_radius=8, wrap="word",
        )
        self.log_text.grid(row=5, column=0, sticky="nsew", pady=(6, 0))
        try:
            self.log_text.tag_config("xml", foreground="#82B1FF")
            self.log_text.tag_config("dim", foreground="#78849A")
            self.log_text.tag_config("warn", foreground="#FFC46B")
        except Exception:
            pass

    def clear_log(self) -> None:
        self.log_text.delete("1.0", "end")

    # ------------------------------------------------------------------
    # Журнал и события
    # ------------------------------------------------------------------

    def log(self, message: str, tag: str | None = None) -> None:
        """Потокобезопасно пишет строку в журнал (из любого потока)."""
        self.events.put(("log", (message, tag)))

    def _append_log(self, message: str, tag: str | None) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{stamp}] ", "dim")
        self.log_text.insert("end", message + "\n", tag or ())
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > LOG_MAX_LINES:
            self.log_text.delete("1.0", f"{line_count - LOG_MAX_LINES}.0")
        self.log_text.see("end")

    # ------------------------------------------------------------------
    # Источник кадров и конвейер
    # ------------------------------------------------------------------

    def on_connect_clicked(self) -> None:
        self.load_source(self.source_var.get())

    def on_disconnect_clicked(self) -> None:
        self.unload_source()

    def load_source(self, spec: str) -> bool:
        spec = spec.strip()
        if not spec:
            messagebox.showerror("Источник", "Укажите индекс камеры или путь к файлу.")
            return False
        source_spec: int | str = int(spec) if spec.isdigit() else spec

        if self.source is not None:
            self.unload_source()

        try:
            shared = SharedSource(source_spec)
            shared.warm_up()
        except Exception as exc:
            messagebox.showerror(
                "Источник",
                f"Не удалось открыть источник {spec!r}:\n{exc}\n\n"
                "Возможно, камера занята другой программой.",
            )
            return False

        self.source = shared
        self.capture_running = True
        self.capture_thread = threading.Thread(
            target=self._pipeline_loop, name="pipeline", daemon=True
        )
        self.capture_thread.start()
        self.set_button_state(self.btn_connect, False)
        self.set_button_state(self.btn_disconnect, True)
        kind = "снимок" if shared.is_static else "поток"
        self.lbl_source_state.configure(text=f"Источник: {spec} ({kind})")
        self.log(f"Источник подключен: {spec} ({kind})")
        return True

    def unload_source(self) -> None:
        if self.server is not None:
            self.stop_server(wait=True)
        if self.capture_running:
            self.capture_running = False
            if self.capture_thread is not None:
                self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        if self.source is not None:
            self.source.release()
            self.source = None
        with self.state_lock:
            self.latest_frame = None
            self.latest_parts = []
        self.board_corners = None
        self.board_ids = None
        self.marker_count = 0
        self.flash_until = 0.0
        self.flash_frame = None
        self.set_button_state(self.btn_connect, True)
        self.set_button_state(self.btn_disconnect, False)
        self.lbl_source_state.configure(text="Источник отключен")
        self.fps = 0.0

    def _pipeline_loop(self) -> None:
        while self.capture_running and self.source is not None:
            started = time.perf_counter()
            try:
                frame = self.source.read()
            except Exception as exc:
                self.log(f"Ошибка источника кадров: {exc}", tag="warn")
                self.events.put(("source-failed", str(exc)))
                break
            try:
                parts = self.detector.detect(frame)
            except Exception as exc:
                self.log(f"Ошибка детекции: {exc}", tag="warn")
                parts = []
            with self.state_lock:
                now = time.time()
                if self.latest_stamp:
                    dt = now - self.latest_stamp
                    if dt > 0:
                        instant = 1.0 / dt
                        self.fps = (
                            instant if self.fps == 0.0 else self.fps * 0.85 + instant * 0.15
                        )
                self.latest_stamp = now
                self.latest_frame = frame
                self.latest_parts = parts
            if self.source.is_static:
                remaining = 0.05 - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)

    # ------------------------------------------------------------------
    # Цикл обновления UI
    # ------------------------------------------------------------------

    def _ui_tick(self) -> None:
        self._tick_count += 1
        self._drain_events()
        self._update_chips()
        self._on_tab_changed()  # на случай программной смены вкладки
        self._refresh_camera_scan()
        self._render_latest()
        self.root.after(POLL_MS, self._ui_tick)

    def _drain_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(*payload)
            elif kind == "source-failed":
                self.unload_source()
                self.lbl_source_state.configure(text=f"Источник остановлен: {payload}")
            elif kind == "server-stopped":
                self._mark_server_stopped(payload)
            elif kind == "capture-flash":
                frame, parts = payload
                self.flash_frame = draw_parts(frame, parts)
                self.flash_until = time.monotonic() + 1.5
                self.server_busy_until = time.monotonic() + 2.0
            elif kind == "server-status":
                self.lbl_server_state.configure(text=payload)

    def _update_chips(self) -> None:
        try:
            port = int(self.port_var.get()) if self.server is not None else 0
        except (tk.TclError, ValueError):
            port = 0
        state = (
            self.camera_calibrated,
            self.field is not None,
            self.server is not None,
            time.monotonic() < self.server_busy_until,
            port,
        )
        if state == self._chips_state:
            return
        self._chips_state = state
        cam_ok, field_ok, running, busy, port = state
        self.set_chip(
            self.chip_camera,
            "● Камера калибрована" if cam_ok else "● Камера не калибрована",
            C["success"] if cam_ok else C["warning"],
            C["success_bg"] if cam_ok else C["warning_bg"],
        )
        self.set_chip(
            self.chip_field,
            "● Поле калибровано" if field_ok else "● Поле не калибровано",
            C["success"] if field_ok else C["warning"],
            C["success_bg"] if field_ok else C["warning_bg"],
        )
        if running and busy:
            self.set_chip(self.chip_server, "● Сервер: отвечает", C["success"], C["success_bg"])
        elif running:
            self.set_chip(
                self.chip_server, f"● Сервер :{port}", C["warning"], C["warning_bg"]
            )
        else:
            self.set_chip(self.chip_server, "● Сервер остановлен", C["text_secondary"], "#E7EBF1")

    def refresh_calibration_labels(self) -> None:
        """Совместимость: статусы теперь показывают чипы appbar."""
        self._chips_state = ()

    def _refresh_camera_scan(self) -> None:
        """Детекция ArUco для вкладки калибровки камеры (не каждый тик)."""
        if (
            not self.aruco_available
            or self.tabview.get() != self.TAB_CAMERA
            or self._tick_count % ARUCO_EVERY
        ):
            return
        with self.state_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None or not self._ensure_board():
            return
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids = camcal.detect_markers(
            self.aruco, self._detector_aruco, None, self.dictionary, gray
        )
        board_corners, board_ids = camcal.select_board_markers(corners, ids, self._point_map)
        self.board_corners = board_corners
        self.board_ids = board_ids
        self.marker_count = 0 if board_ids is None else len(board_ids)
        try:
            min_markers = self.min_markers_var.get()
        except (tk.TclError, ValueError):
            return
        enough = self.marker_count >= min_markers
        color = C["success"] if enough else C["warning"]
        self.lbl_marker_count.configure(
            text=f"Видимых маркеров доски: {self.marker_count}", text_color=color
        )

    def _render_latest(self) -> None:
        with self.state_lock:
            frame = self.latest_frame
            parts = list(self.latest_parts)
        display_override = None
        if time.monotonic() < self.flash_until and self.flash_frame is not None:
            display_override = self.flash_frame

        canvas_w = max(self.canvas.winfo_width(), 320)
        canvas_h = max(self.canvas.winfo_height(), 240)

        if frame is None and display_override is None:
            self.canvas.delete("all")
            self._photo = None
            self.view_params = (1.0, 0.0, 0.0)
            hint = ("Нет сигнала\nПодключите источник кадров"
                    if self.source is not None
                    else "Нет сигнала\nПодключите источник кадров\n(камера, видео или снимок)")
            self.canvas.create_text(
                canvas_w // 2, canvas_h // 2, text=hint, fill="#8FA1B3",
                font=("Segoe UI", 14), justify="center",
            )
            self.lbl_view_info.configure(text="")
            self._update_parts_table([])
            return

        self._update_parts_table(parts)

        rendered = display_override if display_override is not None else \
            self._compose_display(frame, parts)

        fh, fw = rendered.shape[:2]
        scale = min(canvas_w / fw, canvas_h / fh)
        out_w, out_h = max(int(fw * scale), 1), max(int(fh * scale), 1)
        resized = cv.resize(rendered, (out_w, out_h), interpolation=cv.INTER_AREA)
        rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        ox = (canvas_w - out_w) // 2
        oy = (canvas_h - out_h) // 2
        self.view_params = (scale, ox, oy)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, image=self._photo, anchor="nw")
        if display_override is not None:
            self.canvas.create_text(
                ox + 14, oy + 16, text="ЗАХВАТ", fill="#FFD34D",
                font=("Segoe UI", 12, "bold"), anchor="nw",
            )
        self.lbl_view_info.configure(
            text=f"{fw}×{fh} · {self.fps:.1f} fps · деталей: {len(parts)}"
        )

    @staticmethod
    def _draw_label(img: np.ndarray, text: str, x: int, y: int,
                    color: tuple = (255, 255, 255)) -> None:
        """Подпись с плотной подложкой, не выходит за границы кадра."""
        (tw, th), base = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        fh, fw = img.shape[:2]
        left = min(max(x, 0), max(fw - tw - 8, 0))
        top = min(max(y - th - base - 4, 0), max(fh - th - base - 6, 0))
        cv.rectangle(img, (left, top), (left + tw + 8, top + th + base + 6),
                     (22, 26, 30), -1, cv.LINE_AA)
        cv.putText(img, text, (left + 4, top + th + base + 1),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv.LINE_AA)

    def _compose_display(self, frame: np.ndarray, parts: list) -> np.ndarray:
        display = draw_parts(frame, parts) if parts else frame.copy()

        if self.board_corners is not None and self.tabview.get() == self.TAB_CAMERA:
            try:
                cv.aruco.drawDetectedMarkers(display, self.board_corners, self.board_ids)
            except Exception:
                pass

        # реперные точки поля
        for index, (u, v) in enumerate(self.pixel_points):
            pos = (int(u), int(v))
            cv.circle(display, pos, 6, (0, 0, 255), -1, cv.LINE_AA)
            cv.circle(display, pos, 6, (255, 255, 255), 1, cv.LINE_AA)
            label = f"{index + 1}"
            if index < len(self.robot_points):
                rx, ry = self.robot_points[index]
                label += f": {rx:.1f}, {ry:.1f}"
            self._draw_label(display, label, pos[0] + 10, pos[1] - 10)

        # курсор: перекрестие и режимы
        mouse = self.mouse_frame
        if mouse is not None:
            cx, cy = mouse
            color = (80, 210, 255)
            cv.line(display, (cx - 14, cy), (cx + 14, cy), color, 1, cv.LINE_AA)
            cv.line(display, (cx, cy - 14), (cx, cy + 14), color, 1, cv.LINE_AA)
            if self.verify_var.get() and self.field is not None:
                x, y = transform_pixel_to_robot(self.field.matrix, cx, cy)
                self._draw_label(display, f"X={x:.1f} Y={y:.1f} mm",
                                 cx + 12, cy + 26, color=(150, 240, 150))
                self.lbl_mouse_mm.configure(
                    text=f"Курсор: X={x:.1f}, Y={y:.1f} мм",
                    text_color=C["text_secondary"],
                )
            else:
                self.lbl_mouse_mm.configure(text="")
        elif self.lbl_mouse_mm.cget("text"):
            self.lbl_mouse_mm.configure(text="")
        return display

    def _update_parts_table(self, parts: list) -> None:
        rows = []
        field = self.field
        symmetry = self.detector.params.angle_symmetry_deg
        for index, part in enumerate(parts, start=1):
            u, v = part.center
            if field is not None:
                x, y = transform_pixel_to_robot(field.matrix, u, v)
                angle = transform_angle_deg(field.matrix, u, v, part.angle_deg)
                angle = normalize_angle(angle, symmetry)
                cells_x, cells_y, cells_a = f"{x:.1f}", f"{y:.1f}", f"{angle:+.1f}"
            else:
                cells_x = cells_y = "—"
                cells_a = f"{part.angle_deg:+.1f}"
            rows.append((
                str(index), f"{u:.0f}", f"{v:.0f}",
                cells_x, cells_y, cells_a, f"{part.area_px:.0f}",
            ))
        existing = set(self.parts_tree.get_children())
        wanted = {str(i) for i in range(1, len(rows) + 1)}
        if existing == wanted and all(
            tuple(self.parts_tree.item(iid, "values")) == values
            for iid, values in zip(sorted(existing, key=int), rows)
        ):
            return
        self.parts_tree.delete(*self.parts_tree.get_children())
        for i, values in enumerate(rows):
            tag = "even" if i % 2 else "odd"
            self.parts_tree.insert("", "end", iid=str(values[0]), values=values, tags=(tag,))

    def copy_selected_part(self, full: bool = False) -> None:
        selection = self.parts_tree.selection()
        if not selection:
            return
        values = self.parts_tree.item(selection[0], "values")
        text = ",".join(values) if full else f"{values[3]}, {values[4]}, {values[5]}"
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.lbl_view_info.configure(text=f"Скопировано: {text}")

    def _show_parts_menu(self, event) -> None:
        iid = self.parts_tree.identify_row(event.y)
        if iid:
            self.parts_tree.selection_set(iid)
            self.parts_menu.tk_popup(event.x_root, event.y_root)

    # ------------------------------------------------------------------
    # Мышь на кадре
    # ------------------------------------------------------------------

    def _canvas_to_frame(self, event) -> tuple[int, int] | None:
        with self.state_lock:
            frame = self.latest_frame
        if frame is None:
            return None
        scale, ox, oy = self.view_params
        u = int((event.x - ox) / scale)
        v = int((event.y - oy) / scale)
        fh, fw = frame.shape[:2]
        if 0 <= u < fw and 0 <= v < fh:
            return u, v
        return None

    def _on_canvas_motion(self, event) -> None:
        self.mouse_frame = self._canvas_to_frame(event)

    def _on_canvas_click(self, event) -> None:
        point = self._canvas_to_frame(event)
        if point is None:
            return
        if not (self.add_points_mode or self._is_control_click(event)):
            return
        dialog = RobotPointDialog(
            self.root, len(self.pixel_points) + 1, point,
            initial=getattr(self, "last_robot_text", ""),
        )
        if dialog.result is None:
            return
        self.last_robot_text = f"{dialog.result[0]:g},{dialog.result[1]:g}"
        self.add_field_point(point, dialog.result)
        if self._is_control_click(event):
            self.set_add_points_mode(False)

    @staticmethod
    def _is_control_click(event) -> bool:
        return bool(getattr(event, "state", 0) & 0x0004)

    # ------------------------------------------------------------------
    # Калибровка поля
    # ------------------------------------------------------------------

    def toggle_add_points_mode(self) -> None:
        self.set_add_points_mode(not self.add_points_mode)

    def set_add_points_mode(self, enabled: bool) -> None:
        self.add_points_mode = enabled
        self.btn_add_point.configure(
            text="Добавление точек ВКЛ — кликайте кадр (Esc)" if enabled
            else "Добавить точку кликом",
            fg_color=C["success"] if enabled else C["primary"],
            hover_color="#166527" if enabled else C["primary_hover"],
        )
        self.canvas.configure(cursor="crosshair" if enabled else "")

    def add_field_point(
        self, pixel: tuple[float, float], robot: tuple[float, float]
    ) -> None:
        self.pixel_points.append(pixel)
        self.robot_points.append(robot)
        index = len(self.pixel_points)
        tag = "even" if (index - 1) % 2 else "odd"
        self.points_tree.insert(
            "", "end",
            values=(index, f"{pixel[0]:.0f}", f"{pixel[1]:.0f}",
                    f"{robot[0]:.2f}", f"{robot[1]:.2f}"),
            tags=(tag,),
        )
        self.log(
            f"Точка {index}: пиксель ({pixel[0]:.0f}, {pixel[1]:.0f}) → "
            f"робот ({robot[0]:.1f}, {robot[1]:.1f}) мм"
        )

    def undo_field_point(self) -> None:
        if not self.pixel_points:
            return
        self.pixel_points.pop()
        self.robot_points.pop()
        children = self.points_tree.get_children()
        if children:
            self.points_tree.delete(children[-1])

    def clear_field_points(self) -> None:
        if not self.pixel_points:
            return
        if not messagebox.askyesno("Калибровка поля", "Удалить все реперные точки?"):
            return
        self.pixel_points.clear()
        self.robot_points.clear()
        self.points_tree.delete(*self.points_tree.get_children())

    def compute_field(self) -> None:
        count = len(self.pixel_points)
        if count < 4:
            messagebox.showwarning(
                "Калибровка поля", f"Нужно минимум 4 точки, сейчас {count}."
            )
            return
        if count != len(self.robot_points):
            messagebox.showerror("Калибровка поля", "Число пикселей и координат не совпадает.")
            return
        method = "affine" if self.method_var.get() == "Аффинное" else "homography"
        output = Path(self.field_output_var.get())
        try:
            matrix = compute_field_matrix(self.pixel_points, self.robot_points, method)
            residuals = calibration_residuals(matrix, np.asarray(self.pixel_points),
                                              np.asarray(self.robot_points))
            field = FieldCalibration(
                matrix=matrix,
                pixel_points=np.asarray(self.pixel_points, dtype=np.float64),
                robot_points=np.asarray(self.robot_points, dtype=np.float64),
                residual_mm=residuals,
            )
            save_field_calibration(output, field)
        except Exception as exc:
            messagebox.showerror("Калибровка поля", f"Не удалось рассчитать:\n{exc}")
            return
        self.field = field
        per_point = "; ".join(f"{i}: {e:.2f}" for i, e in enumerate(residuals, 1))
        self.log(
            f"Поле ({method}): средняя ошибка {residuals.mean():.2f} мм, "
            f"максимум {residuals.max():.2f} мм [{per_point}]"
        )
        messagebox.showinfo(
            "Калибровка поля",
            f"Сохранено: {output}\n"
            f"Средняя ошибка {residuals.mean():.2f} мм, "
            f"максимальная {residuals.max():.2f} мм.",
        )
        self._chips_state = ()  # перерисовать чипы

    # ------------------------------------------------------------------
    # Калибровка камеры
    # ------------------------------------------------------------------

    def _ensure_board(self) -> bool:
        if self._board is not None:
            return True
        try:
            self.aruco = camcal.require_aruco()
            dictionary_id = getattr(self.aruco, camcal.DICTIONARY_NAME)
            self.dictionary = self.aruco.getPredefinedDictionary(dictionary_id)
            self._board = camcal.create_board(self.aruco, self.dictionary)
            detector, _params = camcal.create_detector(self.aruco, self.dictionary)
            self._detector_aruco = detector
            self._point_map = camcal.board_point_map(self._board)
            return True
        except Exception as exc:
            self.log(f"ArUco недоступен: {exc}", tag="warn")
            self.aruco_available = False
            return False

    def capture_camera_view(self) -> None:
        if not self._ensure_board():
            return
        with self.state_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            messagebox.showwarning("Калибровка", "Нет кадра — подключите источник.")
            return
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids = camcal.detect_markers(
            self.aruco, self._detector_aruco, None, self.dictionary, gray
        )
        board_corners, board_ids = camcal.select_board_markers(corners, ids, self._point_map)
        needed = self.min_markers_var.get()
        if board_ids is None or len(board_ids) < needed:
            messagebox.showwarning(
                "Калибровка",
                f"Видимо {0 if board_ids is None else len(board_ids)} маркеров, "
                f"нужно минимум {needed}.",
            )
            return
        object_points, image_points = camcal.make_calibration_points(
            board_corners, board_ids, self._point_map
        )
        self.cam_object_points.append(object_points)
        self.cam_image_points.append(image_points)
        total = len(self.cam_image_points)
        self.lbl_views.configure(
            text=f"Сохранено ракурсов: {total}", text_color=C["text"]
        )
        self.set_button_state(self.btn_run_camcal, True)
        self.log(f"Добавлен ракурс {total} ({len(board_ids)} маркеров)")

    def reset_camera_views(self) -> None:
        self.cam_object_points.clear()
        self.cam_image_points.clear()
        self.lbl_views.configure(
            text="Сохранено ракурсов: 0", text_color=C["text_secondary"]
        )
        if hasattr(self, "btn_run_camcal"):
            self.set_button_state(self.btn_run_camcal, False)
        self.log("Все ракурсы калибровки камеры удалены.")

    def run_camera_calibration(self) -> None:
        views = len(self.cam_image_points)
        target = self.target_samples_var.get()
        if views < 3:
            messagebox.showwarning("Калибровка", "Нужно минимум 3 ракурса.")
            return
        if views < target and not messagebox.askyesno(
            "Калибровка", f"Ракурсов {views}, рекомендуется {target}. Продолжить?"
        ):
            return
        with self.state_lock:
            frame = self.latest_frame
        if frame is None:
            messagebox.showwarning("Калибровка", "Нет кадра для определения размера.")
            return
        image_size = (frame.shape[1], frame.shape[0])
        try:
            rms, matrix, distortion, rvecs, tvecs = cv.calibrateCamera(
                self.cam_object_points, self.cam_image_points, image_size, None, None
            )
            reprojection = camcal.mean_reprojection_error(
                self.cam_object_points, self.cam_image_points, rvecs, tvecs,
                matrix, distortion,
            )
            prefix = Path(self.cam_output_var.get())
            if prefix.suffix.lower() in (".npz", ".yaml"):
                prefix = prefix.with_suffix("")
            npz_path, yaml_path = camcal.save_calibration(
                prefix, matrix, distortion, image_size, reprojection
            )
        except Exception as exc:
            messagebox.showerror("Калибровка", f"Ошибка:\n{exc}")
            return
        self.detector.camera_matrix = matrix
        self.detector.distortion = distortion
        self.camera_calibrated = True
        self._chips_state = ()
        self.log(
            f"Калибровка камеры: RMS {rms:.4f}, репроекция {reprojection:.4f} px; "
            f"сохранено {npz_path.name} / {yaml_path.name}"
        )
        messagebox.showinfo(
            "Калибровка камеры",
            f"Готово.\nRMS OpenCV: {rms:.4f}\n"
            f"Ошибка репроекции: {reprojection:.4f} px\n\n{npz_path}",
        )
        self.reset_camera_views()

    # ------------------------------------------------------------------
    # Сервер
    # ------------------------------------------------------------------

    def start_server(self) -> bool:
        if self.server is not None:
            return True
        if self.source is None:
            messagebox.showwarning("Сервер", "Сначала подключите источник кадров.")
            return False
        if self.field is None:
            self.log("Внимание: калибровка поля не задана — сервер будет "
                     "отвечать ошибкой field-calibration-missing.", tag="warn")
        try:
            port = int(self.port_var.get())
            part_height = float(self.part_height_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Сервер", "Порт и высота детали должны быть числами.")
            return False
        try:
            server = VisionServer(
                detector=self.detector,
                field=self.field,
                frames=self.source,
                host=self.host_var.get().strip() or "0.0.0.0",
                port=port,
                part_height_mm=part_height,
                show_window=False,
                log=lambda msg: self.log(msg),
                capture_callback=self._on_server_capture,
            )
        except OSError as exc:
            messagebox.showerror("Сервер", f"Не удалось занять порт {port}:\n{exc}")
            return False

        self.server = server

        def serve() -> None:
            try:
                server.serve_forever()
            finally:
                self.events.put(("server-stopped", None))

        self.server_thread = threading.Thread(target=serve, name="vision-server", daemon=True)
        self.server_thread.start()
        host, bound_port = server.address
        self._set_server_widgets(running=True)
        self.lbl_server_state.configure(
            text=f"Слушает {host}:{bound_port} · ожидание подключения робота…",
            text_color=C["warning"],
        )
        self._chips_state = ()
        return True

    def stop_server(self, wait: bool = False) -> None:
        if self.server is None:
            return
        server, self.server = self.server, None
        server.stop()
        thread, self.server_thread = self.server_thread, None
        if wait and thread is not None:
            thread.join(timeout=3.0)
        self._set_server_widgets(running=False)
        self.lbl_server_state.configure(
            text="Сервер остановлен", text_color=C["text_secondary"]
        )
        self._chips_state = ()

    def _mark_server_stopped(self, _payload) -> None:
        if self.server is not None:
            return  # остановку инициировал сам GUI — состояние уже выставлено
        self._set_server_widgets(running=False)
        self.lbl_server_state.configure(
            text="Сервер остановлен", text_color=C["text_secondary"]
        )
        self._chips_state = ()

    def _set_server_widgets(self, running: bool) -> None:
        self.set_button_state(self.btn_server_start, not running)
        self.set_button_state(self.btn_server_stop, running)
        for entry in (self.host_entry, self.port_entry, self.height_entry):
            entry.configure(state="disabled" if running else "normal")

    def _on_server_capture(self, frame, parts, response) -> None:
        self.log("Ответ роботу:\n" + response.rstrip(), tag="xml")
        annotated = draw_parts(frame, parts)
        self.events.put(("capture-flash", (annotated, parts)))

    def manual_capture(self) -> None:
        """Тестовый захват без робота: тот же путь, что и по Trigger."""
        if self.source is None:
            messagebox.showwarning("Захват", "Сначала подключите источник.")
            return
        try:
            if self.server is not None:
                response, frame, parts = self.server.process_capture()
            else:
                frame = self.source.read()
                parts = self.detector.detect(frame)
                if self.field is None:
                    response = build_vision_xml([], error="field-calibration-missing")
                else:
                    robot_parts = parts_to_robot(
                        parts, self.field, float(self.part_height_var.get()),
                        self.detector.params.angle_symmetry_deg,
                    )
                    response = build_vision_xml(robot_parts)
        except Exception as exc:
            messagebox.showerror("Захват", f"Ошибка обработки кадра:\n{exc}")
            return
        self.log("Тестовый захват, ответ:\n" + response.rstrip(), tag="xml")
        self.events.put(("capture-flash", (draw_parts(frame, parts), parts)))
        self.server_busy_until = time.monotonic() + 2.0

    # ------------------------------------------------------------------
    # Файлы калибровок, прочее
    # ------------------------------------------------------------------

    def _load_default_calibrations(self) -> None:
        if DEFAULT_CAMERA_CALIBRATION.exists():
            matrix, distortion = load_camera_calibration(DEFAULT_CAMERA_CALIBRATION)
            self.detector.camera_matrix = matrix
            self.detector.distortion = distortion
            self.camera_calibrated = True
        if DEFAULT_FIELD_CALIBRATION.exists():
            try:
                self.field = load_field_calibration(DEFAULT_FIELD_CALIBRATION)
            except Exception:
                self.field = None

    def open_camera_calibration_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("NumPy archive", "*.npz")])
        if not path:
            return
        try:
            matrix, distortion = load_camera_calibration(Path(path))
        except Exception as exc:
            messagebox.showerror("Калибровка камеры", f"Не удалось загрузить:\n{exc}")
            return
        self.detector.camera_matrix = matrix
        self.detector.distortion = distortion
        self.camera_calibrated = True
        self._chips_state = ()
        self.log(f"Загружена калибровка камеры: {path}")

    def open_field_calibration_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("NumPy archive", "*.npz")])
        if not path:
            return
        try:
            self.field = load_field_calibration(Path(path))
        except Exception as exc:
            messagebox.showerror("Калибровка поля", f"Не удалось загрузить:\n{exc}")
            return
        self._chips_state = ()
        residual = self.field.residual_mm.max()
        self.log(f"Загружена калибровка поля: {path} (макс. ошибка {residual:.2f} мм)")

    def show_about(self) -> None:
        messagebox.showinfo(
            "О программе",
            "GUI оператора vision-ячейки KUKA + OpenCV.\n\n"
            "Просмотр и настройка детекции, калибровка камеры и поля,\n"
            "TCP-сервер KUKA.Ethernet KRL — в одном окне.",
        )

    def on_close(self) -> None:
        try:
            if self.server is not None:
                self.stop_server(wait=True)
            self.unload_source()
        finally:
            self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="GUI оператора vision-ячейки")
    parser.add_argument("--camera", default=None,
                        help="подключить источник при старте (индекс или файл)")
    args = parser.parse_args()

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    ctk.set_appearance_mode("light")
    ctk.set_widget_scaling(1.0)

    root = ctk.CTk()
    app = VisionGui(root)
    if args.camera is not None:
        app.source_var.set(args.camera)
        root.after(150, lambda: app.load_source(args.camera))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
