from __future__ import annotations

from collections import deque

from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QTimer, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QVBoxLayout, QWidget


TOAST_COLORS = {
    "info": "#27d6ff",
    "success": "#47d9a9",
    "warning": "#f2c76b",
    "error": "#ff7c8e",
}


class ToastNotification(QFrame):
    def __init__(self, parent: QWidget, title: str, message: str, tone: str = "info", duration_ms: int = 2800) -> None:
        super().__init__(parent)
        self.setObjectName("toastCard")
        self.setProperty("tone", tone)
        self.duration_ms = duration_ms
        self._closing = False
        self._position_animation: QPropertyAnimation | None = None
        self._opacity_animation: QPropertyAnimation | None = None
        self._manager = None

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowFlags(Qt.SubWindow | Qt.FramelessWindowHint)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(5, 12, 20, 190))
        self.setGraphicsEffect(shadow)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        accent = QFrame()
        accent.setObjectName("toastAccent")
        accent.setProperty("tone", tone)
        accent.setMinimumHeight(72)
        root.addWidget(accent)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(14, 12, 14, 12)
        body_layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setObjectName("toastTitle")
        title_label.setWordWrap(True)
        message_label = QLabel(message)
        message_label.setObjectName("toastMessage")
        message_label.setWordWrap(True)

        body_layout.addWidget(title_label)
        body_layout.addWidget(message_label)
        root.addWidget(body, 1)

        self.setFixedWidth(360)
        self.adjustSize()

    def bind_manager(self, manager) -> None:
        self._manager = manager

    def show_animated(self, start_pos: QPoint, end_pos: QPoint) -> None:
        self.move(start_pos)
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()

        self._position_animation = QPropertyAnimation(self, b"pos", self)
        self._position_animation.setDuration(280)
        self._position_animation.setStartValue(start_pos)
        self._position_animation.setEndValue(end_pos)
        self._position_animation.setEasingCurve(QEasingCurve.OutCubic)

        self._opacity_animation = QPropertyAnimation(self, b"windowOpacity", self)
        self._opacity_animation.setDuration(240)
        self._opacity_animation.setStartValue(0.0)
        self._opacity_animation.setEndValue(1.0)
        self._opacity_animation.setEasingCurve(QEasingCurve.OutCubic)

        self._position_animation.start()
        self._opacity_animation.start()
        QTimer.singleShot(self.duration_ms, self.close_animated)

    def close_animated(self) -> None:
        if self._closing:
            return
        self._closing = True
        start_pos = self.pos()
        end_pos = QPoint(start_pos.x() + 24, start_pos.y())

        self._position_animation = QPropertyAnimation(self, b"pos", self)
        self._position_animation.setDuration(220)
        self._position_animation.setStartValue(start_pos)
        self._position_animation.setEndValue(end_pos)
        self._position_animation.setEasingCurve(QEasingCurve.InCubic)

        self._opacity_animation = QPropertyAnimation(self, b"windowOpacity", self)
        self._opacity_animation.setDuration(200)
        self._opacity_animation.setStartValue(self.windowOpacity())
        self._opacity_animation.setEndValue(0.0)
        self._opacity_animation.setEasingCurve(QEasingCurve.InCubic)
        self._opacity_animation.finished.connect(self._finalize_close)

        self._position_animation.start()
        self._opacity_animation.start()

    def mousePressEvent(self, event) -> None:
        self.close_animated()
        super().mousePressEvent(event)

    def _finalize_close(self) -> None:
        self.hide()
        self.deleteLater()
        if self._manager is not None:
            self._manager.on_toast_closed(self)


class ToastManager:
    def __init__(self, host: QWidget) -> None:
        self.host = host
        self._toasts: list[ToastNotification] = []
        self._queue: deque[tuple[str, str, str, int]] = deque()
        self.max_visible = 4
        self.margin_top = 24
        self.margin_right = 28
        self.spacing = 12

    def show_toast(self, title: str, message: str, tone: str = "info", duration_ms: int = 2800) -> None:
        payload = (title, message, tone, duration_ms)
        if len(self._toasts) >= self.max_visible:
            self._queue.append(payload)
            return
        self._spawn(*payload)

    def _spawn(self, title: str, message: str, tone: str, duration_ms: int) -> None:
        toast = ToastNotification(self.host, title=title, message=message, tone=tone, duration_ms=duration_ms)
        toast.bind_manager(self)
        self._toasts.append(toast)
        end_pos = self._target_pos(len(self._toasts) - 1, toast.height())
        start_pos = QPoint(self.host.width() - toast.width() - self.margin_right + 24, end_pos.y())
        toast.show_animated(start_pos, end_pos)
        self._relayout(animated=True, skip=toast)

    def on_toast_closed(self, toast: ToastNotification) -> None:
        self._toasts = [item for item in self._toasts if item is not toast]
        self._relayout(animated=True)
        if self._queue and len(self._toasts) < self.max_visible:
            self._spawn(*self._queue.popleft())

    def relayout(self) -> None:
        self._relayout(animated=False)

    def _relayout(self, animated: bool, skip: ToastNotification | None = None) -> None:
        for index, toast in enumerate(self._toasts):
            if toast is skip:
                continue
            end_pos = self._target_pos(index, toast.height())
            if animated:
                animation = QPropertyAnimation(toast, b"pos", toast)
                animation.setDuration(220)
                animation.setStartValue(toast.pos())
                animation.setEndValue(end_pos)
                animation.setEasingCurve(QEasingCurve.OutCubic)
                toast._position_animation = animation
                animation.start()
            else:
                toast.move(end_pos)

    def _target_pos(self, index: int, toast_height: int) -> QPoint:
        x = self.host.width() - 360 - self.margin_right
        y = self.margin_top + index * (toast_height + self.spacing)
        return QPoint(max(0, x), y)
