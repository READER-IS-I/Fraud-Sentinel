from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QTimer, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def apply_shadow(widget: QWidget, blur: int = 30, y_offset: int = 10, alpha: int = 115) -> QGraphicsDropShadowEffect:
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, y_offset)
    shadow.setColor(QColor(4, 11, 21, alpha))
    widget.setGraphicsEffect(shadow)
    return shadow


def animate_shadow(widget: QWidget, blur: int, y_offset: int, duration: int = 180) -> None:
    shadow = getattr(widget, "_shadow", None)
    if shadow is None:
        return
    animations = getattr(widget, "_shadow_animations", {})
    blur_animation = QPropertyAnimation(shadow, b"blurRadius", widget)
    blur_animation.setDuration(duration)
    blur_animation.setStartValue(shadow.blurRadius())
    blur_animation.setEndValue(float(blur))
    blur_animation.setEasingCurve(QEasingCurve.OutCubic)

    offset_animation = QPropertyAnimation(shadow, b"offset", widget)
    offset_animation.setDuration(duration)
    offset_animation.setStartValue(shadow.offset())
    offset_animation.setEndValue(QPointF(0.0, float(y_offset)))
    offset_animation.setEasingCurve(QEasingCurve.OutCubic)

    animations["blur"] = blur_animation
    animations["offset"] = offset_animation
    widget._shadow_animations = animations
    blur_animation.start()
    offset_animation.start()


class CardFrame(QFrame):
    def __init__(self, title: str | None = None, subtitle: str | None = None, object_name: str = "card", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._shadow = apply_shadow(self, blur=30, y_offset=10, alpha=100)
        self._shadow_animations: dict[str, QPropertyAnimation] = {}

        shell_layout = QVBoxLayout(self)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        if title or subtitle:
            self.header = QFrame(self)
            self.header.setObjectName("cardHeader")
            header_layout = QVBoxLayout(self.header)
            header_layout.setContentsMargins(22, 18, 22, 14)
            header_layout.setSpacing(6)
            if title:
                title_label = QLabel(title)
                title_label.setObjectName("cardTitle")
                header_layout.addWidget(title_label)
            if subtitle:
                subtitle_label = QLabel(subtitle)
                subtitle_label.setObjectName("cardSubtitle")
                subtitle_label.setWordWrap(True)
                header_layout.addWidget(subtitle_label)
            shell_layout.addWidget(self.header)

        self.body = QFrame(self)
        self.body.setObjectName("cardBody")
        self.body.setAttribute(Qt.WA_StyledBackground, True)
        self.layout = QVBoxLayout(self.body)
        self.layout.setContentsMargins(22, 18, 22, 22)
        self.layout.setSpacing(14)
        shell_layout.addWidget(self.body)

    def enterEvent(self, event) -> None:
        animate_shadow(self, blur=42, y_offset=14, duration=180)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        animate_shadow(self, blur=30, y_offset=10, duration=180)
        super().leaveEvent(event)


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "--", detail: str = "", accent: str = "cyan", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setProperty("accent", accent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._shadow = apply_shadow(self, blur=28, y_offset=10, alpha=92)
        self._shadow_animations: dict[str, QPropertyAnimation] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        accent_bar = QFrame(self)
        accent_bar.setObjectName("metricAccentBar")
        accent_bar.setProperty("accent", accent)
        accent_bar.setFixedHeight(4)
        layout.addWidget(accent_bar)

        body = QFrame(self)
        body.setObjectName("metricBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(20, 18, 20, 18)
        body_layout.setSpacing(8)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        self.detail_label = QLabel(detail)
        self.detail_label.setObjectName("metricDetail")
        self.detail_label.setWordWrap(True)
        body_layout.addWidget(self.title_label)
        body_layout.addWidget(self.value_label)
        body_layout.addWidget(self.detail_label)
        layout.addWidget(body)

    def enterEvent(self, event) -> None:
        animate_shadow(self, blur=36, y_offset=13, duration=170)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        animate_shadow(self, blur=28, y_offset=10, duration=170)
        super().leaveEvent(event)

    def set_value(self, value: str, detail: str | None = None) -> None:
        self.value_label.setText(value)
        if detail is not None:
            self.detail_label.setText(detail)


class HighlightButton(QPushButton):
    def __init__(self, text: str, variant: str = "primary", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("actionButton")
        self.setProperty("variant", variant)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(42)
        self.setProperty("engaged", False)
        self._shadow = apply_shadow(self, blur=22, y_offset=8, alpha=66)
        self._shadow_animations: dict[str, QPropertyAnimation] = {}

    def enterEvent(self, event) -> None:
        animate_shadow(self, blur=30, y_offset=10, duration=150)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        animate_shadow(self, blur=22, y_offset=8, duration=150)
        super().leaveEvent(event)

    def _refresh_style(self) -> None:
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def mousePressEvent(self, event) -> None:
        animate_shadow(self, blur=12, y_offset=4, duration=90)
        if self.property("variant") == "primary":
            self.setProperty("engaged", True)
            self._refresh_style()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self.rect().contains(event.position().toPoint()):
            animate_shadow(self, blur=30, y_offset=10, duration=120)
        else:
            animate_shadow(self, blur=22, y_offset=8, duration=120)
        super().mouseReleaseEvent(event)
        if self.property("variant") == "primary":
            QTimer.singleShot(220, self._clear_engaged)

    def _clear_engaged(self) -> None:
        if self.property("engaged"):
            self.setProperty("engaged", False)
            self._refresh_style()


class NavigationButton(QPushButton):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("navButton")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(58)
        self.setProperty("active", False)


class LogPanel(QPlainTextEdit):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("logPanel")
        self.setReadOnly(True)
        self.setMinimumHeight(150)

    def append_line(self, text: str) -> None:
        self.appendPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class SectionLabel(QLabel):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("sectionLabel")


class InfoBadge(QLabel):
    def __init__(self, text: str, tone: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("infoBadge")
        self.setProperty("tone", tone)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(30)


def hero_stat(title: str, value: str, detail: str) -> QWidget:
    wrapper = QFrame()
    wrapper.setObjectName("heroStat")
    wrapper.setAttribute(Qt.WA_StyledBackground, True)
    wrapper._shadow = apply_shadow(wrapper, blur=22, y_offset=8, alpha=70)
    wrapper._shadow_animations = {}
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(6)
    label_title = QLabel(title)
    label_title.setObjectName("heroStatTitle")
    label_value = QLabel(value)
    label_value.setObjectName("heroStatValue")
    label_detail = QLabel(detail)
    label_detail.setObjectName("heroStatDetail")
    label_detail.setWordWrap(True)
    layout.addWidget(label_title)
    layout.addWidget(label_value)
    layout.addWidget(label_detail)
    return wrapper


def labeled_value(title: str, value: str = "--") -> QWidget:
    wrapper = QFrame()
    wrapper.setObjectName("valueTile")
    wrapper.setAttribute(Qt.WA_StyledBackground, True)
    wrapper._shadow = apply_shadow(wrapper, blur=18, y_offset=7, alpha=62)
    wrapper._shadow_animations = {}
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(6)
    title_label = QLabel(title)
    title_label.setObjectName("valueTileTitle")
    value_label = QLabel(value)
    value_label.setObjectName("valueTileValue")
    layout.addWidget(title_label)
    layout.addWidget(value_label)
    wrapper.value_label = value_label
    return wrapper


def create_path_row(label_text: str, editor, button) -> QWidget:
    wrapper = QWidget()
    wrapper.setObjectName("pathRow")
    layout = QHBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)
    label = QLabel(label_text)
    label.setObjectName("fieldLabel")
    label.setMinimumWidth(110)
    layout.addWidget(label)
    layout.addWidget(editor, 1)
    layout.addWidget(button)
    return wrapper


def _resolve_toast_host(parent: QWidget | None) -> QWidget | None:
    widget = parent
    while widget is not None:
        if hasattr(widget, "show_toast"):
            return widget
        widget = widget.parentWidget()
    return None


def show_error(parent: QWidget, title: str, message: str) -> None:
    host = _resolve_toast_host(parent)
    if host is not None:
        host.show_toast(title, message, tone="error", duration_ms=4200)
        return
    QMessageBox.critical(parent, title, message)


def show_info(parent: QWidget, title: str, message: str) -> None:
    host = _resolve_toast_host(parent)
    if host is not None:
        host.show_toast(title, message, tone="info", duration_ms=2800)
        return
    QMessageBox.information(parent, title, message)


def show_success(parent: QWidget, title: str, message: str) -> None:
    host = _resolve_toast_host(parent)
    if host is not None:
        host.show_toast(title, message, tone="success", duration_ms=2800)
        return
    QMessageBox.information(parent, title, message)


def show_warning(parent: QWidget, title: str, message: str) -> None:
    host = _resolve_toast_host(parent)
    if host is not None:
        host.show_toast(title, message, tone="warning", duration_ms=3200)
        return
    QMessageBox.warning(parent, title, message)
