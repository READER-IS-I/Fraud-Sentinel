from __future__ import annotations

from importlib import import_module

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget

from src.ui.toast import ToastManager
from src.ui.widgets import CardFrame, HighlightButton, InfoBadge, NavigationButton
from src.utils.logger import get_logger
from src.utils.runtime_env import EnvironmentCheckReport


PAGE_SPECS = {
    "home": ("src.ui.pages.home_page", "HomePage", "\u7cfb\u7edf\u603b\u89c8"),
    "demo": ("src.ui.pages.demo_page", "DemoPage", "\u793a\u4f8b\u6f14\u793a"),
    "data": ("src.ui.pages.data_page", "DataPage", "\u6570\u636e\u5bfc\u5165"),
    "analysis": ("src.ui.pages.analysis_page", "AnalysisPage", "\u6570\u636e\u5206\u6790"),
    "compare": ("src.ui.pages.compare_page", "ComparePage", "\u6a21\u578b\u5bf9\u6bd4"),
    "train": ("src.ui.pages.train_page", "TrainPage", "\u6a21\u578b\u8bad\u7ec3"),
    "eval": ("src.ui.pages.eval_page", "EvalPage", "\u6a21\u578b\u8bc4\u4f30"),
    "predict": ("src.ui.pages.predict_page", "PredictPage", "\u98ce\u9669\u63a8\u7406"),
    "crypto": ("src.ui.pages.crypto_page", "CryptoPage", "\u6a21\u578b\u52a0\u5bc6"),
}

PAGE_ORDER = [
    ("home", "\u7cfb\u7edf\u603b\u89c8"),
    ("demo", "\u793a\u4f8b\u6f14\u793a"),
    ("data", "\u6570\u636e\u5bfc\u5165"),
    ("analysis", "\u6570\u636e\u5206\u6790"),
    ("compare", "\u6a21\u578b\u5bf9\u6bd4"),
    ("train", "\u6a21\u578b\u8bad\u7ec3"),
    ("eval", "\u6a21\u578b\u8bc4\u4f30"),
    ("predict", "\u98ce\u9669\u63a8\u7406"),
    ("crypto", "\u6a21\u578b\u52a0\u5bc6"),
]


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("windowRoot")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowTitle("\u91d1\u9274\u667a\u9632 FraudShield")
        self.resize(1520, 920)
        self.nav_buttons: dict[str, NavigationButton] = {}
        self.toast_manager = ToastManager(self)
        self.pages: dict[str, QWidget] = {}
        self.page_views: dict[str, QScrollArea] = {}
        self.page_errors: dict[str, str] = {}
        self.latest_artifacts: dict | None = None
        self.latest_decrypted_model: str | None = None
        self.logger = get_logger()

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(336)
        sidebar.setAttribute(Qt.WA_StyledBackground, True)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(22, 22, 22, 22)
        sidebar_layout.setSpacing(14)
        sidebar_layout.setAlignment(Qt.AlignTop)

        brand_panel = QFrame()
        brand_panel.setObjectName("brandPanel")
        brand_panel.setAttribute(Qt.WA_StyledBackground, True)
        brand_panel.setFixedHeight(342)
        brand_layout = QVBoxLayout(brand_panel)
        brand_layout.setContentsMargins(24, 24, 24, 24)
        brand_layout.setSpacing(8)

        brand_badge = QLabel("FS")
        brand_badge.setObjectName("brandBadge")

        title_cn = QLabel("\u91d1\u9274\u667a\u9632")
        title_cn.setObjectName("sidebarTitleCn")
        title_cn.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_cn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        title_en = QLabel("FraudShield")
        title_en.setObjectName("sidebarTitleEn")
        title_en.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_en.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        app_subtitle = QLabel("\u667a\u80fd\u76d1\u6d4b / \u5b89\u5168\u63a8\u7406 / \u6a21\u578b\u4fdd\u62a4")
        app_subtitle.setObjectName("sidebarSubtitle")
        app_subtitle.setWordWrap(True)
        app_subtitle.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        app_subtitle.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge_row.setSpacing(8)
        badge_row.addWidget(InfoBadge("SMOTE + MLP", tone="warning"))
        badge_row.addWidget(InfoBadge("PaySim", tone="accent"))
        badge_row.addStretch(1)

        brand_layout.addWidget(brand_badge, 0, Qt.AlignLeft)
        brand_layout.addSpacing(6)
        brand_layout.addWidget(title_cn)
        brand_layout.addWidget(title_en)
        brand_layout.addSpacing(2)
        brand_layout.addWidget(app_subtitle)
        brand_layout.addLayout(badge_row)
        sidebar_layout.addWidget(brand_panel)

        nav_caption = QLabel("\u529f\u80fd\u5bfc\u822a")
        nav_caption.setObjectName("navSectionLabel")
        sidebar_layout.addSpacing(6)
        sidebar_layout.addWidget(nav_caption)

        nav_scroll = QScrollArea()
        nav_scroll.setObjectName("sidebarNavScroll")
        nav_scroll.setWidgetResizable(True)
        nav_scroll.setFrameShape(QFrame.NoFrame)
        nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        nav_container = QWidget()
        nav_container.setObjectName("sidebarNavContainer")
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(10)
        nav_layout.setAlignment(Qt.AlignTop)
        nav_scroll.setWidget(nav_container)
        sidebar_layout.addWidget(nav_scroll, 1)

        content_wrapper = QFrame()
        content_wrapper.setObjectName("contentWrapper")
        content_layout = QVBoxLayout(content_wrapper)
        content_layout.setContentsMargins(22, 22, 22, 22)
        content_layout.setSpacing(16)

        topbar = QFrame()
        topbar.setObjectName("topBar")
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(22, 18, 22, 18)
        topbar_layout.setSpacing(12)

        title_group = QVBoxLayout()
        title_group.setContentsMargins(0, 0, 0, 0)
        title_group.setSpacing(4)
        self.page_title = QLabel("\u7cfb\u7edf\u603b\u89c8")
        self.page_title.setObjectName("pageTitle")
        self.page_subtitle = QLabel("\u91d1\u878d\u6b3a\u8bc8\u8bc6\u522b\u4e0e\u98ce\u63a7\u6f14\u793a\u5e73\u53f0")
        self.page_subtitle.setObjectName("pageSubtitle")
        title_group.addWidget(self.page_title)
        title_group.addWidget(self.page_subtitle)

        topbar_layout.addLayout(title_group)
        topbar_layout.addStretch(1)
        topbar_layout.addWidget(InfoBadge("SMOTE + MLP", tone="accent"))
        topbar_layout.addWidget(InfoBadge("\u672c\u5730\u5355\u673a\u6f14\u793a", tone="neutral"))
        content_layout.addWidget(topbar)

        self.stack = QStackedWidget()
        self.stack.setObjectName("pageStack")

        for key, label in PAGE_ORDER:
            button = NavigationButton(label)
            button.clicked.connect(lambda _checked=False, target=key: self.switch_page(target))
            nav_layout.addWidget(button)
            self.nav_buttons[key] = button

            scroll = QScrollArea()
            scroll.setObjectName("pageScroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(self._build_loading_page(label))
            self.page_views[key] = scroll
            self.stack.addWidget(scroll)

        footer = QLabel("Windows \u684c\u9762\u7aef / \u91d1\u878d\u98ce\u63a7 UI / \u7b54\u8fa9\u6f14\u793a\u7248")
        footer.setObjectName("sidebarFooter")
        footer.setWordWrap(True)
        sidebar_layout.addWidget(footer)

        page_shell = QFrame()
        page_shell.setObjectName("pageViewport")
        page_shell.setAttribute(Qt.WA_StyledBackground, True)
        shell_layout = QVBoxLayout(page_shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.addWidget(self.stack)
        content_layout.addWidget(page_shell, 1)

        root.addWidget(sidebar)
        root.addWidget(content_wrapper, 1)

        self.switch_page("home")

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        background = QLinearGradient(0, 0, 0, self.height())
        background.setColorAt(0.0, QColor("#07111E"))
        background.setColorAt(0.45, QColor("#081420"))
        background.setColorAt(1.0, QColor("#050B13"))
        painter.fillRect(self.rect(), background)

        cyan_glow = QRadialGradient(self.width() * 0.76, self.height() * 0.14, self.width() * 0.34)
        cyan_glow.setColorAt(0.0, QColor(53, 208, 255, 52))
        cyan_glow.setColorAt(0.42, QColor(53, 208, 255, 16))
        cyan_glow.setColorAt(1.0, QColor(53, 208, 255, 0))
        painter.setBrush(cyan_glow)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(self.width() * 0.48), int(-self.height() * 0.12), int(self.width() * 0.56), int(self.height() * 0.54))

        violet_glow = QRadialGradient(self.width() * 0.24, self.height() * 0.82, self.width() * 0.28)
        violet_glow.setColorAt(0.0, QColor(124, 108, 255, 34))
        violet_glow.setColorAt(0.5, QColor(124, 108, 255, 10))
        violet_glow.setColorAt(1.0, QColor(124, 108, 255, 0))
        painter.setBrush(violet_glow)
        painter.drawEllipse(int(self.width() * 0.04), int(self.height() * 0.58), int(self.width() * 0.42), int(self.height() * 0.38))

        painter.setBrush(Qt.NoBrush)
        grid_pen = QPen(QColor(100, 156, 220, 12))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        sidebar_width = 286
        for x in range(sidebar_width + 48, self.width(), 72):
            painter.drawLine(x, 0, x, self.height())
        for y in range(24, self.height(), 72):
            painter.drawLine(sidebar_width, y, self.width(), y)

        super().paintEvent(event)

    def show_toast(self, title: str, message: str, tone: str = "info", duration_ms: int = 2800) -> None:
        self.toast_manager.show_toast(title, message, tone=tone, duration_ms=duration_ms)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.toast_manager.relayout()

    def switch_page(self, key: str) -> None:
        index_map = {name: index for index, (name, _label) in enumerate(PAGE_ORDER)}
        if key not in index_map:
            return
        self._ensure_page_loaded(key)
        self.stack.setCurrentIndex(index_map[key])
        page = self.pages.get(key)
        default_title = PAGE_SPECS[key][2]
        self.page_title.setText(getattr(page, "page_title", default_title))
        self.page_subtitle.setText("\u91d1\u878d\u6b3a\u8bc8\u8bc6\u522b / SMOTE / MLP / \u4f01\u4e1a\u7ea7\u98ce\u63a7\u6f14\u793a")
        for name, button in self.nav_buttons.items():
            button.setProperty("active", name == key)
            button.style().unpolish(button)
            button.style().polish(button)
        self._animate_current_page()

    def report_startup_self_check(self, report: EnvironmentCheckReport) -> None:
        def _notify() -> None:
            if report.errors:
                self.show_toast("\u542f\u52a8\u81ea\u68c0\u5931\u8d25", report.errors[0], tone="error", duration_ms=5500)
            elif report.warnings:
                self.show_toast("\u542f\u52a8\u81ea\u68c0\u63d0\u793a", report.warnings[0], tone="warning", duration_ms=4200)

        QTimer.singleShot(300, _notify)

    def _animate_current_page(self) -> None:
        widget = self.stack.currentWidget()
        if widget is None:
            return
        if widget.graphicsEffect() is not None:
            widget.setGraphicsEffect(None)
        widget.update()

    def _ensure_page_loaded(self, key: str) -> QWidget:
        if key in self.pages:
            return self.pages[key]

        module_name, class_name, title = PAGE_SPECS[key]
        try:
            page_cls = getattr(import_module(module_name), class_name)
            page = page_cls()
            self.pages[key] = page
            self.page_views[key].setWidget(page)
            self._connect_page_signals(key, page)
            self._apply_cached_state_to_page(key, page)
            self.logger.info("Page loaded: %s", key)
            return page
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.page_errors[key] = message
            self.logger.exception("Failed to load page %s", key)
            error_page = self._build_error_page(key, title, message)
            self.pages[key] = error_page
            self.page_views[key].setWidget(error_page)
            return error_page

    def _connect_page_signals(self, key: str, page: QWidget) -> None:
        if key == "home" and hasattr(page, "navigate_requested"):
            page.navigate_requested.connect(self.switch_page)
        if key in {"train", "demo"} and hasattr(page, "artifacts_ready"):
            page.artifacts_ready.connect(self._propagate_artifacts)
        if key == "crypto" and hasattr(page, "decrypted_ready"):
            page.decrypted_ready.connect(self._on_decrypted_model)

    def _apply_cached_state_to_page(self, key: str, page: QWidget) -> None:
        if self.latest_artifacts and key in {"eval", "predict", "crypto"}:
            model_path = self.latest_artifacts.get("model_path", "")
            preprocessor_path = self.latest_artifacts.get("preprocessor_path", "")
            if key in {"eval", "predict"} and hasattr(page, "set_artifact_paths"):
                page.set_artifact_paths(model_path, preprocessor_path)
            if key == "crypto" and hasattr(page, "set_model_path"):
                page.set_model_path(model_path)
        if self.latest_decrypted_model and key in {"eval", "predict"}:
            if key == "predict" and hasattr(page, "model_edit"):
                page.model_edit.setText(self.latest_decrypted_model)
            if key == "eval" and hasattr(page, "model_edit"):
                page.model_edit.setText(self.latest_decrypted_model)

    def _build_loading_page(self, title: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(CardFrame(title, "\u9875\u9762\u5c06\u5728\u9996\u6b21\u8fdb\u5165\u65f6\u6309\u9700\u52a0\u8f7d\u3002"))
        layout.addStretch(1)
        return page

    def _build_error_page(self, key: str, title: str, message: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        card = CardFrame(f"{title} \u52a0\u8f7d\u5931\u8d25", "\u4e3b\u7a0b\u5e8f\u5df2\u7ee7\u7eed\u8fd0\u884c\uff0c\u4f46\u5f53\u524d\u529f\u80fd\u6a21\u5757\u672a\u80fd\u52a0\u8f7d\u3002")
        detail = QLabel(message)
        detail.setWordWrap(True)
        detail.setObjectName("bodyText")
        retry = HighlightButton("\u91cd\u8bd5\u52a0\u8f7d", variant="secondary")
        retry.clicked.connect(lambda: self._retry_page_load(key))
        card.layout.addWidget(detail)
        card.layout.addWidget(retry)
        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _retry_page_load(self, key: str) -> None:
        self.pages.pop(key, None)
        self.page_errors.pop(key, None)
        self.page_views[key].setWidget(self._build_loading_page(PAGE_SPECS[key][2]))
        self.switch_page(key)

    def _propagate_artifacts(self, result: dict) -> None:
        self.latest_artifacts = result
        model_path = result.get("model_path", "")
        preprocessor_path = result.get("preprocessor_path", "")
        for key in ("eval", "predict"):
            page = self.pages.get(key)
            if page is not None and hasattr(page, "set_artifact_paths"):
                page.set_artifact_paths(model_path, preprocessor_path)
        crypto_page = self.pages.get("crypto")
        if crypto_page is not None and hasattr(crypto_page, "set_model_path"):
            crypto_page.set_model_path(model_path)

    def _on_decrypted_model(self, model_path: str) -> None:
        self.latest_decrypted_model = model_path
        predict_page = self.pages.get("predict")
        eval_page = self.pages.get("eval")
        if predict_page is not None and hasattr(predict_page, "model_edit"):
            predict_page.model_edit.setText(model_path)
        if eval_page is not None and hasattr(eval_page, "model_edit"):
            eval_page.model_edit.setText(model_path)
