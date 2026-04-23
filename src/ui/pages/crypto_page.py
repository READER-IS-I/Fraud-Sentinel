from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QLabel, QLineEdit, QVBoxLayout, QWidget

from src.core.crypto_utils import decrypt_file, encrypt_file, ensure_rsa_keypair
from src.ui.widgets import CardFrame, HighlightButton, LogPanel, create_path_row, show_error, show_info
from src.utils.file_utils import MODELS_DIR, OUTPUTS_DIR, get_dialog_start_dir


class CryptoPage(QWidget):
    decrypted_ready = Signal(str)
    page_title = "\u6a21\u578b\u52a0\u5bc6 / \u89e3\u5bc6"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.default_private = MODELS_DIR / "keys" / "fraudshield_private.pem"
        self.default_public = MODELS_DIR / "keys" / "fraudshield_public.pem"
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(18)

        key_card = CardFrame("\u5bc6\u94a5\u7ba1\u7406", "\u751f\u6210\u6216\u590d\u7528 RSA \u5bc6\u94a5\uff0c\u7528\u4e8e\u6a21\u578b\u6587\u4ef6\u4fdd\u62a4\u3002")
        self.private_edit = QLineEdit(str(self.default_private))
        self.public_edit = QLineEdit(str(self.default_public))
        key_card.layout.addWidget(create_path_row("\u79c1\u94a5", self.private_edit, self._browse_button(self.private_edit, "\u4fdd\u5b58\u79c1\u94a5", True, "PEM Files (*.pem)")))
        key_card.layout.addWidget(create_path_row("\u516c\u94a5", self.public_edit, self._browse_button(self.public_edit, "\u4fdd\u5b58\u516c\u94a5", True, "PEM Files (*.pem)")))
        generate_button = HighlightButton("\u751f\u6210\u5bc6\u94a5", variant="primary")
        generate_button.clicked.connect(self.generate_keys)
        key_card.layout.addWidget(generate_button)
        root.addWidget(key_card)

        encrypt_card = CardFrame("\u6a21\u578b\u52a0\u5bc6", "\u5bf9 model.pt \u6267\u884c\u52a0\u5bc6\uff0c\u5e76\u751f\u6210\u542b SHA-256 \u6458\u8981\u7684\u5143\u6570\u636e\u6587\u4ef6\u3002")
        self.model_edit = QLineEdit()
        self.encrypt_out_edit = QLineEdit(str(OUTPUTS_DIR / "encrypted"))
        encrypt_card.layout.addWidget(create_path_row("\u6a21\u578b\u6587\u4ef6", self.model_edit, self._browse_button(self.model_edit, "\u9009\u62e9\u6a21\u578b\u6587\u4ef6", False, "PyTorch Files (*.pt)")))
        encrypt_card.layout.addWidget(create_path_row("\u8f93\u51fa\u76ee\u5f55", self.encrypt_out_edit, self._dir_button(self.encrypt_out_edit, "\u9009\u62e9\u8f93\u51fa\u76ee\u5f55")))
        encrypt_button = HighlightButton("\u6267\u884c\u52a0\u5bc6", variant="secondary")
        encrypt_button.clicked.connect(self.run_encrypt)
        encrypt_card.layout.addWidget(encrypt_button)
        root.addWidget(encrypt_card)

        decrypt_card = CardFrame("\u6a21\u578b\u89e3\u5bc6", "\u6062\u590d\u5df2\u52a0\u5bc6\u7684\u6a21\u578b\u6587\u4ef6\uff0c\u7528\u4e8e\u5b89\u5168\u63a8\u7406\u3002")
        self.encrypted_edit = QLineEdit()
        self.meta_edit = QLineEdit()
        self.decrypt_out_edit = QLineEdit(str(OUTPUTS_DIR / "decrypted" / "model_restored.pt"))
        decrypt_card.layout.addWidget(create_path_row("\u52a0\u5bc6\u6587\u4ef6", self.encrypted_edit, self._browse_button(self.encrypted_edit, "\u9009\u62e9\u52a0\u5bc6\u6587\u4ef6", False, "Encrypted Files (*.fsenc)")))
        decrypt_card.layout.addWidget(create_path_row("Meta JSON", self.meta_edit, self._browse_button(self.meta_edit, "\u9009\u62e9\u5143\u6570\u636e\u6587\u4ef6", False, "JSON Files (*.json)")))
        decrypt_card.layout.addWidget(create_path_row("\u8f93\u51fa\u6587\u4ef6", self.decrypt_out_edit, self._browse_button(self.decrypt_out_edit, "\u4fdd\u5b58\u89e3\u5bc6\u540e\u6a21\u578b", True, "PyTorch Files (*.pt)")))
        decrypt_button = HighlightButton("\u6267\u884c\u89e3\u5bc6", variant="primary")
        decrypt_button.clicked.connect(self.run_decrypt)
        root.addWidget(decrypt_card)
        decrypt_card.layout.addWidget(decrypt_button)

        status_card = CardFrame("\u72b6\u6001\u770b\u677f")
        self.status_label = QLabel("\u72b6\u6001\uff1a\u5f85\u5904\u7406")
        self.hash_label = QLabel("SHA-256\uff1a--")
        self.path_label = QLabel("\u8def\u5f84\uff1a--")
        self.status_label.setObjectName("bodyText")
        self.hash_label.setObjectName("bodyText")
        self.path_label.setObjectName("bodyText")
        for widget in [self.status_label, self.hash_label, self.path_label]:
            widget.setWordWrap(True)
            status_card.layout.addWidget(widget)
        root.addWidget(status_card)

        log_card = CardFrame("\u52a0\u5bc6\u65e5\u5fd7")
        self.log_panel = LogPanel()
        log_card.layout.addWidget(self.log_panel)
        root.addWidget(log_card)
        root.addStretch(1)

    def _browse_button(self, target: QLineEdit, title: str, save_mode: bool, pattern: str) -> HighlightButton:
        button = HighlightButton("\u6d4f\u89c8", variant="ghost")
        button.clicked.connect(lambda: self._browse_file(target, title, save_mode, pattern))
        return button

    def _dir_button(self, target: QLineEdit, title: str) -> HighlightButton:
        button = HighlightButton("\u9009\u62e9", variant="ghost")
        button.clicked.connect(lambda: self._browse_directory(target, title))
        return button

    def _browse_file(self, target: QLineEdit, title: str, save_mode: bool, pattern: str) -> None:
        if save_mode:
            path, _ = QFileDialog.getSaveFileName(self, title, target.text(), pattern)
        else:
            path, _ = QFileDialog.getOpenFileName(self, title, get_dialog_start_dir(MODELS_DIR), pattern)
        if path:
            target.setText(path)

    def _browse_directory(self, target: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title, target.text() or str(OUTPUTS_DIR))
        if path:
            target.setText(path)

    def set_model_path(self, model_path: str) -> None:
        self.model_edit.setText(model_path)

    def generate_keys(self) -> None:
        try:
            private_path, public_path = ensure_rsa_keypair(self.private_edit.text().strip(), self.public_edit.text().strip())
            self.private_edit.setText(str(private_path))
            self.public_edit.setText(str(public_path))
            self.log_panel.append_line("RSA \u5bc6\u94a5\u5bf9\u5df2\u5c31\u7eea\u3002")
            show_info(self, "\u5bc6\u94a5\u5c31\u7eea", "RSA \u5bc6\u94a5\u5df2\u751f\u6210\u3002")
        except Exception as exc:
            show_error(self, "\u5bc6\u94a5\u751f\u6210\u5931\u8d25", str(exc))

    def run_encrypt(self) -> None:
        try:
            ensure_rsa_keypair(self.private_edit.text().strip(), self.public_edit.text().strip())
            result = encrypt_file(self.model_edit.text().strip(), self.encrypt_out_edit.text().strip(), self.public_edit.text().strip())
            self.encrypted_edit.setText(result["encrypted_path"])
            self.meta_edit.setText(result["meta_path"])
            self.status_label.setText("\u72b6\u6001\uff1a\u5df2\u52a0\u5bc6")
            self.hash_label.setText(f"SHA-256\uff1a{result['sha256'][:20]}...")
            self.path_label.setText(f"\u8def\u5f84\uff1a{result['encrypted_path']}")
            self.log_panel.append_line(f"\u5df2\u52a0\u5bc6\uff1a{result['encrypted_path']}")
            show_info(self, "\u52a0\u5bc6\u6210\u529f", "\u6a21\u578b\u52a0\u5bc6\u5df2\u5b8c\u6210\u3002")
        except Exception as exc:
            show_error(self, "\u52a0\u5bc6\u5931\u8d25", str(exc))
            self.log_panel.append_line(f"\u52a0\u5bc6\u5931\u8d25: {exc}")

    def run_decrypt(self) -> None:
        try:
            result = decrypt_file(self.encrypted_edit.text().strip(), self.meta_edit.text().strip(), self.private_edit.text().strip(), self.decrypt_out_edit.text().strip())
            self.status_label.setText("\u72b6\u6001\uff1a\u5df2\u89e3\u5bc6")
            self.hash_label.setText(f"SHA-256\uff1a{result['sha256'][:20]}...")
            self.path_label.setText(f"\u8def\u5f84\uff1a{result['decrypted_path']}")
            self.log_panel.append_line(f"\u5df2\u89e3\u5bc6\uff1a{result['decrypted_path']}")
            self.decrypted_ready.emit(result["decrypted_path"])
            show_info(self, "\u89e3\u5bc6\u6210\u529f", "\u6a21\u578b\u5df2\u6062\u590d\uff0c\u53ef\u7528\u4e8e\u63a8\u7406\u3002")
        except Exception as exc:
            show_error(self, "\u89e3\u5bc6\u5931\u8d25", str(exc))
            self.log_panel.append_line(f"\u89e3\u5bc6\u5931\u8d25: {exc}")
