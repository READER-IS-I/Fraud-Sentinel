from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.ui.widgets import CardFrame, HighlightButton, InfoBadge, hero_stat


class HomePage(QWidget):
    navigate_requested = Signal(str)
    page_title = "\u7cfb\u7edf\u603b\u89c8"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(18)

        hero = CardFrame(object_name="heroCard")
        title = QLabel("\u91d1\u9274\u667a\u9632 FraudShield")
        title.setObjectName("heroTitle")
        subtitle = QLabel("\u91d1\u878d\u6b3a\u8bc8\u98ce\u9669\u8bc6\u522b\u5e73\u53f0")
        subtitle.setObjectName("heroSubtitle")
        description = QLabel(
            "\u9762\u5411 PaySim \u5927\u89c4\u6a21\u4ea4\u6613\u6570\u636e\u7684\u684c\u9762\u7aef\u98ce\u63a7\u5e73\u53f0\uff0c"
            "\u805a\u7126\u4e8e SMOTE \u7c7b\u522b\u5e73\u8861\u3001MLP \u8bad\u7ec3\u3001\u6a21\u578b\u8bc4\u4f30\u3001\u5b89\u5168\u63a8\u7406\u3001\u6570\u636e\u5206\u6790\u4e0e\u6a21\u578b\u5bf9\u6bd4\u5c55\u793a\u3002"
        )
        description.setObjectName("heroDescription")
        description.setWordWrap(True)

        badge_row = QHBoxLayout()
        for text, tone in [("PaySim \u5c31\u7eea", "success"), ("SMOTE + MLP", "accent"), ("\u5206\u6790\u4e0e\u5bf9\u6bd4", "warning")]:
            badge_row.addWidget(InfoBadge(text, tone=tone))
        badge_row.addStretch(1)

        button_row = QHBoxLayout()
        for text, page, variant in [
            ("\u5feb\u901f\u5f00\u59cb", "data", "primary"),
            ("\u6570\u636e\u5206\u6790", "analysis", "secondary"),
            ("\u6a21\u578b\u5bf9\u6bd4", "compare", "ghost"),
        ]:
            button = HighlightButton(text, variant=variant)
            button.clicked.connect(lambda _checked=False, target=page: self.navigate_requested.emit(target))
            button_row.addWidget(button)
        button_row.addStretch(1)

        stats = QHBoxLayout()
        stats.addWidget(hero_stat("\u6838\u5fc3\u4e3b\u7ebf", "PaySim -> SMOTE -> MLP", "\u4fdd\u7559\u6700\u7ec8\u7b54\u8fa9\u4e3b\u7ebf\uff0c\u907f\u514d\u6570\u636e\u6cc4\u6f0f"))
        stats.addWidget(hero_stat("\u63a2\u7d22\u5206\u6790", "PCA / t-SNE / \u57fa\u7ebf\u6a21\u578b", "\u8865\u8db3\u8bf4\u660e\u4e66\u4e2d\u7684\u964d\u7ef4\u5bf9\u6bd4\u5206\u6790\u4e0e\u673a\u5668\u5b66\u4e60\u5bf9\u6bd4"))
        stats.addWidget(hero_stat("\u5c55\u793a\u5f62\u6001", "Dashboard UI", "\u9762\u5411\u7b54\u8fa9\u4e0e\u8bc4\u59d4\u73b0\u573a\u6d4b\u8bd5\u7684\u9ad8\u89c2\u611f\u5c55\u793a\u754c\u9762"))

        hero.layout.addWidget(title)
        hero.layout.addWidget(subtitle)
        hero.layout.addWidget(description)
        hero.layout.addLayout(badge_row)
        hero.layout.addLayout(button_row)
        hero.layout.addLayout(stats)
        root.addWidget(hero)

        capability_card = CardFrame("\u6838\u5fc3\u80fd\u529b", "\u4ece\u539f\u59cb\u4ea4\u6613\u6570\u636e\u5230\u5b89\u5168\u63a8\u7406\u7684\u5b8c\u6574\u4e1a\u52a1\u95ed\u73af\u3002")
        capability_grid = QGridLayout()
        capability_grid.setHorizontalSpacing(12)
        capability_grid.setVerticalSpacing(12)
        capabilities = [
            ("01", "\u6570\u636e\u5bfc\u5165", "\u81ea\u52a8\u8bc6\u522b PaySim \u5b57\u6bb5\u7ed3\u6784\uff0c\u652f\u6301\u5927\u6587\u4ef6\u7edf\u8ba1\u4e0e\u9884\u89c8"),
            ("02", "\u6570\u636e\u5206\u6790", "\u63d0\u4f9b\u7c7b\u522b\u5206\u5e03\u3001\u4ea4\u6613\u7c7b\u578b\u5206\u6790\u3001PCA\u3001t-SNE \u4e0e\u76f8\u5173\u6027\u70ed\u529b\u56fe"),
            ("03", "\u6a21\u578b\u5bf9\u6bd4", "\u903b\u8f91\u56de\u5f52\u3001KNN\u3001SVM\u3001\u51b3\u7b56\u6811\u4e0e MLP \u6307\u6807\u5bf9\u7167"),
            ("04", "SMOTE \u5e73\u8861", "\u4ec5\u5728 train split \u6267\u884c\u8fc7\u91c7\u6837\uff0c\u907f\u514d\u4f30\u8ba1\u6307\u6807\u865a\u9ad8"),
            ("05", "MLP \u8bad\u7ec3", "PyTorch \u5f02\u6b65\u8bad\u7ec3\uff0c\u5b9e\u65f6\u65e5\u5fd7\u4e0e loss \u66f2\u7ebf"),
            ("06", "\u6a21\u578b\u8bc4\u4f30", "\u63d0\u4f9b ROC / PR / \u6df7\u6dc6\u77e9\u9635 / \u5206\u5e03\u5bf9\u6bd4\u770b\u677f"),
            ("07", "\u52a0\u5bc6\u89e3\u5bc6", "\u6743\u91cd\u6587\u4ef6\u52a0\u5bc6\u5c01\u88c5\u3001\u54c8\u5e0c\u6821\u9a8c\u548c\u89e3\u5bc6\u6062\u590d"),
            ("08", "\u98ce\u9669\u63a8\u7406", "\u652f\u6301\u5355\u6761\u4ea4\u6613\u5f55\u5165\u4e0e\u6279\u91cf CSV \u98ce\u9669\u8bc6\u522b"),
        ]
        for index, (num, name, desc) in enumerate(capabilities):
            card = CardFrame(object_name="capabilityCard")
            num_label = QLabel(num)
            num_label.setObjectName("capabilityIndex")
            name_label = QLabel(name)
            name_label.setObjectName("capabilityTitle")
            desc_label = QLabel(desc)
            desc_label.setObjectName("capabilityDesc")
            desc_label.setWordWrap(True)
            card.layout.addWidget(num_label)
            card.layout.addWidget(name_label)
            card.layout.addWidget(desc_label)
            capability_grid.addWidget(card, index // 4, index % 4)
        capability_card.layout.addLayout(capability_grid)
        root.addWidget(capability_card)

        guide_card = CardFrame("\u6f14\u793a\u6d41\u7a0b", "\u63a8\u8350\u73b0\u573a\u6f14\u793a\u8def\u5f84\u3002")
        guide = QLabel(
            "1. \u8fdb\u5165\u793a\u4f8b\u6f14\u793a\uff0c\u4e00\u952e\u8bad\u7ec3\u5185\u7f6e PaySim \u6837\u672c\u6a21\u578b\u3002\n"
            "2. \u5728\u6570\u636e\u5206\u6790\u9875\u5c55\u793a\u4ea4\u6613\u5206\u5e03\u3001PCA\u3001t-SNE \u4e0e\u70ed\u529b\u56fe\u3002\n"
            "3. \u5728\u6a21\u578b\u5bf9\u6bd4\u9875\u5c55\u793a\u4f20\u7edf\u6a21\u578b\u4e0e MLP \u5bf9\u6bd4\u3002\n"
            "4. \u5728\u8bc4\u4f30\u9875\u5c55\u793a ROC\u3001PR \u4e0e\u6df7\u6dc6\u77e9\u9635\u3002\n"
            "5. \u5728\u63a8\u7406\u9875\u4e0e\u52a0\u5bc6\u9875\u5b8c\u6210\u6700\u7ec8\u4e1a\u52a1\u6f14\u793a\u3002"
        )
        guide.setObjectName("bodyText")
        guide.setWordWrap(True)
        guide_card.layout.addWidget(guide)
        root.addWidget(guide_card)
        root.addStretch(1)
