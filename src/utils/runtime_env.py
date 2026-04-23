from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

from .file_utils import (
    APP_NAME,
    ASSETS_DIR,
    DATA_DIR,
    DEMO_DIR,
    EXAMPLES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    RESOURCE_ROOT,
    USER_DATA_DIR,
    ensure_dir,
    is_frozen_app,
)
from .logger import get_logger


@dataclass(slots=True)
class EnvironmentCheckReport:
    summary: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _check_file(report: EnvironmentCheckReport, path: Path, label: str, required: bool = True) -> None:
    report.details[label] = str(path)
    if path.exists():
        return
    message = f"{label} 不存在: {path}"
    if required:
        report.errors.append(message)
    else:
        report.warnings.append(message)


def _module_version(module_name: str, package_name: str | None = None) -> str:
    import_module(module_name)
    package = package_name or module_name
    try:
        return version(package)
    except PackageNotFoundError:
        module = import_module(module_name)
        return getattr(module, "__version__", "unknown")


def run_environment_self_check() -> EnvironmentCheckReport:
    logger = get_logger()
    report = EnvironmentCheckReport(summary=f"{APP_NAME} 运行环境自检")

    for path in [USER_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
        ensure_dir(path)

    report.details["运行模式"] = "PyInstaller" if is_frozen_app() else "source"
    report.details["当前工作目录"] = str(Path.cwd())
    report.details["资源目录"] = str(RESOURCE_ROOT)
    report.details["用户数据目录"] = str(USER_DATA_DIR)
    report.details["日志目录"] = str(LOGS_DIR)

    _check_file(report, ASSETS_DIR / "styles" / "dark_finance.qss", "样式文件")
    _check_file(report, DEMO_DIR / "sample_demo.csv", "示例数据")
    _check_file(report, EXAMPLES_DIR / "sample_inference.csv", "批量推理示例")

    data_csv = DATA_DIR / "PS_20174392719_1491204439457_log.csv"
    if data_csv.exists():
        report.details["主数据集"] = str(data_csv)
    else:
        report.warnings.append(f"主数据集不存在: {data_csv}，将回退到示例数据。")

    latest_model = sorted((MODELS_DIR / "training_runs").glob("**/model.pt"))[-1:] if (MODELS_DIR / "training_runs").exists() else []
    latest_preprocessor = sorted((MODELS_DIR / "training_runs").glob("**/preprocessor.joblib"))[-1:] if (MODELS_DIR / "training_runs").exists() else []
    if latest_model:
        report.details["最近模型"] = str(latest_model[0])
    else:
        report.warnings.append("未发现已训练模型文件 model.pt，评估/推理页面需先训练或手动选择模型。")
    if latest_preprocessor:
        report.details["最近预处理器"] = str(latest_preprocessor[0])
    else:
        report.warnings.append("未发现已训练预处理器 preprocessor.joblib，评估/推理页面需先训练或手动选择预处理器。")

    dependencies = [
        ("numpy", None),
        ("scipy", None),
        ("sklearn", "scikit-learn"),
        ("joblib", None),
        ("torch", None),
        ("PySide6", None),
        ("imblearn", "imbalanced-learn"),
    ]
    for module_name, package_name in dependencies:
        try:
            dep_version = _module_version(module_name, package_name)
            report.details[f"依赖 {module_name}"] = dep_version
        except Exception as exc:
            report.errors.append(f"依赖导入失败 {module_name}: {exc}")

    if is_frozen_app():
        internal_dir = Path(sys.executable).resolve().parent / "_internal"
        report.details["安装目录 _internal"] = str(internal_dir)
        if not internal_dir.exists():
            report.errors.append(f"安装目录缺少 _internal: {internal_dir}")

    if report.ok:
        logger.info("环境自检通过")
    else:
        logger.error("环境自检失败: %s", " | ".join(report.errors))
    for warning in report.warnings:
        logger.warning(warning)
    return report
