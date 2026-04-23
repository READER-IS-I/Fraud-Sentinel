# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules


spec_path = Path(globals().get("__file__", os.path.abspath("FraudShield.spec"))).resolve()
project_root = spec_path.parent
block_cipher = None


def _dedupe(items):
    seen = set()
    result = []
    for item in items:
        key = tuple(item) if isinstance(item, (list, tuple)) else item
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


datas = [
    (str(project_root / "assets"), "assets"),
    (str(project_root / "data"), "data"),
    (str(project_root / "examples"), "examples"),
]
binaries = []
hiddenimports = [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
]

# Local modules loaded via import_module() are invisible to static analysis.
hiddenimports += collect_submodules("src")


for package_name in [
    "imblearn",
    "sklearn",
    "joblib",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "PySide6",
    "cryptography",
    "torch",
]:
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports


binaries += collect_dynamic_libs("torch")

datas = _dedupe(datas)
binaries = _dedupe(binaries)
hiddenimports = sorted(set(hiddenimports))


a = Analysis(
    [str(project_root / "app.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FraudShield",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FraudShield",
)
