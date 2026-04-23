# 金融智盾 FraudShield

金融智盾 FraudShield 是一个基于 `PySide6 + PyTorch + scikit-learn + imbalanced-learn + cryptography` 构建的 Windows 本地桌面端金融欺诈检测原型系统。

当前桌面端主线围绕 PaySim 交易数据集进行设计，覆盖数据导入、预处理、SMOTE 平衡、MLP 训练、模型评估、单条/批量推理、模型加密解密、数据分析与模型对比模块。

## 功能概览

- PaySim 原始交易数据导入与结构校验
- 预处理与特征构建
- 类别不平衡处理（SMOTE 为主方案）
- MLP 模型训练、验证、测试
- ROC / PR / 混淆矩阵 / loss 曲线等评估看板
- 单条交易风险推理与批量 CSV 推理
- 模型权重加密、解密、SHA-256 完整性校验
- PCA / t-SNE 降维可视化与多模型对比

## 数据结构

系统默认支持 PaySim 结构：

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

## 开发环境启动

```powershell
conda activate fraudshield
cd e:\fraud_shield_software
python -m pip install -r requirements.txt
python app.py
```

也可以直接双击 `run_app.bat` 启动开发态程序。

## 打包为 Windows 发布版

### 1. 安装打包依赖

```powershell
conda activate fraudshield
cd e:\fraud_shield_software
python -m pip install -r requirements-build.txt
```

### 2. 构建 PyInstaller 发布版

```powershell
build_release.bat
```

或手动执行：

```powershell
python -m PyInstaller --noconfirm --clean FraudShield.spec
```

构建完成后输出目录为：

```text
dist\FraudShield\
```

### 3. 发布时需要一起交付的内容

- `FraudShield.exe`
- `assets/`
- `data/demo/`
- `examples/`
- PyInstaller 打包后生成的 `.dll` / `.pyd` 文件

## 安装包说明

仓库中已提供 Inno Setup 脚本模板：

- `installer\FraudShield.iss`

当你完成 `dist\FraudShield` 构建后，可使用 Inno Setup 将其打成可安装的 `.exe` 安装包。

## 打包后运行路径规则

为了避免安装在 `Program Files` 时出现写权限问题，程序在打包运行时会将模型、日志、导出结果等写入到用户目录：

```text
%LOCALAPPDATA%\FraudShield\
```

包括：

- `models\`
- `outputs\`
- `outputs\logs\`
- `outputs\predictions\`
- `outputs\encrypted\`
- `outputs\decrypted\`

程序资源（QSS、示例数据、内置推理 CSV）则随发布包一起走。

## 目录结构

```text
FraudShield/
|-- app.py
|-- requirements.txt
|-- requirements-build.txt
|-- FraudShield.spec
|-- build_release.bat
|-- run_app.bat
|-- installer/
|   `-- FraudShield.iss
|-- assets/
|   `-- styles/
|-- data/
|   `-- demo/
|-- examples/
|-- src/
|   |-- core/
|   |-- ui/
|   |-- utils/
|   `-- workers/
|-- models/      # 本地模型与训练产物，默认不上传 GitHub
|-- outputs/     # 本地运行输出，默认不上传 GitHub
|-- logs/        # 本地日志，默认不上传 GitHub
`-- README.md
```

## 备注

- GUI 与数据分析模块以现场演示为优先。
- 全量 PaySim 大文件训练仍需根据 CPU/GPU 能力和内存合理配置。
