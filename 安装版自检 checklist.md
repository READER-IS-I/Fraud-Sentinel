# 安装版自检 checklist

## A. 安装前自检

### dist 目录结构

- [ ] `dist/FraudShield/FraudShield.exe` 存在
- [ ] `dist/FraudShield/_internal` 存在
- [ ] `dist/FraudShield/_internal/imblearn` 存在
- [ ] `dist/FraudShield/_internal/sklearn` 存在
- [ ] `dist/FraudShield/_internal/scipy` 存在
- [ ] `dist/FraudShield/_internal/PySide6` 存在
- [ ] `dist/FraudShield/assets/styles/dark_finance.qss` 存在
- [ ] `dist/FraudShield/data/demo/sample_demo.csv` 存在
- [ ] `dist/FraudShield/examples/sample_inference.csv` 存在

### 直接运行验证

- [ ] 在 `dist/FraudShield` 目录中直接启动 `FraudShield.exe`
- [ ] 主界面能够显示
- [ ] 未再出现 `imblearn/VERSION.txt` 相关启动崩溃
- [ ] 若功能页依赖缺失，错误应被限制在单功能，不应导致整程序退出

### 关键依赖与文件

- [ ] `torch` 相关 `.pyd/.dll` 已随包收集
- [ ] `numpy/scipy/sklearn/joblib/imblearn` 相关目录存在
- [ ] `PySide6/plugins/platforms/qwindows.dll` 存在
- [ ] 样式、示例数据、示例推理文件可读取

### 开发机路径残留

- [ ] 使用全文搜索检查 `dist` 中未残留明显开发机绝对路径
- [ ] 未发现 `D:\ABC\`、源码目录、开发虚拟环境路径被写入业务配置
- [ ] `.spec` 已使用 `Path(__file__).resolve().parent`，不再依赖构建时 `cwd`

## B. 安装后自检

### 启动验证

- [ ] 从安装目录直接启动 `FraudShield.exe`
- [ ] 主窗口能够显示
- [ ] 首次启动不会闪退

### 日志验证

- [ ] 已生成 `logs/startup.log`
- [ ] 已生成 `logs/error.log`
- [ ] `startup.log` 中能看到环境自检记录
- [ ] 若有错误，`error.log` 中能看到完整堆栈

### 页面打开验证

- [ ] 首页可打开
- [ ] 数据导入页可打开
- [ ] 数据分析页可打开
- [ ] 模型对比页可打开
- [ ] 模型训练页可打开
- [ ] 模型评估页可打开
- [ ] 风险推理页可打开
- [ ] 模型加密页可打开

### compare 功能验证

- [ ] 点击“模型对比”页不会在进入时崩溃
- [ ] 点击“开始对比”后，依赖可正常导入
- [ ] 若 compare 失败，只提示功能错误，不导致主程序退出

### 资源访问验证

- [ ] 样式文件可访问
- [ ] 示例数据可访问
- [ ] 示例推理 CSV 可访问
- [ ] 用户数据目录可创建
- [ ] `models/outputs/logs` 等运行目录可创建

### 训练/评估/推理功能验证

- [ ] 点击“开始训练”后能进入训练流程
- [ ] 训练完成后能生成 `model.pt` 和 `preprocessor.joblib`
- [ ] 评估页可加载训练产物
- [ ] 推理页可加载训练产物

## C. 建议判定标准

只有以下条件同时满足，才建议进入正式发布：

- [ ] `onedir` 自检通过
- [ ] 安装版自检通过
- [ ] compare/train/eval/predict 关键流程都已验证
- [ ] 日志中无新的启动级异常
