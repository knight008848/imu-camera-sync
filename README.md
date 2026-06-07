# IMU-Camera Sync

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
![License](https://img.shields.io/badge/license-MIT-green)

iPhone IMU 和 Camera 数据清洗与时间同步工具。

## 功能

- **数据加载** — 读取 iPhone 传感器日志（加速度计、陀螺仪）和视频文件，统一为内部格式
- **数据清洗** — 离群值检测与去除、缺失值填充、时间戳异常修复
- **时间同步** — 插值对齐、最近邻匹配、重采样到统一时间轴
- **可视化检查** — 时间戳分布、对齐前后对比图

## 环境与依赖要求

- **Python**: >= 3.10
- **核心依赖**: `numpy`, `scipy`, `pandas`, `opencv-python-headless`, `matplotlib`
- **系统依赖**: 请确保您的系统已安装 **[FFmpeg](https://ffmpeg.org/)** 并将其加入了系统环境变量（使其在命令行中可调用 `ffprobe`）。工具在读取无 Odometry 的视频文件时，需要使用 `ffprobe` 提取视频的真实 UTC 创建时间。

## 安装

推荐在虚拟环境（如 venv 或 conda）中进行安装：

```bash
# 克隆仓库
git clone https://github.com/knight008848/imu-camera-sync.git
cd imu-camera-sync

# 安装核心依赖
pip install -e .

# 如果需要进行开发或运行测试，请安装附加依赖
pip install -e ".[dev]"
```

## 快速开始

### 推荐：使用 Pipeline 一键处理

如果你整理好的数据目录（如 `data/session_01`）包含 `imu.csv`、`rgb.mp4`，以及可选的 `odometry.csv`，推荐使用高级 API 一键完成所有操作，并自动生成 `aligned.csv` 和可视化图表：

```python
from imu_camera_sync import pipeline

# 处理单个会话目录 (开启 with_visualization 会在目录下生成对齐对比图)
result = pipeline.process_session("data/session_01", method="nearest", with_visualization=True)
print(f"对齐误差平均值: {result['alignment']['error_mean_ms']:.2f} ms")

# 批量处理包含多个子目录的数据根目录
batch_results = pipeline.batch_process("data_root/", method="nearest")
```

### 进阶：分步按需调用

您也可以根据需求分步调用各个模块：

```python
from imu_camera_sync import loader, cleaner, synchronizer

# 1. 加载数据
imu_data = loader.load_imu("data/imu.csv")
camera_data = loader.load_camera("data/video.mov") # 如果有的话，支持传入 odometry_path

# 2. 清洗数据（去除离群值、修复时间戳等）
imu_clean = cleaner.clean_imu(imu_data)
camera_clean = cleaner.clean_camera(camera_data)

# 3. 对齐数据（可选方法: 'nearest' 或 'interp'）
synced = synchronizer.align(imu_clean, camera_clean, method="nearest")

# 4. 导出为 CSV 文件
synchronizer.to_csv(synced, "aligned.csv")
```

## 目录结构

```text
imu-camera-sync/
├── src/imu_camera_sync/   # 核心源代码
├── examples/              # 使用示例
├── tests/                 # 测试用例
└── data/                  # 原始数据目录（已被 .gitignore 忽略）
```

## 开发与测试

本项目使用 `pytest` 进行单元测试，并强制使用 `ruff` 保证代码规范。

```bash
# 运行全部单元测试
pytest tests/

# 运行代码检查和格式化
ruff check src/
ruff format src/
```

## License

本项目基于 [MIT License](LICENSE) 协议开源。
