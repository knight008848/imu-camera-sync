# IMU-Camera Sync

iPhone IMU 和 Camera 数据清洗与时间同步工具。

## 功能

- **数据加载** — 读取 iPhone 传感器日志（加速度计、陀螺仪）和视频文件，统一为内部格式
- **数据清洗** — 离群值检测与去除、缺失值填充、时间戳异常修复
- **时间同步** — 插值对齐、最近邻匹配、重采样到统一时间轴
- **可视化检查** — 时间戳分布、对齐前后对比图

## 安装

```bash
pip install -e .
```

## 快速开始

```python
from imu_camera_sync import loader, cleaner, synchronizer

# 加载数据
imu_data = loader.load_imu("data/imu.csv")
camera_data = loader.load_camera("data/video.mov")

# 清洗
imu_clean = cleaner.clean_imu(imu_data)
camera_clean = cleaner.clean_camera(camera_data)

# 对齐
synced = synchronizer.align(imu_clean, camera_clean)
```

## 目录结构

```
imu-camera-sync/
├── src/imu_camera_sync/   # 源代码
├── examples/              # 使用示例
├── tests/                 # 测试
└── data/                  # 原始数据（gitignore）
```
