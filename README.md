# 先进控制课程报告：倒立摆（Cart-Pole）PID、LQR、模糊控制与MPC 对比实验

一个统一的 Python 仿真平台，对比四类控制方法：PID、LQR、模糊控制与MPC，在小角度稳摆与位置抑制任务下进行一致性评估。支持动画展示、统一对比三联图与指标导出。

### 无控制（参考基线）
<div align="center">
  <img src="结果/results_无控制/cartpole_animation.gif" alt="无控制 动画" width="360" />
</div>
<p align="center">
  <img src="结果/results_无控制/animate_trace_theta.png" alt="无控制 角度θ" width="260" />
  <img src="结果/results_无控制/animate_trace_x.png" alt="无控制 位置x" width="260" />
  <img src="结果/results_无控制/animate_trace_F.png" alt="无控制 力F" width="260" />
</p>

### PID 控制
<div align="center">
  <img src="结果/results_pid控制/cartpole_animation.gif" alt="PID 动画" width="360" />
</div>
<p align="center">
  <img src="结果/results_pid控制/animate_trace_theta.png" alt="PID 角度θ" width="260" />
  <img src="结果/results_pid控制/animate_trace_x.png" alt="PID 位置x" width="260" />
  <img src="结果/results_pid控制/animate_trace_F.png" alt="PID 力F" width="260" />
</p>

### LQR 控制
<div align="center">
  <img src="结果/results_lqr控制/cartpole_animation.gif" alt="LQR 动画" width="360" />
</div>
<p align="center">
  <img src="结果/results_lqr控制/animate_trace_theta.png" alt="LQR 角度θ" width="260" />
  <img src="结果/results_lqr控制/animate_trace_x.png" alt="LQR 位置x" width="260" />
  <img src="结果/results_lqr控制/animate_trace_F.png" alt="LQR 力F" width="260" />
</p>

### MPC 控制
<div align="center">
  <img src="结果/results_mpc控制/cartpole_animation.gif" alt="MPC 动画" width="360" />
</div>
<p align="center">
  <img src="结果/results_mpc控制/animate_trace_theta.png" alt="MPC 角度θ" width="260" />
  <img src="结果/results_mpc控制/animate_trace_x.png" alt="MPC 位置x" width="260" />
  <img src="结果/results_mpc控制/animate_trace_F.png" alt="MPC 力F" width="260" />
</p>

### 模糊控制
<div align="center">
  <img src="结果/results_模糊控制/cartpole_animation.gif" alt="模糊控制 动画" width="360" />
</div>
<p align="center">
  <img src="结果/results_模糊控制/animate_trace_theta.png" alt="模糊 角度θ" width="260" />
  <img src="结果/results_模糊控制/animate_trace_x.png" alt="模糊 位置x" width="260" />
  <img src="结果/results_模糊控制/animate_trace_F.png" alt="模糊 力F" width="260" />
</p>

## 目录结构
- `models/` 倒立摆非线性动力学与数值步进
- `controllers/` 四类控制器实现：`pid.py`、`lqr.py`、`fuzzy.py`、`mpc.py`
- `experiments/` 动画与统一对比脚本：`animate_cartpole.py`、`run_scenarios.py`
- `results_*` 运行输出：动画轨迹、状态与力曲线、统一对比图与指标CSV
- `main.py` 菜单式入口（含动画与统一对比）
- `先进控制基础课程报告.pdf` 课程报告


## 快速开始
- 菜单入口：`python main.py`，运行后根据菜单选择不同功能：
``` shell
=== 倒立摆演示 主菜单 ===
1) 动画：无控制（开环）
2) 动画：模糊控制器
3) 动画：PID控制器
4) 动画：LQR控制器
5) 动画：MPC控制器
6) 统一对比仿真（模糊/PID/LQR/MPC）
0) 退出
请选择（数字）：6
```