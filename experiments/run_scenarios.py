import os
import sys
# 允许从项目根目录导入模块（解决从 experiments 目录运行时的相对导入问题）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from models.cartpole import CartPole, CartPoleParams
from controllers.fuzzy import FuzzyController
from controllers.pid import PIDController
from controllers.lqr import LQRController
from controllers.mpc import MPCController


def ensure_dirs():
    os.makedirs('results', exist_ok=True)


def simulate_controller(name: str, sys: CartPole, controller, T: float = 6.0) -> Dict[str, np.ndarray]:
    Ts = sys.p.Ts
    steps = int(T / Ts)
    sys.reset((0.0, 0.0, 0.1, 0.0))  # 轻微偏差

    xs, us, ts = [], [], []

    # 若控制器有内部状态，先重置一次
    if hasattr(controller, 'reset'):
        try:
            controller.reset()
        except Exception:
            pass

    for k in range(steps):
        x = np.array(sys.get_state(), dtype=np.float32)
        if isinstance(controller, FuzzyController):
            u = controller.compute_control(theta=float(x[2]), theta_dot=float(x[3]), x=float(x[0]), x_dot=float(x[1]))
        elif isinstance(controller, PIDController):
            u = controller.compute_control(theta=float(x[2]), theta_dot=float(x[3]), x=float(x[0]), x_dot=float(x[1]))
        elif isinstance(controller, LQRController):
            u = controller.compute_control(theta=float(x[2]), theta_dot=float(x[3]), x=float(x[0]), x_dot=float(x[1]))
        elif isinstance(controller, MPCController):
            u = controller.compute_control(theta=float(x[2]), theta_dot=float(x[3]), x=float(x[0]), x_dot=float(x[1]))
        else:
            raise ValueError('Unknown controller type')

        sys.step(u)
        xs.append(sys.get_state())
        us.append(u)
        ts.append(k * Ts)

    return {
        'name': name,
        'ts': np.array(ts),
        'xs': np.array(xs),
        'us': np.array(us),
    }


def compute_metrics(run: Dict[str, np.ndarray]) -> Dict[str, float]:
    xs, us, ts = run['xs'], run['us'], run['ts']
    theta = xs[:, 2]
    # 指标：theta 的 IAE/ISE、最大绝对值、控制能量
    iae = float(np.trapz(np.abs(theta), ts))
    ise = float(np.trapz(theta**2, ts))
    max_abs_theta = float(np.max(np.abs(theta)))
    energy = float(np.trapz(us**2, ts))
    return {
        'IAE_theta': iae,
        'ISE_theta': ise,
        'MaxAbsTheta': max_abs_theta,
        'Energy_u2': energy,
    }


def main():
    ensure_dirs()
    sys = CartPole(CartPoleParams())
    T_total = 15.0  # 统一对比的仿真总时长与横坐标范围（秒）

    # 控制器
    fuzzy = FuzzyController(u_max=sys.p.umax)
    pid = PIDController(u_max=sys.p.umax, Ts=sys.p.Ts)
    # LQR 使用线性化模型初始化
    A_d, B_d = sys.linearize_upright()
    import numpy as np
    lqr = LQRController(A_d=np.array(A_d), B_d=np.array(B_d), params=CartPoleParams(), u_max=sys.p.umax, Ts=sys.p.Ts)
    # MPC 控制器（线性化 + QP；若无cvxpy则回退为采样式）
    mpc = MPCController(A_d=np.array(A_d), B_d=np.array(B_d), params=CartPoleParams(), u_max=sys.p.umax, Ts=sys.p.Ts, N=30)

    # 仿真
    runs: List[Dict[str, np.ndarray]] = []
    runs.append(simulate_controller('Fuzzy', sys, fuzzy, T=T_total))
    runs.append(simulate_controller('PID', sys, pid, T=T_total))
    runs.append(simulate_controller('LQR', sys, lqr, T=T_total))
    runs.append(simulate_controller('MPC', sys, mpc, T=T_total))

    # 画图并保存
    ts = runs[0]['ts']
    plt.figure(figsize=(12, 8))
    # theta 对比
    plt.subplot(3, 1, 1)
    for r in runs:
        plt.plot(r['ts'], r['xs'][:, 2], label=f"{r['name']}")
    plt.ylabel('theta (rad)'); plt.legend(); plt.grid(True); plt.xlim(0.0, T_total)
    # x 对比
    plt.subplot(3, 1, 2)
    for r in runs:
        plt.plot(r['ts'], r['xs'][:, 0], label=f"{r['name']}")
    plt.ylabel('x (m)'); plt.legend(); plt.grid(True); plt.xlim(0.0, T_total)
    # u 对比
    plt.subplot(3, 1, 3)
    for r in runs:
        plt.plot(r['ts'], r['us'], label=f"{r['name']}")
    plt.ylabel('F(N)'); plt.xlabel('t (s)'); plt.legend(); plt.grid(True); plt.xlim(0.0, T_total)

    ts_str = time.strftime('%Y%m%d_%H%M%S')
    fig_path = f'results/compare_{ts_str}.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f'仿真曲线已保存：{fig_path}')

    # 指标保存
    csv_path = f'results/metrics_{ts_str}.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('controller,IAE_theta,ISE_theta,MaxAbsTheta,Energy_u2\n')
        for r in runs:
            m = compute_metrics(r)
            f.write(f"{r['name']},{m['IAE_theta']},{m['ISE_theta']},{m['MaxAbsTheta']},{m['Energy_u2']}\n")
    print(f'指标已保存：{csv_path}')


if __name__ == '__main__':
    main()
