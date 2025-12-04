import os
import sys
# 允许从项目根目录导入模块（解决从 experiments 目录运行时的相对导入问题）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import math
import argparse
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from models.cartpole import CartPole, CartPoleParams


def ensure_dirs():
    os.makedirs('results', exist_ok=True)


def simulate(
    controller: Optional[str] = None,
    T: float = 8.0,
    theta0_deg: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    模拟倒立摆运动。
    controller: None / 'fuzzy' / 'pid' / 'lqr' / 'mpc'
    返回：ts, xs(states), us
    """
    sys = CartPole(CartPoleParams())
    theta0 = math.radians(theta0_deg)
    sys.reset((0.0, 0.0, theta0, 0.0))

    Ts = sys.p.Ts
    steps = int(T / Ts)
    ts, xs, us = [], [], []

    ctrl = None
    if controller == 'fuzzy':
        from controllers.fuzzy import FuzzyController
        ctrl = FuzzyController(u_max=sys.p.umax)
    elif controller == 'pid':
        from controllers.pid import PIDController
        ctrl = PIDController(u_max=sys.p.umax, Ts=sys.p.Ts)
    elif controller == 'lqr':
        from controllers.lqr import LQRController
        # 使用直立点线性化，初始化 LQR
        A_d, B_d = sys.linearize_upright()
        ctrl = LQRController(A_d=np.array(A_d), B_d=np.array(B_d), params=CartPoleParams(), u_max=sys.p.umax, Ts=sys.p.Ts)
    elif controller == 'mpc':
        from controllers.mpc import MPCController
        A_d, B_d = sys.linearize_upright()
        ctrl = MPCController(A_d=np.array(A_d), B_d=np.array(B_d), params=CartPoleParams(), u_max=sys.p.umax, Ts=sys.p.Ts, N=30)

    # 若控制器包含内部状态，仿真前初始化一次
    if ctrl is not None and hasattr(ctrl, 'reset'):
        try:
            ctrl.reset()
        except Exception:
            pass

    for k in range(steps):
        x = np.array(sys.get_state(), dtype=np.float32)
        if controller is None:
            u = 0.0
        elif controller == 'fuzzy':
            u = float(ctrl.compute_control(theta=float(x[2]), theta_dot=float(x[3]), x=float(x[0]), x_dot=float(x[1])))
        elif controller == 'pid':
            u = float(ctrl.compute_control(theta=float(x[2]), theta_dot=float(x[3]), x=float(x[0]), x_dot=float(x[1])))
        elif controller == 'lqr':
            u = float(ctrl.compute_control(theta=float(x[2]), theta_dot=float(x[3]), x=float(x[0]), x_dot=float(x[1])))
        elif controller == 'mpc':
            u = float(ctrl.compute_control(theta=float(x[2]), theta_dot=float(x[3]), x=float(x[0]), x_dot=float(x[1])))
        else:
            raise ValueError('unknown controller')

        sys.step(u)
        xs.append(sys.get_state())
        us.append(u)
        ts.append(k * Ts)

    return np.array(ts), np.array(xs), np.array(us)


def to_coords(x: float, theta: float, l: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """由状态计算支点与端点坐标（y向上，x向右）。"""
    px, py = x, 0.0
    bx = x + l * math.sin(theta)
    by = l * math.cos(theta)
    return (px, py), (bx, by)


def animate_run(ts: np.ndarray, xs: np.ndarray, us: np.ndarray, save: bool = True, fname: Optional[str] = None, fps: int = 60):
    ensure_dirs()
    l = CartPoleParams().l
    Ts = CartPoleParams().Ts
    sim_fps = 1.0 / Ts
    # 抽帧步长：控制保存/播放的帧率，同时保证总时长与仿真时长一致
    step = max(1, int(round(sim_fps / float(fps))))
    frame_indices = list(range(0, len(ts), step))
    interval_ms = Ts * step * 1000.0

    # 设定画布
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.5, 1.5)
    # 让横纵坐标单位长度一致（等比例显示）
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(True)
    ax.set_title('Cart-Pole Animation')

    # 轨道与图元
    track, = ax.plot([-2.0, 2.0], [0, 0], 'k-', lw=2)
    cart = plt.Rectangle((0, 0), 0.3, 0.2, fc='tab:blue', ec='k')
    ax.add_patch(cart)
    rod, = ax.plot([], [], 'r-', lw=2)
    bob, = ax.plot([], [], 'ro', ms=8)

    time_text = ax.text(0.02, 0.94, '', transform=ax.transAxes)

    def init():
        rod.set_data([], [])
        bob.set_data([], [])
        cart.set_xy((-0.15, -0.1))
        time_text.set_text('')
        return rod, bob, cart, time_text

    def update(i):
        x, _, theta, _ = xs[i]
        (px, py), (bx, by) = to_coords(x, theta, l)
        # 更新小车矩形（以中心对齐）
        cart_width, cart_height = 0.3, 0.2
        cart.set_xy((x - cart_width / 2.0, -cart_height / 2.0))
        # 更新杆与端点
        rod.set_data([px, bx], [py, by])
        bob.set_data([bx], [by])
        time_text.set_text(f't = {ts[i]:.2f}s')
        return rod, bob, cart, time_text

    anim = animation.FuncAnimation(fig, update, frames=frame_indices, init_func=init, interval=interval_ms, blit=True)

    if save:
        gif_name = fname or 'results/cartpole_animation.gif'
        mp4_name = 'results/cartpole_animation.mp4'
        fps_out = max(1, int(round(sim_fps / step)))

        # 同时尝试保存 GIF
        try:
            from matplotlib.animation import PillowWriter
            anim.save(gif_name, writer=PillowWriter(fps=fps_out))
            print(f'动画GIF已保存：{gif_name}')
        except Exception as e:
            print('保存GIF失败：', e)

        # 同时尝试保存 MP4（需要系统安装 ffmpeg）
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps_out, metadata=dict(artist='cartpole'))
            anim.save(mp4_name, writer=writer)
            print(f'动画MP4已保存：{mp4_name}')
        except Exception as e2:
            print('保存MP4失败（可能未安装ffmpeg）：', e2)
            # 若两者都失败则直接显示
            if not os.path.exists(gif_name):
                plt.show()
    else:
        plt.show()


def save_trace(ts: np.ndarray, xs: np.ndarray, us: np.ndarray, path: str):
    ensure_dirs()
    # 保存CSV（不含加速度列）
    with open(path, 'w', encoding='utf-8') as f:
        f.write('t,x,x_dot,theta,theta_dot,u\n')
        for i in range(len(ts)):
            x, x_dot, theta, theta_dot = xs[i]
            f.write(f'{ts[i]},{x},{x_dot},{theta},{theta_dot},{us[i]}\n')
    print(f'轨迹已保存：{path}')

    # 同步生成静态曲线图：分别保存 theta、x、theta_dot、x_dot、F
    try:
        base, _ = os.path.splitext(path)
        # theta
        fig_theta = base + '_theta.png'
        plt.figure(figsize=(10, 4))
        plt.plot(ts, xs[:, 2], label='theta (rad)')
        plt.ylabel('theta (rad)'); plt.xlabel('t (s)'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(fig_theta, dpi=150)
        print(f'theta 曲线图已保存：{fig_theta}')

        # x
        fig_x = base + '_x.png'
        plt.figure(figsize=(10, 4))
        plt.plot(ts, xs[:, 0], label='x (m)')
        plt.ylabel('x (m)'); plt.xlabel('t (s)'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(fig_x, dpi=150)
        print(f'x 曲线图已保存：{fig_x}')

        # theta_dot
        fig_theta_dot = base + '_theta_dot.png'
        plt.figure(figsize=(10, 4))
        plt.plot(ts, xs[:, 3], label='theta_dot (rad/s)')
        plt.ylabel('theta_dot (rad/s)'); plt.xlabel('t (s)'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(fig_theta_dot, dpi=150)
        print(f'theta_dot 曲线图已保存：{fig_theta_dot}')

        # x_dot
        fig_x_dot = base + '_x_dot.png'
        plt.figure(figsize=(10, 4))
        plt.plot(ts, xs[:, 1], label='x_dot (m/s)')
        plt.ylabel('x_dot (m/s)'); plt.xlabel('t (s)'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(fig_x_dot, dpi=150)
        print(f'x_dot 曲线图已保存：{fig_x_dot}')

        # F（原u）
        fig_F = base + '_F.png'
        plt.figure(figsize=(10, 4))
        plt.plot(ts, us, label='F')
        plt.ylabel('F'); plt.xlabel('t (s)'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(fig_F, dpi=150)
        print(f'F 曲线图已保存：{fig_F}')
    except Exception as e:
        print('保存轨迹曲线图失败：', e)


def main():
    """
    主函数：解析命令行参数，运行仿真并生成动画。
    支持的控制器：无控制器、模糊控制、PID控制、DQN（需已训练模型）。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加控制器类型参数
    parser.add_argument('--controller', type=str, default=None, choices=[None, 'fuzzy', 'pid', 'lqr', 'mpc'], help='控制器类型')
    # 添加仿真总时长参数（秒）
    parser.add_argument('--T', type=float, default=8.0, help='仿真时长（秒）')
    # 添加初始摆角参数（度，顺时针为正）
    parser.add_argument('--theta0_deg', type=float, default=15.0, help='初始角度（度，顺时针为正）')
    # 添加是否保存动画的参数（若指定则仅显示不保存）
    parser.add_argument('--nosave', action='store_true', help='不保存动画，仅显示')
    parser.add_argument('--fps', type=int, default=60, help='动画播放帧率（确保与仿真时长一致）')
    # 解析命令行参数
    args = parser.parse_args()

    # 运行仿真，获取时间序列、状态序列和控制序列
    ts, xs, us = simulate(controller=args.controller, T=args.T, theta0_deg=args.theta0_deg)
    # 保存轨迹CSV（满足测试结果保存要求）
    save_trace(ts, xs, us, 'results/animate_trace.csv')
    # 生成/显示动画，根据参数决定是否保存
    animate_run(ts, xs, us, save=(not args.nosave), fps=args.fps)


if __name__ == '__main__':
    main()