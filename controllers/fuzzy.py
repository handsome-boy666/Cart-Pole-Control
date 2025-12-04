from typing import Tuple

import numpy as np


class FuzzyController:
    """
    简化的模糊控制器（含位置回零）：
    输入：theta (rad), theta_dot (rad/s)，可选 x (m), x_dot (m/s)
    输出：u (N)

    设计思路：按照倒立摆直立附近的常识规则：
    - 若角度偏离较大（正/负），给出与角度同号的较大力（将其拉回直立并使角速度朝向减小）
    - 若角速度很大，优先用力抵消角速度（防止越过）
    - 若位置偏离原点（x≠0），加入回零项（与位置同号的反向力）
    - 在小角度/小角速度/小位移时，力趋近于零

    采用连续函数近似模糊推理，以降低依赖（避免强耦合 scikit-fuzzy）。
    """

    def __init__(self, u_max: float = 10.0) -> None:
        self.u_max = u_max
        # 灵敏度系数，可调
        self.k_theta = 50.0   # 对角度的放大（rad -> N）
        self.k_dtheta = 5.0   # 对角速度的放大（rad/s -> N）
        # 加入位置回零项（默认较温和，可按需调大）
        self.k_x = 5.0        # 对位置的放大（m -> N）
        self.k_dx = 2.0       # 对速度的放大（m/s -> N）
        # 非线性压缩避免过激（tanh）

    def compute_control(self, theta: float, theta_dot: float, x: float = None, x_dot: float = None) -> float:
        # 基础线性组合（角度与角速度）
        u_raw = self.k_theta * theta + self.k_dtheta * theta_dot
        # 若提供位置与速度，则加入回零项（反向力）
        if x is not None:
            u_raw += self.k_x * x
        if x_dot is not None:
            u_raw += self.k_dx * x_dot
        # 非线性压缩（使响应在大角度时不过度增大）
        u = np.tanh(0.2 * u_raw) * self.u_max
        # 饱和
        return float(max(-self.u_max, min(self.u_max, u)))


def demo_run() -> None:
    from models.cartpole import CartPole, CartPoleParams
    import matplotlib.pyplot as plt

    sys = CartPole(CartPoleParams())
    fz = FuzzyController(u_max=sys.p.umax)
    Ts = sys.p.Ts
    sys.reset((0.0, 0.0, 0.1, 0.0))

    T = 5.0
    steps = int(T / Ts)
    xs, us, ts = [], [], []
    for k in range(steps):
        x = sys.get_state()
        theta, theta_dot = x[2], x[3]
        u = fz.compute_control(theta, theta_dot)
        sys.step(u)
        xs.append(sys.get_state())
        us.append(u)
        ts.append(k * Ts)

    xs = np.array(xs)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(ts, xs[:, 2], label='theta (rad)')
    plt.plot(ts, xs[:, 0], label='x (m)')
    plt.legend(); plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(ts, us, label='u (N)')
    plt.legend(); plt.grid(True)
    plt.suptitle('Fuzzy demo (cart-pole)')
    plt.show()


if __name__ == "__main__":
    demo_run()