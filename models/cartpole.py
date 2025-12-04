import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CartPoleParams:
    M: float = 0.5      # 小车质量 (kg)
    m: float = 0.2      # 摆杆质量 (kg)
    l: float = 0.3      # 摆杆质心到支点距离 (m)
    g: float = 9.81     # 重力加速度 (m/s^2)
    Ts: float = 0.01   # 采样周期 (s)
    umax: float = 10.0  # 最大推力 (N)


class CartPole:
    """
    倒立摆非线性仿真模型，采用显式欧拉积分。
    状态 x = [x, x_dot, theta, theta_dot] = [x1, x2, x3, x4]
    控制 u = F (N)，限制在 [-umax, umax]

    角度约定：theta 为与竖直方向的夹角（顺时针为正），theta=0 为竖直向上。
    """

    def __init__(self, params: CartPoleParams = CartPoleParams()) -> None:
        self.p = params
        self.reset()

    def reset(self, x0: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> Tuple[float, float, float, float]:
        self.x = list(x0)
        return tuple(self.x)

    def _derivatives(self, state: Tuple[float, float, float, float], u: float) -> Tuple[float, float, float, float]:
        """连续时间动力学导数 f(x, u) = dx/dt，忽略摩擦。"""
        x, x_dot, theta, theta_dot = state
        M, m, l, g = self.p.M, self.p.m, self.p.l, self.p.g
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        denom = M + m * (sin_t * sin_t)

        x_acc = (u + m * l * (theta_dot ** 2) * sin_t - m * g * cos_t * sin_t) / denom
        theta_acc = (-u * cos_t - m * l * (theta_dot ** 2) * sin_t * cos_t + (M + m) * g * sin_t) / (l * denom)

        return (x_dot, x_acc, theta_dot, theta_acc)

    def step(self, u: float) -> Tuple[float, float, float, float]:
        """单步积分（RK4），并进行输入饱和。"""
        # 饱和
        u = max(-self.p.umax, min(self.p.umax, u))
        Ts = self.p.Ts
        x0 = tuple(self.x)

        # RK4
        k1 = self._derivatives(x0, u)
        x1 = tuple(x0[i] + 0.5 * Ts * k1[i] for i in range(4))
        k2 = self._derivatives(x1, u)
        x2 = tuple(x0[i] + 0.5 * Ts * k2[i] for i in range(4))
        k3 = self._derivatives(x2, u)
        x3 = tuple(x0[i] + Ts * k3[i] for i in range(4))
        k4 = self._derivatives(x3, u)

        x_next = [
            x0[i] + (Ts / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            for i in range(4)
        ]

        self.x = x_next
        return tuple(self.x)

    def get_state(self) -> Tuple[float, float, float, float]:
        return tuple(self.x)

    def linearize_upright(self) -> Tuple[list, list]:
        """
        在直立平衡点 theta=0 处对上述给定动力学的连续线性化，
        并做前向欧拉离散化：x_{k+1} = A_d x_k + B_d u_k。

        小角度近似（忽略高阶项、无摩擦）：
        ẍ ≈ (1/M) * u - (m*g/M) * theta
        θ̈ ≈ -(1/(l*M)) * u + ((M+m)*g/(l*M)) * theta

        因此：
        A_c = [[0, 1, 0, 0],
               [0, 0, -(m*g)/M, 0],
               [0, 0, 0, 1],
               [0, 0, ((M+m)*g)/(l*M), 0]]
        B_c = [[0], [1/M], [0], [-1/(l*M)]]
        """
        M, m, l, g, Ts = self.p.M, self.p.m, self.p.l, self.p.g, self.p.Ts

        A_c = [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -(m * g) / M, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, ((M + m) * g) / (l * M), 0.0],
        ]
        B_c = [[0.0], [1.0 / M], [0.0], [-(1.0 / (l * M))]]

        # 离散化（显式欧拉）：A_d = I + A_c*Ts, B_d = B_c*Ts
        I = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        A_d = [[I[i][j] + Ts * A_c[i][j] for j in range(4)] for i in range(4)]
        B_d = [[Ts * B_c[i][0]] for i in range(4)]
        return A_d, B_d