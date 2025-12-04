import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from models.cartpole import CartPoleParams
except Exception:
    CartPoleParams = None


@dataclass
class LQRConfig:
    # 离散 LQR 权重（直立点线性化）
    q_x: float = 1.0
    q_xdot: float = 0.2
    q_theta: float = 20.0
    q_thetadot: float = 2.0
    r_u: float = 0.2
    # 迭代求解 Riccati 的参数
    max_iter: int = 500
    tol: float = 1e-8


class LQRController:
    """
    倒立摆直立点的离散 LQR 控制器。
    compute_control(theta, theta_dot, x, x_dot) -> u
    """

    def __init__(
        self,
        A_d: Optional[np.ndarray] = None,
        B_d: Optional[np.ndarray] = None,
        params: Optional[CartPoleParams] = None,
        u_max: Optional[float] = None,
        Ts: Optional[float] = None,
        cfg: Optional[LQRConfig] = None,
    ) -> None:
        if params is None and CartPoleParams is not None:
            params = CartPoleParams()
        self.p = params
        self.u_max = float(u_max if u_max is not None else (params.umax if params is not None else 10.0))
        self.Ts = float(Ts if Ts is not None else (params.Ts if params is not None else 0.01))
        self.cfg = cfg or LQRConfig()

        # 获取线性化的离散模型
        if (A_d is None or B_d is None) and params is not None:
            from models.cartpole import CartPole
            A_d_list, B_d_list = CartPole(params).linearize_upright()
            A_d = np.array(A_d_list, dtype=np.float64)
            B_d = np.array(B_d_list, dtype=np.float64)
        elif A_d is None or B_d is None:
            raise ValueError("LQR 需要提供 (A_d, B_d) 或 params 用于线性化")

        self.A = np.array(A_d, dtype=np.float64)
        self.B = np.array(B_d, dtype=np.float64)

        # 权重矩阵
        c = self.cfg
        self.Q = np.diag([c.q_x, c.q_xdot, c.q_theta, c.q_thetadot])
        self.R = np.array([[c.r_u]], dtype=np.float64)

        # 迭代求解离散代数 Riccati 方程，得到 P 与反馈增益 K
        self.K = self._dlqr(self.A, self.B, self.Q, self.R, c.max_iter, c.tol)

    @staticmethod
    def _dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
        P = Q.copy()
        AT = A.T
        BT = B.T
        for _ in range(max_iter):
            PB = P @ B
            S = R + BT @ PB
            K = np.linalg.solve(S, BT @ P @ A)  # (R + B^T P B)^{-1} B^T P A
            P_next = AT @ P @ A - AT @ PB @ K + Q
            if np.linalg.norm(P_next - P, ord='fro') < tol:
                P = P_next
                break
            P = P_next
        # 最终增益
        K = np.linalg.solve(R + BT @ P @ B, BT @ P @ A)
        return K

    def reset(self) -> None:
        pass

    def compute_control(self, theta: float, theta_dot: float, x: float, x_dot: float) -> float:
        # 状态按 linearize_upright 的顺序：[x, x_dot, theta, theta_dot]
        x_vec = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        u = float(-self.K @ x_vec)
        # 饱和
        u = max(-self.u_max, min(self.u_max, u))
        return u


__all__ = ["LQRController", "LQRConfig"]