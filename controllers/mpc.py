import numpy as np

try:
    import cvxpy as cp  # 可选依赖：用于构建并求解二次规划（QP）
    _CVXPY_AVAILABLE = True
except Exception:
    cp = None
    _CVXPY_AVAILABLE = False


class MPCController:
    """
    倒立摆的线性模型预测控制（MPC）控制器（基于直立点线性化）。

    - 若安装了 cvxpy，则使用 QP 求解器（默认 OSQP）进行滚动优化
    - 若未安装 cvxpy，则回退为简单采样式 MPC（常值控制序列网格搜索）

    compute_control(theta, theta_dot, x, x_dot) -> u
    """

    def __init__(
        self,
        A_d: np.ndarray | None = None,
        B_d: np.ndarray | None = None,
        params=None,
        u_max: float | None = None,
        Ts: float | None = None,
        N: int = 30,  # 预测时域长度
        q_x: float = 20.0,
        q_xdot: float = 2.0,
        q_theta: float = 80.0,
        q_thetadot: float = 4.0,
        r_u: float = 0.05,
        x_max: float = 1.8,  # 位置软约束（动画轨道约 ±2m）
        solver: str = 'OSQP',
    ) -> None:
        # 参数与模型
        if params is not None and (A_d is None or B_d is None):
            from models.cartpole import CartPole
            A_list, B_list = CartPole(params).linearize_upright()
            A_d = np.array(A_list, dtype=np.float64)
            B_d = np.array(B_list, dtype=np.float64)
        if A_d is None or B_d is None:
            raise ValueError("MPC 需要提供 (A_d, B_d) 或 params 用于线性化")

        self.A = np.array(A_d, dtype=np.float64)
        self.B = np.array(B_d, dtype=np.float64)
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1] if self.B.ndim == 2 else 1

        # 基本配置
        self.u_max = float(u_max if u_max is not None else 10.0)
        self.Ts = float(Ts if Ts is not None else 0.01)
        self.N = int(N)
        self.x_max = float(x_max)

        # 权重矩阵
        self.Q = np.diag([q_x, q_xdot, q_theta, q_thetadot]).astype(np.float64)
        self.R = np.array([[r_u]], dtype=np.float64)
        # 终端权重：用离散代数Riccati的解作为 Qf，更利于收敛
        self.Qf = self._compute_terminal_P(self.A, self.B, self.Q, self.R)

        # cvxpy 问题（若可用则提前构建以便复用）
        self._use_cvxpy = _CVXPY_AVAILABLE
        self._solver = solver
        self._problem = None
        self._x0_param = None
        self._U_var = None
        if self._use_cvxpy:
            try:
                self._build_cvxpy_problem()
            except Exception:
                # 构建失败则回退为采样式 MPC
                self._use_cvxpy = False

    @staticmethod
    def _compute_terminal_P(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, max_iter: int = 500, tol: float = 1e-9) -> np.ndarray:
        """迭代求解离散代数Riccati方程，返回终端权重 P。"""
        P = Q.copy()
        for _ in range(max_iter):
            K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
            P_next = A.T @ P @ A - A.T @ P @ B @ K + Q
            if np.linalg.norm(P_next - P, ord='fro') < tol:
                P = P_next
                break
            P = P_next
        return P

    def _build_cvxpy_problem(self) -> None:
        nx, nu, N = self.nx, self.nu, self.N
        A, B = self.A, self.B

        X = cp.Variable((nx, N + 1))
        U = cp.Variable((nu, N))
        x0 = cp.Parameter(nx)

        cost = 0
        constraints = [X[:, 0] == x0]
        for k in range(N):
            cost += cp.quad_form(X[:, k], self.Q) + cp.quad_form(U[:, k], self.R)
            constraints += [X[:, k + 1] == A @ X[:, k] + B @ U[:, k]]
            constraints += [cp.abs(U[:, k]) <= self.u_max]
            # 位置约束：|x| ≤ x_max
            constraints += [cp.abs(X[0, k]) <= self.x_max]
        cost += cp.quad_form(X[:, N], self.Qf)
        constraints += [cp.abs(X[0, N]) <= self.x_max]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        # 保存句柄用于后续复用
        self._problem = prob
        self._x0_param = x0
        self._U_var = U

    def reset(self) -> None:
        # MPC 无内部状态需要显式重置
        pass

    def _solve_cvxpy(self, x0_vec: np.ndarray) -> float | None:
        self._x0_param.value = x0_vec
        try:
            self._problem.solve(solver=self._solver, warm_start=True, verbose=False)
            if self._U_var.value is None:
                return None
            u0 = float(self._U_var.value[0, 0])
            # 饱和保护
            u0 = max(-self.u_max, min(self.u_max, u0))
            return u0
        except Exception:
            return None

    def _fallback_sample(self, x0_vec: np.ndarray) -> float:
        """简易采样式 MPC：在常值控制网格中选择预测代价最小的第一步控制。"""
        A, B, Q, R, Qf = self.A, self.B, self.Q, self.R, self.Qf
        N = self.N
        # 候选控制（常值）：11 点网格
        candidates = np.linspace(-self.u_max, self.u_max, 11)
        best_u, best_cost = 0.0, float('inf')
        for u_const in candidates:
            x = x0_vec.copy()
            cost = 0.0
            for k in range(N):
                cost += float(x.T @ Q @ x) + float(R[0, 0] * (u_const ** 2))
                x = A @ x + B[:, 0] * u_const
                # 位置软约束违背惩罚
                if abs(x[0]) > self.x_max:
                    cost += 1e6 * (abs(x[0]) - self.x_max)
            cost += float(x.T @ Qf @ x)
            if abs(x[0]) > self.x_max:
                cost += 1e6 * (abs(x[0]) - self.x_max)
            if cost < best_cost:
                best_cost = cost
                best_u = u_const
        return float(best_u)

    def compute_control(self, theta: float, theta_dot: float, x: float, x_dot: float) -> float:
        # 状态向量遵循 linearize_upright 的顺序：[x, x_dot, theta, theta_dot]
        x_vec = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)

        if self._use_cvxpy and self._problem is not None:
            u = self._solve_cvxpy(x_vec)
            if u is not None:
                return u

        # 回退方案：采样式 MPC
        u_fb = self._fallback_sample(x_vec)
        # 饱和
        u_fb = max(-self.u_max, min(self.u_max, u_fb))
        return float(u_fb)


__all__ = ["MPCController"]