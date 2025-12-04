import numpy as np


class PIDController:
    """基于角度与位置的PID控制器，含简单抗积分饱和与输入饱和。"""

    def __init__(
        self,
        u_max: float = 10.0,
        Ts: float = 0.01,
        kp_theta: float = 60.0,
        kd_theta: float = 6.0,
        ki_theta: float = 2.0,
        kp_x: float = 6.0,
        kd_x: float = 2.5,
        ki_x: float = 0.5,
    ) -> None:
        self.u_max = float(u_max)
        self.Ts = float(Ts)
        # 增益（默认偏向先稳杆，适度回中心）
        self.kp_theta = float(kp_theta)
        self.kd_theta = float(kd_theta)
        self.ki_theta = float(ki_theta)
        self.kp_x = float(kp_x)
        self.kd_x = float(kd_x)
        self.ki_x = float(ki_x)

        # 积分项
        self.int_theta = 0.0
        self.int_x = 0.0

    def reset(self) -> None:
        self.int_theta = 0.0
        self.int_x = 0.0

    def compute_control(
        self,
        theta: float,
        theta_dot: float,
        x: float | None = None,
        x_dot: float | None = None,
    ) -> float:
        # 线性项（符号与现有模糊控制保持一致）
        u_lin = self.kp_theta * theta + self.kd_theta * theta_dot + self.ki_theta * self.int_theta
        if x is not None:
            u_lin += self.kp_x * x + self.ki_x * self.int_x
        if x_dot is not None:
            u_lin += self.kd_x * x_dot

        # 饱和
        u = float(np.clip(u_lin, -self.u_max, self.u_max))

        # 抗积分饱和：当输出处于饱和且线性项推动继续饱和时，暂停积分
        eps = 1e-9
        push_same_dir = (abs(u) > self.u_max - eps) and (np.sign(u_lin) == np.sign(u))
        if not push_same_dir:
            self.int_theta += theta * self.Ts
            if x is not None:
                self.int_x += x * self.Ts

        return u


__all__ = ["PIDController"]