import os
import sys
import time

# 允许从项目根目录导入模块（解决从子目录运行时的相对导入问题）
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def ensure_dirs():
    os.makedirs('results', exist_ok=True)


def run_animation(controller=None, T=8.0, theta0_deg=15.0, fps=60):
    from experiments.animate_cartpole import simulate, animate_run, save_trace
    print(f"开始动画仿真：controller={controller}, T={T}s, theta0={theta0_deg}°, fps={fps}")
    ts, xs, us = simulate(controller=controller, T=T, theta0_deg=theta0_deg)
    save_trace(ts, xs, us, 'results/animate_trace.csv')
    animate_run(ts, xs, us, save=True, fps=fps)
    print("动画仿真完成：轨迹CSV与GIF/MP4已保存到 results/ 目录。")


def run_compare():
    print("开始统一对比仿真（模糊/PID/LQR/MPC）...")
    try:
        from experiments import run_scenarios
        run_scenarios.main()
        print("对比仿真完成：图表与指标CSV已保存到 results/ 目录。")
    except Exception as e:
        print("运行对比仿真失败：", e)


    


def _input_float(prompt: str, default: float) -> float:
    s = input(f"{prompt}（默认 {default}）：").strip()
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        print("输入非法，使用默认值。")
        return default


def print_menu():
    print("\n=== 倒立摆演示 主菜单 ===")
    print("1) 动画：无控制（开环）")
    print("2) 动画：模糊控制器")
    print("3) 动画：PID控制器")
    print("4) 动画：LQR控制器")
    print("5) 动画：MPC控制器")
    print("6) 统一对比仿真（模糊/PID/LQR/MPC）")
    print("0) 退出")


def main():
    ensure_dirs()
    while True:
        print_menu()
        choice = input("请选择（数字）：").strip()
        if choice == '0':
            print("退出程序。")
            break
        elif choice == '1':
            T = _input_float("仿真时长 T (秒)", 8.0)
            theta0 = _input_float("初始角度 theta0 (度)", 15.0)
            fps = int(_input_float("动画帧率 fps", 60))
            run_animation(controller=None, T=T, theta0_deg=theta0, fps=fps)
        elif choice == '2':
            T = _input_float("仿真时长 T (秒)", 8.0)
            theta0 = _input_float("初始角度 theta0 (度)", 10.0)
            fps = int(_input_float("动画帧率 fps", 60))
            run_animation(controller='fuzzy', T=T, theta0_deg=theta0, fps=fps)
        elif choice == '3':
            T = _input_float("仿真时长 T (秒)", 8.0)
            theta0 = _input_float("初始角度 theta0 (度)", 10.0)
            fps = int(_input_float("动画帧率 fps", 60))
            run_animation(controller='pid', T=T, theta0_deg=theta0, fps=fps)
        elif choice == '4':
            T = _input_float("仿真时长 T (秒)", 8.0)
            theta0 = _input_float("初始角度 theta0 (度)", 10.0)
            fps = int(_input_float("动画帧率 fps", 60))
            run_animation(controller='lqr', T=T, theta0_deg=theta0, fps=fps)
        elif choice == '5':
            T = _input_float("仿真时长 T (秒)", 8.0)
            theta0 = _input_float("初始角度 theta0 (度)", 10.0)
            fps = int(_input_float("动画帧率 fps", 60))
            run_animation(controller='mpc', T=T, theta0_deg=theta0, fps=fps)
        elif choice == '6':
            run_compare()
        else:
            print("无效选择，请重试。")


if __name__ == '__main__':
    main()