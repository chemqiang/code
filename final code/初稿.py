import numpy as np
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"] 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.integrate import solve_ivp
from dataclasses import dataclass
import math
from pathlib import Path
@dataclass

#############################################这是电池容量和老化的参数

@dataclass
class AgingParams:
    Q_eff_C: float    # 【常数】电池有效容量，单位：库仑 C
    a: float          # 【常数】容量老化系数 a
    N: float          # >>> 变量相关 <<< 循环次数 N（可随时间/实验条件变化）
    alpha: float      # 【派生量】容量衰减因子 alpha = 1 - a*sqrt(N)


aging = AgingParams(
    Q_eff_C=4422.0 * 3.6,                 # 4422 mAh -> 15919.2 C
    a=6.3e-3,
    N=0,
    alpha=1.0 - 6.3e-3 * math.sqrt(0)
)

#############################################这是电压和soc的参数

@dataclass
class VoltageParams:
    V_max: float      # 【常数】满电参考电压 V_max [V]
    V_min: float      # 【常数】关机电压 V_min [V]
    SOC_max: float    # 【常数】SOC 上界
    SOC_min: float    # 【常数】SOC 下界（安全余量）

voltage = VoltageParams(
    V_max=4.35,
    V_min=3.30,
    SOC_max=1.0,
    SOC_min=0.05
)


#############################################这是内阻模型的参数

@dataclass
class ResistanceParams:
    R0_Ohm: float     # 【常数】参考温度下的内阻 R0，单位：Ω
    T_ref_C: float    # 【常数】参考温度 T_ref，单位：℃
    eta_per_C: float  # 【常数】低温区内阻变化率 η，单位：1/℃

resistance = ResistanceParams(
    R0_Ohm=30e-3,         # 30 mΩ
    T_ref_C=25.0,
    eta_per_C=0.03
)

#############################################这是内阻温度的参数

@dataclass
class ThermalParams:
    m_g: float            # 【常数】电池等效质量 m，单位：g
    c_J_per_gK: float     # 【常数】等效比热 c，单位：J/(g·K)
    C_a_J_per_K: float    # 【派生量】等效热容 C_a = m*c，单位：J/K
    T_env_C: float        # >>> 变量相关 <<< 环境温度 T_e，单位：℃
    R_th: float           # 【常数】热阻 R_th，单位：K/W（假设常数）  


thermal = ThermalParams(
    m_g=70.0,
    c_J_per_gK=0.9,
    C_a_J_per_K=70.0 * 0.9,   # 63 J/K
    T_env_C=20.0,
    R_th=15.0              # 热阻 R_th，单位：K/W（假设常数）  
)

#############################################这是屏幕 / 计算 / 待机电流参数

@dataclass
class LoadParams:
    I_w_A: float          # 【常数】待机电流 I_w，单位：A
    k_screen_A: float     # >>> 变量相关 <<< 屏幕电流系数 k，单位：A
    k_compute_A: float    # >>> 变量相关 <<< 计算负载系数 k_c，单位：A


load = LoadParams(
    I_w_A=50e-3,          # 50 mA
    k_screen_A=321e-4,    # 32.1 mA
    k_compute_A=0.4
)
#############################################这是网络电流的参数

@dataclass
class NetworkParams:
    I_idle_A: float       # >>> 变量相关 <<< 无数据状态网络电流，单位：A
    I_wifi_A: float       # >>> 变量相关 <<< WiFi 状态网络电流，单位：A
    I_G_A: float          # >>> 变量相关 <<< 蜂窝网络基准电流，单位：A
    gamma: float          # >>> 变量相关 <<< 信号弱增幅系数 γ
    s_min: float          # 【常数】信号强度下限
    s_max: float          # 【常数】信号强度上限


network = NetworkParams(
    I_idle_A=23e-3,
    I_wifi_A=50e-3,
    I_G_A=100e-3,
    gamma=0.6,
    s_min=0.3,
    s_max=1.0
)

########################################################################这是初始条件的参数

@dataclass
class InitialConditions:
    SOC0: float           # [-]初始时刻电池的容量状态 SOC
    T0_C: float           # [°C]初始时刻电池的温度

y0 = InitialConditions(
    SOC0=1.0,
    T0_C=25.0
)

#############################################这是总的参数容器

@dataclass
class ModelParams:
    """
    总模型参数容器（ModelParams）

    用途说明：
    ----------
    - 该容器集中存放电池 SOC–热–功耗耦合模型中的全部参数
    - 用于作为 solve_ivp / 自定义积分器 的统一参数输入
    - 各子模块职责清晰，便于：
        * 单独调参
        * 做灵敏度分析
        * 做参数标定 / 对照实验

    子模块说明：
    ----------
    aging       : 电池容量与老化参数（含循环次数 N）
    voltage     : 电压–SOC 映射关系参数
    resistance  : 内阻–温度解析模型参数
    thermal     : 热模型参数（含环境温度）
    load        : 屏幕 / 计算 / 待机功耗参数
    network     : 网络通信功耗与信号强度模型参数
    """
    aging: AgingParams
    voltage: VoltageParams
    resistance: ResistanceParams
    thermal: ThermalParams
    load: LoadParams
    network: NetworkParams

params = ModelParams(
    aging=aging,
    voltage=voltage,
    resistance=resistance,
    thermal=thermal,
    load=load,
    network=network
)


#############################################这是五个时间序列

def u_of_t(t: float) -> int:
    """
    屏幕开关函数 u(t)

    含义：
    - u = 1：屏幕点亮
    - u = 0：屏幕熄灭

    示例设定：
    - 前 10 分钟（600 s）亮屏
    - 之后灭屏
    """
    if t < 600.0:
        return 1
    else:
        return 0




def b_of_t(t: float) -> float:
    """
    屏幕亮度 b(t)，归一化到 [0, 1]

    示例设定：
    - 屏幕点亮期间，保持中等亮度 0.7
    - 屏幕关闭时，亮度自动为 0（安全处理）
    """
    if u_of_t(t) == 1:
        return 0.7
    else:
        return 0.0




def x_of_t(t: float) -> float:
    """
    计算负载强度 x(t)，归一化到 [0, 1]

    示例设定：
    - 前 5 分钟：重负载（例如视频、游戏）
    - 之后：轻负载（后台计算）
    """
    if t < 300.0:
        return 0.8
    else:
        return 0.2



def s_of_t(t: float) -> float:
    """
    信号强度 s(t)，归一化到 [0.3, 1.0]

    示例设定：
    - 初始信号较好
    - 中段信号较弱
    - 后段恢复
    """
    if t < 400.0:
        return 0.9
    elif t < 800.0:
        return 0.4
    else:
        return 0.8


def mode_of_t(t: float) -> str:
    """
    网络模式 mode(t)
    返回:
    - "idle" : 无数据
    - "wifi" : WiFi
    - "cell" : 蜂窝(4G/5G)
    
    示例设定：
    - 0~200s：WiFi
    - 200s 之后：蜂窝
    """
    if t < 200.0:
        return "wifi"
    else:
        return "cell"


#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################下面开始解方程

# ============================================================
# 0) 工具函数：范围裁剪（保证输入变量落在合法范围内）
# ============================================================

def clamp(x, xmin, xmax):
    """把 x 限制在 [xmin, xmax]，防止数值/输入越界导致求解不稳定"""
    return max(xmin, min(float(x), xmax))

# ===========================================================
# 1) 物理模型函数：开路电压 Voc(SOC)
#    公式：
#    Voc(SOC) = Vmin + (Vmax - Vmin)/(1-SOCmin) * (SOC - SOCmin)
# ============================================================

def voc_of_soc(SOC: float, params) -> float:
    """开路电压 Voc，单位 [V]"""
    SOC = clamp(SOC, params.voltage.SOC_min, params.voltage.SOC_max)
    Vmin = params.voltage.V_min
    Vmax = params.voltage.V_max
    SOCmin = params.voltage.SOC_min
    return Vmin + (Vmax - Vmin) / (1.0 - SOCmin) * (SOC - SOCmin)

# ============================================================
# 3) 物理模型函数：内阻 R(T)（解析形式）
#    公式：
#      R(T)=R0*exp(-eta*(T-Tref))  (T < Tref)
#      R(T)=R0                     (T >= Tref)
# ============================================================

def r_of_T(T_C: float, params) -> float:
    """内阻 R(T)，单位 [Ohm]"""
    R0 = params.resistance.R0_Ohm
    Tref = params.resistance.T_ref_C
    eta = params.resistance.eta_per_C
    if T_C < Tref:
        return R0 * math.exp(-eta * (T_C - Tref))
    else:
        return R0
    
# ============================================================
# 4) 电流分量：屏幕电流 Is(t)
#    公式：Is = u(t) * k * b(t)
# ============================================================

def I_screen(t: float, params) -> float:
    u = int(u_of_t(t))                      # {0,1}
    b = clamp(b_of_t(t), 0.0, 1.0)          # [0,1]
    return u * params.load.k_screen_A * b   # [A]

# ============================================================
# 5) 电流分量：计算负载电流 Ic(t)
#    公式：Ic = u(t) * k_c * x(t)
# ============================================================

def I_compute(t: float, params) -> float:
    u = int(u_of_t(t))                      # {0,1}
    x = clamp(x_of_t(t), 0.0, 1.0)          # [0,1]
    return u * params.load.k_compute_A * x  # [A]

# ============================================================
# 6) 电流分量：网络电流 Inet(t)
#    公式：
#      idle: I_idle
#      wifi: I_wifi
#      cell: I_G*(1 + gamma*(1/s - 1))
#    且 s ∈ [0.3, 1.0]
# ============================================================

def I_net(t: float, params) -> float:
    mode = mode_of_t(t)
    u = int(u_of_t(t))  # 0-1 开关
    if mode == "idle":
        return u * params.network.I_idle_A
    elif mode == "wifi":
        return u * params.network.I_wifi_A
    elif mode == "cell":
        s = clamp(s_of_t(t), params.network.s_min, params.network.s_max)
        return u * params.network.I_G_A * (1.0 + params.network.gamma * (1.0 / s - 1.0))
    else:
        raise ValueError(f"未知的网络模式 mode(t)={mode}")
    
# ============================================================
# 7) 总电流 I(t)
#    公式：I = Is + Ic + Inet + Iw
# ============================================================

def I_total(t: float, params) -> float:
    return I_screen(t, params) + I_compute(t, params) + I_net(t, params) + params.load.I_w_A

# ============================================================
# 8) ODE 右端函数 rhs(t, y)
#    状态 y = [SOC, T]
#
# 公式总览：
#  (1) alpha = 1 - a*sqrt(N)  (已在 params.aging.alpha 中计算好)
#  (2) V = Voc(SOC) - I*R(T)
#  (3) phi = Vmax / V
#  (4) dSOC/dt = -phi * I / (alpha * Qeff)
#  (5) dT/dt   = (Voc*I)/Ca  - (T-Te)/(R(T)*Ca)
# ============================================================

def rhs(t: float, y: np.ndarray, params):
    SOC, T_C = float(y[0]), float(y[1])

    # --- 计算电流 I(t) ---
    I = I_total(t, params)  # [A]

    # --- 计算 Voc, R(T), V(t) ---
    Voc = voc_of_soc(SOC, params)          # [V]
    R = r_of_T(T_C, params)                # [Ohm]
    V = Voc - I * R                        # [V]

    # --- 修正系数 phi(SOC) = Vmax / V ---
    # 为避免 V 接近 0 导致发散，给一个很小的下限（数值稳定性）
    V_safe = max(V, 1e-6)
    phi = params.voltage.V_max / V_safe

    # --- dSOC/dt ---
    alpha = params.aging.alpha
    Qeff = params.aging.Q_eff_C
    dSOC_dt = -phi * I / (alpha * Qeff)

    # --- dT/dt ---
    Ca = params.thermal.C_a_J_per_K
    Te = params.thermal.T_env_C
    R_th = params.thermal.R_th
    dT_dt = (I * I * R) / Ca - (T_C - Te) / (R_th * Ca)

    # （可选）防止 SOC 数值积分超范围：当到边界时让导数为 0
    # 这样不会改变方程结构，只是数值保护
    if SOC <= params.voltage.SOC_min and dSOC_dt < 0:
        dSOC_dt = 0.0
    if SOC >= params.voltage.SOC_max and dSOC_dt > 0:
        dSOC_dt = 0.0

    return [dSOC_dt, dT_dt]

# ============================================================
# 9) 用 solve_ivp 求解（直到 SOC 恰好到达 SOC_min 停止）
#    关键：使用事件函数 event_shutdown，使积分在 SOC=SOC_min 时终止
# ============================================================

def solve_until_shutdown(params, y0, t_max_guess_s=24*3600, dt_eval_s=1.0, max_extend=6, extend_factor=2.0):
    """
    求解 ODE，直到 SOC 降到 SOC_min 为止，模拟“手机关机”。

    参数说明：
    ----------
    params : ModelParams
        总参数容器
    y0 : InitialConditions
        初始条件（SOC0, T0_C）
    t_max_guess_s : float
        最大尝试求解时间（秒），事件触发前到达该时间会停止（一般给足够大，例如 24h）
    dt_eval_s : float
        采样间隔（秒），用于生成输出时间序列 sol.t / sol.y
    max_extend : int
        若未触发事件，最多自动延长的次数
    extend_factor : float
        每次延长时间轴的倍率（例如 2.0 表示时间上限翻倍）

    返回：
    ------
    sol : OdeResult
        solve_ivp 返回对象（已在 SOC=SOC_min 处终止）
    """

    # 初始状态：y = [SOC, T]
    y_init = np.array([y0.SOC0, y0.T0_C], dtype=float)

    # 事件函数：当 SOC - SOC_min = 0 时触发（时间轴终点）
    def event_shutdown(t, y):
        # y[0] = SOC
        return y[0] - params.voltage.SOC_min

    # 事件属性：触发即终止；只检测下降穿越（direction=-1）
    event_shutdown.terminal = True
    event_shutdown.direction = -1

    # 若在给定 t_max 内未触达 SOC_min，则自动延长时间轴直到触发事件
    t0 = 0.0
    y_curr = y_init
    t_end = float(t_max_guess_s)
    all_t = None
    all_y = None

    for _ in range(max_extend + 1):
        # 生成输出采样点（首段包含 t0；后续段从 t0+dt_eval 开始避免重复）
        if t0 == 0.0:
            t_eval = np.arange(t0, t_end + dt_eval_s, dt_eval_s)
        else:
            t_eval = np.arange(t0 + dt_eval_s, t_end + dt_eval_s, dt_eval_s)

        sol = solve_ivp(
            fun=lambda t, y: rhs(t, y, params),
            t_span=(t0, t_end),
            y0=y_curr,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            events=event_shutdown
        )

        if not sol.success:
            raise RuntimeError(f"ODE 求解失败：{sol.message}")

        # 累加拼接时间序列（避免段首重复）
        if all_t is None:
            all_t = sol.t
            all_y = sol.y
        else:
            all_t = np.concatenate([all_t, sol.t])
            all_y = np.concatenate([all_y, sol.y], axis=1)

        # 若事件触发，sol.t / sol.y 已自动截断到事件时刻附近；
        # 事件精确时刻在 sol.t_events[0][0]
        if sol.t_events and len(sol.t_events[0]) > 0:
            sol.t = all_t
            sol.y = all_y
            return sol

        # 未触达 SOC_min，延长时间轴继续积分
        t0 = sol.t[-1]
        y_curr = sol.y[:, -1]
        t_end *= extend_factor

    raise RuntimeError("在最大延长次数内仍未到达 SOC_min，请增大 t_max_guess_s 或 max_extend。")


# ======= 调用：直到关机为止 =======
sol = solve_until_shutdown(params, y0, t_max_guess_s=24*3600, dt_eval_s=1.0)


# ============================================================
# 10) 结果整理与可视化（所有可参考量进入 df；x 轴单位改为“小时”）
# ============================================================

# ---------- 时间轴 ----------
t_s = sol.t                               # 时间 [s]
t_h = t_s / 3600.0                        # 时间 [h]（用于画图 x 轴）
SOC = sol.y[0]                            # SOC
T_C = sol.y[1]                            # 温度 [°C]

# ---------- 把“所有可用于分析的量”都算出来放进 df ----------
# 中文注释写清每个量的物理含义与单位

# 输入/工况（时间序列）
u_arr = np.array([int(u_of_t(tt)) for tt in t_s])                               # 屏幕开关 u(t) ∈ {0,1}
b_arr = np.array([clamp(b_of_t(tt), 0.0, 1.0) for tt in t_s])                   # 亮度 b(t) ∈ [0,1]
x_arr = np.array([clamp(x_of_t(tt), 0.0, 1.0) for tt in t_s])                   # 负载强度 x(t) ∈ [0,1]
s_arr = np.array([clamp(s_of_t(tt), params.network.s_min, params.network.s_max) for tt in t_s])  # 信号强度 s(t) ∈ [0.3,1]
mode_arr = np.array([mode_of_t(tt) for tt in t_s])                              # 网络模式 mode(t)

# 电流分量（A）
I_s_arr = np.array([I_screen(tt, params) for tt in t_s])                        # 屏幕电流 Is [A]
I_c_arr = np.array([I_compute(tt, params) for tt in t_s])                       # 计算电流 Ic [A]
I_net_arr = np.array([I_net(tt, params) for tt in t_s])                         # 网络电流 Inet [A]
I_w_arr = np.full_like(t_s, params.load.I_w_A, dtype=float)                     # 待机电流 Iw [A]（常数）
I_arr = I_s_arr + I_c_arr + I_net_arr + I_w_arr                                 # 总电流 I [A]

# 电压/内阻相关（V, Ohm）
Voc_arr = np.array([voc_of_soc(ss, params) for ss in SOC])                      # 开路电压 Voc(SOC) [V]
R_arr = np.array([r_of_T(TT, params) for TT in T_C])                            # 内阻 R(T) [Ohm]
V_arr = Voc_arr - I_arr * R_arr                                                 # 端电压 V = Voc - I*R [V]

# 修正系数与导数（用于检查/诊断）
V_safe_arr = np.maximum(V_arr, 1e-6)                                            # 防止除零（数值保护）
phi_arr = params.voltage.V_max / V_safe_arr                                     # 修正系数 phi = Vmax / V [-]

alpha = params.aging.alpha                                                      # 老化因子 alpha（常数）
Qeff = params.aging.Q_eff_C                                                     # 有效容量 Qeff [C]
dSOC_dt_arr = -phi_arr * I_arr / (alpha * Qeff)                                 # dSOC/dt [1/s]

Ca = params.thermal.C_a_J_per_K                                                 # 等效热容 Ca [J/K]
Te = params.thermal.T_env_C                                                     # 环境温度 Te [°C]
R_th = params.thermal.R_th                                                      # 热阻 R_th [K/W]
dT_dt_arr = (I_arr * I_arr * R_arr) / Ca - (T_C - Te) / (R_th * Ca)            # dT/dt [°C/s]

# ---------- 汇总到 DataFrame（用于分析和画图） ----------
df = pd.DataFrame({
    # 时间
    "t_s": t_s,                 # 时间 [s]
    "t_h": t_h,                 # 时间 [h] —— 画图推荐用这个

    # 状态
    "SOC": SOC,                 # SOC [-]
    "T_C": T_C,                 # 温度 [°C]

    # 输入/工况
    "u": u_arr,                 # 屏幕开关 {0,1}
    "b": b_arr,                 # 亮度 [0,1]
    "x": x_arr,                 # 计算强度 [0,1]
    "s": s_arr,                 # 信号强度 [0.3,1]
    "mode": mode_arr,           # 网络模式

    # 电流（A）
    "I_s_A": I_s_arr,           # 屏幕电流 [A]
    "I_c_A": I_c_arr,           # 计算电流 [A]
    "I_net_A": I_net_arr,       # 网络电流 [A]
    "I_w_A": I_w_arr,           # 待机电流 [A]
    "I_A": I_arr,               # 总电流 [A]

    # 电压/内阻（V, Ohm）
    "Voc_V": Voc_arr,           # 开路电压 [V]
    "R_Ohm": R_arr,             # 内阻 [Ohm]
    "V_V": V_arr,               # 端电压 [V]

    # 修正与导数（诊断/分析用）
    "phi": phi_arr,             # 修正系数 phi [-]
    "dSOC_dt_1_per_s": dSOC_dt_arr,  # dSOC/dt [1/s]
    "dT_dt_C_per_s": dT_dt_arr,      # dT/dt [°C/s]

    # 常量（放进 df 便于导出/复现实验）
    "alpha": np.full_like(t_s, alpha, dtype=float),      # 老化因子 alpha
    "Q_eff_C": np.full_like(t_s, Qeff, dtype=float),     # 有效容量 Qeff [C]
    "T_env_C": np.full_like(t_s, Te, dtype=float),       # 环境温度 Te [°C]
})

print(df.head())
print("\n关机时刻（小时）≈", df["t_h"].iloc[-1], "h")
print("关机时 SOC ≈", df["SOC"].iloc[-1])
out_path = Path(__file__).resolve().parent / "battery_results.xlsx"
df.to_excel(out_path, index=False)







#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################画图

t_h = df["t_h"]
phi_I = df["phi"] * df["I_A"]

fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)

axes[0, 0].plot(t_h, df["V_V"])
axes[0, 0].set_title("电压 V")
axes[0, 0].set_ylabel("V [V]")
axes[0, 0].grid(True)

axes[0, 1].plot(t_h, df["R_Ohm"])
axes[0, 1].set_title("内阻 R")
axes[0, 1].set_ylabel("R [Ohm]")
axes[0, 1].grid(True)

axes[1, 0].plot(t_h, df["T_C"])
axes[1, 0].set_title("温度 T")
axes[1, 0].set_ylabel("T [°C]")
axes[1, 0].grid(True)

axes[1, 1].plot(t_h, df["SOC"])
axes[1, 1].set_title("SOC")
axes[1, 1].set_ylabel("SOC [-]")
axes[1, 1].grid(True)

axes[2, 0].plot(t_h, df["I_A"])
axes[2, 0].set_title("总电流 I")
axes[2, 0].set_ylabel("I [A]")
axes[2, 0].set_xlabel("时间 t [h]")
axes[2, 0].grid(True)

axes[2, 1].plot(t_h, phi_I)
axes[2, 1].set_title("phi × I")
axes[2, 1].set_ylabel("phi·I [-·A]")
axes[2, 1].set_xlabel("时间 t [h]")
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()








