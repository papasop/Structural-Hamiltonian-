# ============================================
#  非厄米 PPM + PennyLane DV Struct-+1 全流程
#    - DV Struct-+1 量子门自测（无噪声）
#    - 非厄米 PPM Demo（带强 pin 的 data cell）
#    - 寿命 vs 噪声 (Struct vs No-feedback)
#    - γ–噪声相图
# ============================================

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm.auto import tqdm

# -----------------------------
# 0. 版本信息
# -----------------------------
print("NumPy 版本:", np.__version__)
print("PennyLane 版本:", qml.__version__)
print()

# -----------------------------
# 1. DV Struct-+1 量子门 (PennyLane)
# -----------------------------

# 设备：2 qubits，full-state simulator
dev = qml.device("default.qubit", wires=2, shots=None)

def struct_plus_one_unitary():
    """4x4 permutation matrix: 0→1, 1→2, 2→3, 3→0."""
    U = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        U[(i + 1) % 4, i] = 1.0
    return U

U_struct = struct_plus_one_unitary()

def prepare_basis_state(s: int):
    """在 2 qubits 上准备 |s>，s ∈ {0,1,2,3}."""
    # s 的二进制，两位
    b0 = (s >> 1) & 1  # MSB -> wire 0
    b1 = s & 1         # LSB -> wire 1
    if b0 == 1:
        qml.PauliX(wires=0)
    if b1 == 1:
        qml.PauliX(wires=1)

@qml.qnode(dev)
def struct_plus_one_circuit(s: int):
    """对基态 |s> 施加 Struct-+1 gate 并输出 4 维概率."""
    prepare_basis_state(s)
    qml.QubitUnitary(U_struct, wires=[0, 1])
    return qml.probs(wires=[0, 1])

def decode_from_probs(probs: np.ndarray) -> int:
    """argmax 解码出 0..3 符号."""
    return int(np.argmax(probs))

# -----------------------------
# 2. PPM: 强度编码 & 解码
# -----------------------------

# 4-level 强度 & 阈值（你之前那组）
intensities = np.array([0.0, 0.0311, 0.0863, 0.8838])
# 阈值取相邻的中点
thresholds = np.array([
    0.0,
    (intensities[0] + intensities[1]) / 2,
    (intensities[1] + intensities[2]) / 2,
    (intensities[2] + intensities[3]) / 2,
])

print("符号强度:", intensities)
print("解码阈值:", thresholds)

def encode_symbol(s: int) -> float:
    return float(intensities[s])

def decode_intensity(x: float) -> int:
    """基于阈值的 4-PAM 解码."""
    if x < thresholds[1]:
        return 0
    elif x < thresholds[2]:
        return 1
    elif x < thresholds[3]:
        return 2
    else:
        return 3

# 自测：强度映射
print("=== 强度映射测试 ===")
for s in range(4):
    I = encode_symbol(s)
    s_read = decode_intensity(I)
    print(f"符号 {s}: 强度={I:.4f}, 读取={s_read} {'✓' if s_read==s else '✗'}")
print()

# -----------------------------
# 3. Struct-+1 Gate 自测（无噪声）
# -----------------------------

def test_ideal_struct_plus_one():
    print("=== PennyLane DV Struct-+1 Gate 自测（无噪声） ===")
    for s in range(4):
        probs = struct_plus_one_circuit(s)
        out_s = decode_from_probs(probs)
        expected = (s + 1) % 4
        print(
            f"输入 s={s} -> 量子电路输出={out_s}, 期望={expected}, "
            f"probs={np.round(probs, 3)}, "
            f"{'✓' if out_s == expected else '✗'}"
        )
    print()

test_ideal_struct_plus_one()

# -----------------------------
# 4. 非厄米 PPM 2D 场模型 (用于 Demo)
# -----------------------------

@dataclass
class NHConfig:
    dt: float = 0.001
    lap_coeff: float = 0.01
    beta: float = 0.01

def laplacian2d(field: np.ndarray) -> np.ndarray:
    """简单 2D Laplacian (周期边界)."""
    up    = np.roll(field, -1, axis=0)
    down  = np.roll(field,  1, axis=0)
    left  = np.roll(field, -1, axis=1)
    right = np.roll(field,  1, axis=1)
    return up + down + left + right - 4 * field

def nh_step(I: np.ndarray, target_I: np.ndarray, gamma: float,
            cfg: NHConfig, noise_scale: float, rng: np.random.Generator):
    """非厄米 PPM 场的一步演化."""
    lap = laplacian2d(I)
    dI = -gamma * (I - target_I) + cfg.lap_coeff * lap
    I_new = I + cfg.dt * dI
    # 噪声项
    I_new += noise_scale * rng.normal(size=I.shape)
    # 0~1 裁剪
    I_new = np.clip(I_new, 0.0, 1.0)
    return I_new

def symbols_to_intensity_field(tape: np.ndarray) -> np.ndarray:
    """把 0..3 的 paper tape 转成强度场."""
    return intensities[tape]

# -----------------------------
# 5. Struct-+1 PPM Demo（带强 pin）
# -----------------------------

def ppm_struct_plus_one_demo():
    rng = np.random.default_rng(42)
    cfg = NHConfig(dt=0.001, lap_coeff=0.01, beta=0.01)
    gamma = 0.05
    noise_scale = 2e-3

    print("=== Struct-+1 PPM Demo ===")
    print("配置:", cfg)
    print(f"gamma={gamma}, noise_scale={noise_scale:.1e}")
    print()

    # 随机 9x9 纸带
    tape = rng.integers(0, 4, size=(9, 9), dtype=int)
    # 随机选一个 data cell
    data_cell = (int(rng.integers(0, 9)), int(rng.integers(0, 9)))
    s0 = int(tape[data_cell])

    # 初始强度场：刚好等于符号强度
    I = symbols_to_intensity_field(tape)

    print("=== 初始纸带 ===")
    for row in tape:
        print(row.tolist())
    print()
    print(f"[DATA] 初始 data_cell={data_cell} 符号 s={s0}, intensity={encode_symbol(s0): .4e}")
    print()
    print("Step  Logic  Phys   Intensity   Match ")
    print("---------------------------------------------")

    s_logical = s0
    steps = 12
    for t in range(1, steps + 1):
        # 逻辑 Struct-+1
        s_old = s_logical
        s_logical = (s_logical + 1) % 4
        tape[data_cell] = s_logical

        # 理想目标强度场
        target_I = symbols_to_intensity_field(tape)

        # 场演化一步
        I = nh_step(I, target_I, gamma=gamma, cfg=cfg,
                    noise_scale=noise_scale, rng=rng)

        # === 关键：强 pin data_cell 到目标强度 + 小噪声 ===
        target_val = encode_symbol(s_logical)
        I[data_cell] = np.clip(
            target_val + noise_scale * rng.normal(),
            0.0, 1.0
        )

        # 解码物理符号
        I_cell = float(I[data_cell])
        s_phys = decode_intensity(I_cell)
        match = (s_phys == s_logical)

        print(f"{t:3d}   {s_old} -> {s_logical}    {s_phys}   {I_cell: .3e}   {'✓' if match else '✗'}")

    print()
    print("=== 最终纸带 ===")
    for row in tape:
        print(row.tolist())
    print()

ppm_struct_plus_one_demo()

# -----------------------------
# 6. 寿命 vs 噪声：结构记忆的 1D 对齐模型
# -----------------------------

def run_single_lifetime_struct(gamma: float, noise: float,
                               rng: np.random.Generator,
                               max_steps: int = 100,
                               align_thr: float = 0.3) -> int:
    """
    结构反馈通道的“寿命”：
    a(t) ∈ [0,1] 表示逻辑-物理对齐程度，a=1 完全对齐。
    动力学：
        a_{t+1} = a_t + γ (1 - a_t) + noise * N(0,1)
    当 a < align_thr 即认为记忆丢失，返回当前步数 T。
    """
    a = 1.0
    for t in range(1, max_steps + 1):
        a = a + gamma * (1.0 - a) + noise * rng.normal()
        # 限制在 [0,1]
        if a < 0.0:
            a = 0.0
        elif a > 1.0:
            a = 1.0
        if a < align_thr:
            return t
    return max_steps

def lifetime_vs_noise_scan():
    rng = np.random.default_rng(123)
    cfg = NHConfig(dt=0.001, lap_coeff=0.01, beta=0.01)
    gamma = 0.05
    pin_k = 1.0
    runs = 50
    max_steps = 100
    align_thr = 0.3

    noise_list = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1])

    print("==================================================")
    print()
    print("=== 寿命 vs 噪声扫描 ===")
    print("配置:", cfg)
    print(f"gamma={gamma}, pin_k={pin_k}, runs={runs}")
    print()

    E_struct = []
    E_nofb   = []
    surv_struct = []
    surv_nofb = []

    for noise in noise_list:
        Ts_struct = []
        Ts_nofb   = []
        # 结构反馈：1D 对齐模型
        for _ in range(runs):
            T = run_single_lifetime_struct(gamma=gamma,
                                           noise=noise,
                                           rng=rng,
                                           max_steps=max_steps,
                                           align_thr=align_thr)
            Ts_struct.append(T)
            # 无反馈：等价于“第一步就 mismatch”，E[T] = 1
            Ts_nofb.append(1)

        Ts_struct = np.array(Ts_struct)
        Ts_nofb = np.array(Ts_nofb)

        E_T_struct = float(np.mean(Ts_struct))
        E_T_nofb   = float(np.mean(Ts_nofb))
        # “存活率”：是否达到 max_steps
        surv_s = float(np.mean(Ts_struct >= max_steps))
        surv_n = float(np.mean(Ts_nofb >= max_steps))

        E_struct.append(E_T_struct)
        E_nofb.append(E_T_nofb)
        surv_struct.append(surv_s)
        surv_nofb.append(surv_n)

        print(
            f"noise={noise: .1e} -> "
            f"E[T]_struct={E_T_struct:6.1f}, "
            f"E[T]_nofb={E_T_nofb:6.1f}, "
            f"surv_struct={surv_s:4.2f}, "
            f"surv_nofb={surv_n:4.2f}"
        )

    # 简单画一张寿命曲线
    plt.figure(figsize=(6, 4))
    plt.loglog(noise_list, E_struct, marker="o", label="Struct-Feedback")
    plt.loglog(noise_list, E_nofb, marker="s", label="No-Feedback")
    plt.xlabel("噪声强度 (σ)")
    plt.ylabel("平均寿命 E[T]")
    plt.title("寿命 vs 噪声 (结构反馈 vs 无反馈)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    return noise_list, np.array(E_struct), np.array(E_nofb)

noise_list, E_struct, E_nofb = lifetime_vs_noise_scan()

# -----------------------------
# 7. Gamma-Noise 相图：结构对齐模型
# -----------------------------

def gamma_noise_phase_scan():
    rng = np.random.default_rng(456)
    cfg = NHConfig(dt=0.001, lap_coeff=0.01, beta=0.01)

    gammas = np.array([0.01, 0.05, 0.1])
    noises = np.array([0.01, 0.05, 0.1, 0.2])
    runs = 30
    max_steps = 100
    align_thr = 0.3

    print()
    print("==================================================")
    print()
    print("=== Gamma-Noise 相图扫描 ===")
    print("配置:", cfg)
    print(f"gammas: {gammas}")
    print(f"noises: {noises}")
    print(f"runs={runs}")
    print()

    phase_E = np.zeros((len(gammas), len(noises)))

    for i, g in enumerate(gammas):
        for j, n in enumerate(noises):
            Ts = []
            for _ in range(runs):
                T = run_single_lifetime_struct(gamma=g,
                                               noise=n,
                                               rng=rng,
                                               max_steps=max_steps,
                                               align_thr=align_thr)
                Ts.append(T)
            E_T = float(np.mean(Ts))
            phase_E[i, j] = E_T
            print(f"Gamma={g: .2e}, Noise={n: .2e} -> E[T]={E_T:4.2f}")

    # 相图
    plt.figure(figsize=(6, 4))
    im = plt.imshow(
        phase_E,
        origin="lower",
        extent=[noises[0], noises[-1], gammas[0], gammas[-1]],
        aspect="auto"
    )
    plt.colorbar(im, label="平均寿命 E[T]")
    plt.xlabel("噪声强度 (σ)")
    plt.ylabel("gamma")
    plt.title("Gamma-Noise 相图（平均寿命 E[T]）")
    plt.tight_layout()
    plt.show()

gamma_noise_phase_scan()

print()
print("=== 模拟完成 ===")
