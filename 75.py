# -*- coding: utf-8 -*-
# 完整版：非厄米 PPM + Struct-+1 + 寿命/相图 (Colab适配)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

print("=== 非厄米 PPM 完整模拟系统 ===")
print("NumPy版本:", np.__version__)

# -------------------------------
# 0. 全局参数：网格 & 符号强度
# -------------------------------
N = 9  # 9x9 纸带/介质
I_SYMBOL = np.array([0.0, 0.0311, 0.0863, 0.8838], dtype=np.float64)
THRESHOLDS = np.array([0.0, 
                      (I_SYMBOL[0] + I_SYMBOL[1])/2, 
                      (I_SYMBOL[1] + I_SYMBOL[2])/2, 
                      (I_SYMBOL[2] + I_SYMBOL[3])/2], dtype=np.float64)

DATA_CELL = (1, 0)  # (row, col)

print(f"符号强度: {I_SYMBOL}")
print(f"解码阈值: {THRESHOLDS}")

# -------------------------------
# 1. 物理核：拉普拉斯 + 稳定非厄米步进
# -------------------------------
class NHConfig:
    """
    非厄米介质配置：
    - dt: 时间步长
    - lap_coeff: 拉普拉斯系数
    - beta: 非线性软饱和耗散强度
    """
    def __init__(self, dt=1e-3, lap_coeff=1e-2, beta=1e-2):
        self.dt = dt
        self.lap_coeff = lap_coeff
        self.beta = beta

    def __str__(self):
        return f"NHConfig(dt={self.dt}, lap_coeff={self.lap_coeff}, beta={self.beta})"


def laplacian_periodic(field: np.ndarray) -> np.ndarray:
    """二维周期边界拉普拉斯"""
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )


def nh_step(psi: np.ndarray,
            gamma: float,
            noise_scale: float,
            cfg: NHConfig) -> np.ndarray:
    """
    单步非厄米演化（稳定版）：
      dψ/dt = i * lap_coeff * ∇²ψ - γ ψ - β |ψ|² ψ + 噪声
    """
    lap = laplacian_periodic(psi)

    # 线性：非厄米传播 + 线性耗散
    dpsi_dt = 1j * cfg.lap_coeff * lap - gamma * psi

    # 非线性软饱和：幅度越大，额外耗散越强，防止爆炸
    dpsi_dt -= cfg.beta * (np.abs(psi) ** 2) * psi

    # 高斯噪声（复数）
    noise = noise_scale * (
        np.random.randn(*psi.shape) + 1j * np.random.randn(*psi.shape)
    )

    psi_new = psi + cfg.dt * dpsi_dt + np.sqrt(cfg.dt) * noise
    
    # 稳定性监控
    max_amp = np.max(np.abs(psi_new))
    if max_amp > 10.0:
        print(f"警告：场振幅过大 {max_amp:.2e}，进行裁剪")
        psi_new = np.clip(psi_new, -10.0, 10.0)
        
    return psi_new


def intensity(psi: np.ndarray) -> np.ndarray:
    """计算场强度 |ψ|²"""
    return np.real(psi * np.conj(psi))


# ----------------------------------------
# 2. 符号编码/解码：四级强度码 + 自校准测试
# ----------------------------------------
def symbol_from_intensity(I: float) -> int:
    """从强度值解码符号"""
    if I < THRESHOLDS[1]:
        return 0
    elif I < THRESHOLDS[2]:
        return 1
    elif I < THRESHOLDS[3]:
        return 2
    else:
        return 3


vec_symbol_from_intensity = np.vectorize(symbol_from_intensity)


def test_intensity_mapping():
    """测试强度到符号的映射是否正确"""
    print("\n=== 强度映射测试 ===")
    all_correct = True
    for s in range(4):
        I = I_SYMBOL[s]
        read = symbol_from_intensity(I)
        ok = read == s
        all_correct &= ok
        status = "✓" if ok else "✗"
        print(f"符号 {s}: 强度={I:.4f}, 读取={read} {status}")
    
    if all_correct:
        print("✅ 所有符号映射正确！")
    else:
        print("❌ 存在映射错误！")
    return all_correct


# ----------------------------------------
# 3. 纸带初始化 & Struct-+1 Gate
# ----------------------------------------
def random_tape(n: int = N) -> np.ndarray:
    """生成随机 0..3 纸带"""
    return np.random.randint(0, 4, size=(n, n), dtype=np.int64)


def tape_to_psi(tape: np.ndarray) -> np.ndarray:
    """用符号强度把纸带编码成复场 ψ（初始相位取 0）"""
    amp = np.sqrt(I_SYMBOL[tape])
    return amp.astype(np.complex128)


def psi_to_tape(psi: np.ndarray) -> np.ndarray:
    """从 ψ 读回符号"""
    I = intensity(psi)
    return vec_symbol_from_intensity(I)


def apply_struct_plus_one_gate(psi: np.ndarray,
                               logical_sym: int,
                               pin_k: float = 1.0) -> (np.ndarray, int):
    """
    Struct-+1 Gate:
      logical_sym: n -> (n+1) mod 4
      同时把 data_cell 写成对应强度
    """
    new_logical = (logical_sym + 1) % 4
    target_I = I_SYMBOL[new_logical] * pin_k
    target_amp = np.sqrt(target_I)
    psi = psi.copy()
    r, c = DATA_CELL
    psi[r, c] = target_amp + 0j
    return psi, new_logical


# ----------------------------------------
# 4. 单次 Struct-+1 PPM 测试（结构锁定演示）
# ----------------------------------------
def demo_struct_ppm(cfg: NHConfig,
                    gamma: float = 0.05,
                    noise_scale: float = 2e-3,
                    steps_per_logic: int = 50,
                    logic_steps: int = 8,
                    pin_k: float = 1.0):
    """
    演示：在稳定非厄米介质上跑若干次 Struct-+1 逻辑循环，
    观察逻辑/物理符号历史的一致性。
    """
    print(f"\n=== Struct-+1 PPM Demo ===")
    print(f"配置: {cfg}")
    print(f"gamma={gamma}, noise_scale={noise_scale:.1e}")

    # 初始化随机纸带 & ψ
    np.random.seed(42)  # 可重复性
    tape0 = random_tape(N)
    print("\n=== 初始纸带 ===")
    for row in tape0.tolist():
        print(row)

    psi = tape_to_psi(tape0)

    # 初始 data_cell 逻辑符号
    r0, c0 = DATA_CELL
    I0 = intensity(psi)[r0, c0]
    logical_sym = symbol_from_intensity(I0)
    print(f"\n[DATA] 初始 data_cell={DATA_CELL} 符号 s={logical_sym}, intensity={I0:.4e}")

    logic_hist = [logical_sym]
    phys_hist = [logical_sym]
    intensity_hist = [I0]

    print(f"\n{'Step':^4} {'Logic':^6} {'Phys':^5} {'Intensity':^12} {'Match':^6}")
    print("-" * 45)

    for k in range(1, logic_steps + 1):
        # 应用 Struct-+1 Gate（写逻辑 + 写介质）
        psi, logical_sym = apply_struct_plus_one_gate(psi, logical_sym, pin_k=pin_k)

        # 在介质中演化 steps_per_logic 步
        for _ in range(steps_per_logic):
            psi = nh_step(psi, gamma=gamma, noise_scale=noise_scale, cfg=cfg)

        I = intensity(psi)
        s_phys = symbol_from_intensity(I[r0, c0])
        match = "✓" if logical_sym == s_phys else "✗"
        
        logic_hist.append(logical_sym)
        phys_hist.append(s_phys)
        intensity_hist.append(I[r0, c0])

        print(f"{k:4d} {logic_hist[-2]:3} -> {logical_sym:1} {s_phys:5} {I[r0,c0]:11.3e} {match:^6}")

    # 最终纸带
    tape_final = psi_to_tape(psi)
    print(f"\n=== 最终纸带 ===")
    for row in tape_final.tolist():
        print(row)

    # 统计逻辑/物理匹配率（跳过初始态）
    matches = sum(int(l == p) for l, p in zip(logic_hist[1:], phys_hist[1:]))
    total = len(logic_hist) - 1
    acc = matches / total if total > 0 else 0.0
    
    print(f"\n=== 性能统计 ===")
    print(f"逻辑与物理状态匹配: {matches}/{total} ({acc*100:.1f}%)")
    
    # 可视化演化历史
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(logic_hist, 'o-', label='逻辑状态', linewidth=2, markersize=8)
    plt.plot(phys_hist, 's-', label='物理状态', linewidth=2, markersize=6)
    plt.xlabel('逻辑步')
    plt.ylabel('符号状态')
    plt.title('Struct-+1 PPM 状态演化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(intensity_hist, 'o-', color='red', linewidth=2, markersize=6)
    plt.xlabel('逻辑步')
    plt.ylabel('强度 (log scale)')
    plt.title('Data Cell 强度演化')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return acc, logic_hist, phys_hist


# ----------------------------------------
# 5. 寿命扫描：结构反馈 vs 无反馈
# ----------------------------------------
def simulate_lifetime_single_run_struct(cfg: NHConfig,
                                        gamma: float,
                                        noise_scale: float,
                                        max_logic_steps: int = 200,
                                        steps_per_logic: int = 50,
                                        pin_k: float = 1.0) -> int:
    """
    单次寿命模拟（有结构反馈）：
      返回第一次逻辑/物理不一致的逻辑步编号（1..max），
      如始终一致则返回 max_logic_steps+1。
    """
    tape0 = random_tape(N)
    psi = tape_to_psi(tape0)
    r0, c0 = DATA_CELL
    I0 = intensity(psi)[r0, c0]
    logical_sym = symbol_from_intensity(I0)

    for step_idx in range(1, max_logic_steps + 1):
        # 写入逻辑 + 写入介质
        psi, logical_sym = apply_struct_plus_one_gate(psi, logical_sym, pin_k=pin_k)

        # 演化
        for _ in range(steps_per_logic):
            psi = nh_step(psi, gamma=gamma, noise_scale=noise_scale, cfg=cfg)

        I = intensity(psi)
        s_phys = symbol_from_intensity(I[r0, c0])

        if s_phys != logical_sym:
            return step_idx

    return max_logic_steps + 1  # 视作未失败


def simulate_lifetime_single_run_nofb(cfg: NHConfig,
                                      gamma: float,
                                      noise_scale: float,
                                      max_logic_steps: int = 200,
                                      steps_per_logic: int = 50) -> int:
    """
    单次寿命模拟（无结构反馈对照）：
      逻辑层仍做 +1 mod 4，但不向介质写入 Struct-+1。
      物理只受噪声 + 非厄米漂移。
    """
    tape0 = random_tape(N)
    psi = tape_to_psi(tape0)
    r0, c0 = DATA_CELL
    I0 = intensity(psi)[r0, c0]
    logical_sym = symbol_from_intensity(I0)

    for step_idx in range(1, max_logic_steps + 1):
        # 仅逻辑 +1，不写入 psi
        logical_sym = (logical_sym + 1) % 4

        # 演化
        for _ in range(steps_per_logic):
            psi = nh_step(psi, gamma=gamma, noise_scale=noise_scale, cfg=cfg)

        I = intensity(psi)
        s_phys = symbol_from_intensity(I[r0, c0])

        if s_phys != logical_sym:
            return step_idx

    return max_logic_steps + 1


def scan_lifetime_vs_noise(cfg: NHConfig,
                           gamma: float = 0.05,
                           pin_k: float = 1.0,
                           max_logic_steps: int = 200,
                           steps_per_logic: int = 50,
                           runs: int = 100):
    """扫描寿命随噪声的变化"""
    print(f"\n=== 寿命 vs 噪声扫描 ===")
    print(f"配置: {cfg}")
    print(f"gamma={gamma}, pin_k={pin_k}, runs={runs}")

    noise_list = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1]
    et_struct = []
    et_nofb = []
    surv_struct = []
    surv_nofb = []

    for noise in tqdm(noise_list, desc="噪声扫描"):
        T_struct = []
        T_nofb = []
        
        for _ in tqdm(range(runs), desc=f"Noise={noise:.1e}", leave=False):
            T_struct.append(
                simulate_lifetime_single_run_struct(
                    cfg, gamma, noise, max_logic_steps, steps_per_logic, pin_k=pin_k
                )
            )
            T_nofb.append(
                simulate_lifetime_single_run_nofb(
                    cfg, gamma, noise, max_logic_steps, steps_per_logic
                )
            )
            
        T_struct = np.array(T_struct, dtype=float)
        T_nofb = np.array(T_nofb, dtype=float)

        E_struct = T_struct.mean()
        E_nofb = T_nofb.mean()
        surv_frac_struct = np.mean(T_struct > max_logic_steps)
        surv_frac_nofb = np.mean(T_nofb > max_logic_steps)

        et_struct.append(E_struct)
        et_nofb.append(E_nofb)
        surv_struct.append(surv_frac_struct)
        surv_nofb.append(surv_frac_nofb)

        print(f"noise={noise:7.1e} -> "
              f"E[T]_struct={E_struct:6.1f}, E[T]_nofb={E_nofb:6.1f}, "
              f"surv_struct={surv_frac_struct:.2f}, surv_nofb={surv_frac_nofb:.2f}")

    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(noise_list, et_struct, 'o-', label='Struct-+1 (有反馈)', linewidth=2, markersize=6)
    plt.semilogx(noise_list, et_nofb, 's-', label='无反馈', linewidth=2, markersize=6)
    plt.xlabel('噪声强度')
    plt.ylabel('平均寿命 (逻辑步)')
    plt.title('寿命 vs 噪声')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogx(noise_list, surv_struct, 'o-', label='Struct-+1 (有反馈)', linewidth=2, markersize=6)
    plt.semilogx(noise_list, surv_nofb, 's-', label='无反馈', linewidth=2, markersize=6)
    plt.xlabel('噪声强度')
    plt.ylabel('存活概率')
    plt.title('存活概率 vs 噪声')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    return noise_list, et_struct, et_nofb, surv_struct, surv_nofb


# ----------------------------------------
# 6. gamma-noise 相图：结构锁定 vs 失稳
# ----------------------------------------
def scan_phase_diagram(cfg: NHConfig,
                       gammas=None,
                       noises=None,
                       max_logic_steps: int = 200,
                       steps_per_logic: int = 50,
                       runs: int = 50,
                       pin_k: float = 1.0):
    """扫描 gamma-noise 相图"""
    if gammas is None:
        gammas = [1e-2, 2e-2, 5e-2, 1e-1, 2e-1]
    if noises is None:
        noises = [1e-2, 2e-2, 5e-2, 1e-1, 2e-1]

    print(f"\n=== Gamma-Noise 相图扫描 ===")
    print(f"配置: {cfg}")
    print(f"gammas: {gammas}")
    print(f"noises: {noises}")
    print(f"runs={runs}")

    E_mat = np.zeros((len(gammas), len(noises)))
    S_mat = np.zeros_like(E_mat)

    for i, g in enumerate(tqdm(gammas, desc="Gamma扫描")):
        for j, n in enumerate(tqdm(noises, desc=f"Gamma={g:.1e}", leave=False)):
            T_list = []
            for _ in range(runs):
                T_list.append(
                    simulate_lifetime_single_run_struct(
                        cfg, gamma=g, noise_scale=n,
                        max_logic_steps=max_logic_steps,
                        steps_per_logic=steps_per_logic,
                        pin_k=pin_k,
                    )
                )
            T_arr = np.array(T_list, dtype=float)
            E = T_arr.mean()
            surv = np.mean(T_arr > max_logic_steps)
            E_mat[i, j] = E
            S_mat[i, j] = surv

    # 可视化相图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 平均寿命热图
    im1 = ax1.imshow(E_mat, cmap='viridis', origin='lower', 
                    extent=[min(noises), max(noises), min(gammas), max(gammas)], 
                    aspect='auto')
    ax1.set_xlabel('噪声强度')
    ax1.set_ylabel('耗散强度 γ')
    ax1.set_title('平均寿命热图')
    plt.colorbar(im1, ax=ax1, label='平均寿命')
    
    # 添加数值标注
    for i in range(len(gammas)):
        for j in range(len(noises)):
            ax1.text(noises[j], gammas[i], f'{E_mat[i,j]:.0f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    # 存活概率热图
    im2 = ax2.imshow(S_mat, cmap='plasma', origin='lower',
                    extent=[min(noises), max(noises), min(gammas), max(gammas)],
                    aspect='auto', vmin=0, vmax=1)
    ax2.set_xlabel('噪声强度')
    ax2.set_ylabel('耗散强度 γ')
    ax2.set_title('存活概率热图')
    plt.colorbar(im2, ax=ax2, label='存活概率')
    
    # 添加数值标注
    for i in range(len(gammas)):
        for j in range(len(noises)):
            ax2.text(noises[j], gammas[i], f'{S_mat[i,j]:.2f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

    return gammas, noises, E_mat, S_mat


# ----------------------------------------
# 7. 主执行函数
# ----------------------------------------
def main():
    """主执行函数"""
    print("开始非厄米 PPM 完整模拟...")
    start_time = time.time()
    
    # 配置参数
    cfg = NHConfig(dt=1e-3, lap_coeff=1e-2, beta=1e-2)
    
    # 1. 测试强度映射
    mapping_ok = test_intensity_mapping()
    if not mapping_ok:
        print("❌ 强度映射测试失败，停止执行")
        return
    
    # 2. 单次 PPM 演示
    print("\n" + "="*50)
    acc, logic_hist, phys_hist = demo_struct_ppm(
        cfg, gamma=0.05, noise_scale=2e-3, 
        steps_per_logic=50, logic_steps=12, pin_k=1.0
    )
    
    # 3. 寿命扫描（简化版，减少运行次数以节省时间）
    print("\n" + "="*50)
    noise_list, et_struct, et_nofb, surv_struct, surv_nofb = scan_lifetime_vs_noise(
        cfg, gamma=0.05, pin_k=1.0,
        max_logic_steps=100, steps_per_logic=50, runs=50  # 减少运行次数
    )
    
    # 4. 相图扫描（简化版）
    print("\n" + "="*50)
    gammas, noises, E_mat, S_mat = scan_phase_diagram(
        cfg,
        gammas=[1e-2, 5e-2, 1e-1],  # 简化参数
        noises=[1e-2, 5e-2, 1e-1, 2e-1],
        max_logic_steps=100, steps_per_logic=50, runs=30  # 减少运行次数
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n=== 模拟完成 ===")
    print(f"总运行时间: {elapsed_time:.1f} 秒")
    print(f"最终 PPM 准确率: {acc*100:.1f}%")

# 执行主函数
if __name__ == "__main__":
    main()
