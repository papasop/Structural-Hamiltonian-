import numpy as np
import matplotlib.pyplot as plt

# 空间参数
L = 10
N = 20
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# 时间参数
dt = 0.005
steps = 800

# 网格初始化（1D简化，方便调试，可升级至3D）
X = x

# 初始波包
def gaussian_1d(x0, sigma=0.5, amp=5.0):
    return amp * np.exp(-0.5 * ((X - x0)/sigma)**2)

# 动态外势（时空扰动）
def phi_res_1d(t, omega=2*np.pi/20, amp=3.0):
    return amp * np.cos(omega * t) * np.exp(-0.5 * ((X - 5)**2))

# 平滑异常点耗散激活函数（sigmoid）
def smooth_gamma(psi, threshold=1.0, slope=5.0, base=6.0):
    grad = np.gradient(np.abs(psi), dx)
    grad_mag = np.abs(grad)
    gamma = base / (1 + np.exp(-slope * (grad_mag - threshold)))
    return gamma

# 非线性反馈
def nonlinear_feedback(psi, coeff=0.3):
    return coeff * np.abs(psi)**2 * psi

# 图灵机参数和状态
tape_length = 15
cell_size = N // tape_length
head_pos = 5  # 初始头位置
state = 0     # 初始状态

# 初始带状态波函数叠加
psi = np.zeros(N, dtype=np.complex128)
for i in range(tape_length):
    # 假设随机初始带状态 0 或 1，用不同位置高斯波包表示1
    if np.random.rand() > 0.5:
        psi += gaussian_1d((i + 0.5) * cell_size, sigma=cell_size/4, amp=5.0)

# 跳转规则定义：(state, symbol_read) -> (new_state, symbol_write, move)
transition = {
    (0, 0): (1, 1, 1),
    (1, 0): (0, 1, -1),
    (0, 1): (0, 0, 1),
    (1, 1): (1, 0, -1),
}

def read_symbol(psi, head_pos):
    idx_start = head_pos * cell_size
    idx_end = idx_start + cell_size
    intensity = np.sum(np.abs(psi[idx_start:idx_end])**2)
    return 1 if intensity > 0.5 else 0

def write_symbol(psi, head_pos, write_val):
    idx_start = head_pos * cell_size
    idx_end = idx_start + cell_size
    # 写1时叠加波包，写0时抑制波包
    if write_val == 1:
        psi += gaussian_1d((head_pos + 0.5) * cell_size, sigma=cell_size/4, amp=3.0)
    else:
        psi[idx_start:idx_end] *= 0.3
    return psi

# 时间演化循环
for step in range(steps):
    # 动能项（简化1D拉普拉斯）
    laplacian = (np.roll(psi, 1) + np.roll(psi, -1) - 2 * psi) / dx**2
    phi = phi_res_1d(step * dt)
    gamma = smooth_gamma(psi)
    nonlin = nonlinear_feedback(psi)

    dpsi_dt = -1j * (-0.5 * laplacian + phi * psi + nonlin) - gamma * psi
    psi += dpsi_dt * dt

    # 归一化
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    if norm > 1e-10:
        psi /= norm

    # 图灵机读写跳转（每隔固定步数模拟一条“指令周期”）
    if step % 40 == 0:
        symbol = read_symbol(psi, head_pos)
        if (state, symbol) in transition:
            new_state, write_val, move = transition[(state, symbol)]
            psi = write_symbol(psi, head_pos, write_val)
            head_pos += move
            head_pos = max(0, min(tape_length - 1, head_pos))  # 限制边界
            state = new_state

    if step % 100 == 0:
        print(f"Step {step}, Norm: {norm:.4f}, State: {state}, Head: {head_pos}, Max gamma: {np.max(gamma):.3f}")

# 波函数强度可视化
plt.figure(figsize=(10,4))
plt.plot(X, np.abs(psi)**2)
plt.title("Wavefunction Intensity on 1D Tape after Evolution")
plt.xlabel("Position")
plt.ylabel("|ψ|² intensity")
plt.grid(True)
plt.show()
