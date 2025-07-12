# 运行此代码前，确保在Colab中运行：Runtime > Change runtime type > GPU (加速)
import numpy as np
import matplotlib.pyplot as plt

# 参数设定：模拟无限3D网格作为图灵带 (大N模拟无限)
L = 10  # 空间范围
N = 20  # 网格点，3D N^3 = 8000点模拟无限带
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
z = np.linspace(0, L, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
dx = x[1] - x[0]

# 时间参数
dt = 0.01
steps = 300  # 增强演化

# 3D 高斯波包初始化带 (无限带模拟有限段)
def gaussian_3d(x0, y0, z0, sigma=1.0, amp=10.0):
    r2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
    return amp * np.exp(-r2 / sigma**2)

# 初始波函数 ψ (带状态0/1作为峰值)
tape = [0, 1, 0]  # 模拟带，无限扩展用pad
psi = np.zeros((N, N, N), dtype=np.complex128)
gamma = np.zeros((N, N, N))  # 初始化gamma
for i, bit in enumerate(tape):
    if bit:
        psi += gaussian_3d(5 + i*2, 10, 10)

# 头位置和状态 (图灵机)
head_pos = 1  # 初始头
state = 0  # 初始状态

# 异常点跳转规则 (EP作为跳转)
transition = {
    (0, 0): (1, 1, 1),  # 状态0读0 -> 写1, 右移, 新状态1
    (1, 0): (0, 1, -1),  # 状态1读0 -> 写1, 左移, 新状态0
    (0, 1): (0, 0, 1),  # 状态0读1 -> 写0, 右移, 新状态0
    (1, 1): (1, 0, -1),  # 状态1读1 -> 写0, 左移, 新状态1
}

# 耗散强度
eta = 5.0  # 耗散强度

# 时间演化（复波函数 + 非厄米耗散 + 异常点跳转）
for _ in range(steps):
    lap = (
        np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
        np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) +
        np.roll(psi, 1, axis=2) + np.roll(psi, -1, axis=2) - 6 * psi
    ) / dx**2
    dpsi_dt = -1j * (-0.5 * lap) - gamma * psi
    psi += dpsi_dt * dt
    psi = np.clip(psi, -1, 1)  # 防溢出

    # 异常点跳转机制 (模拟读/写/移)
    head_slice = slice(head_pos*5, min((head_pos*5)+5, psi.shape[2]))
    symbol = 1 if np.sum(np.abs(psi[:, :, head_slice])**2) > 0.5 else 0
    if (state, symbol) in transition:
        new_state, write, move = transition[(state, symbol)]
        if write:
            pad_size = 5 if move > 0 else 0
            psi = np.pad(psi, ((0,0), (0,0), (0,pad_size)), mode='constant')  # 扩展无限带
            gamma = np.pad(gamma, ((0,0), (0,0), (0,pad_size)), mode='constant')  # 扩展gamma
            psi[:, :, head_pos*5: (head_pos*5)+5] += gaussian_3d(head_pos*2, 10, 10)[:, :, :5]
        else:
            psi[:, :, head_slice] *= 0.5  # 衰减0
        head_pos += move
        state = new_state
        # 无限带扩展 (循环边界)
        if head_pos < 0:
            head_pos = psi.shape[2]//5 - 1  # 循环
            psi = np.roll(psi, shift=5, axis=2)  # 循环移位
            gamma = np.roll(gamma, shift=5, axis=2)
        elif head_pos >= psi.shape[2]//5:
            psi = np.pad(psi, ((0,0), (0,0), (0,5)), mode='constant')  # 扩展
            gamma = np.pad(gamma, ((0,0), (0,0), (0,5)), mode='constant')
            head_pos = psi.shape[2]//5 - 1

# 输出强度检测 (无限带模拟有限切片)
def output_intensity(mask):
    return np.sum(np.abs(psi[mask])**2) * dx**3

# 可视化 z=10 中间层
plt.figure(figsize=(8, 4))
plt.imshow(np.abs(psi[:, :, psi.shape[2]//2])**2, extent=[0, L, 0, L], origin='lower', cmap='viridis')
plt.title("Final |ψ(x,y,z=10)|² for Turing Simulation")
plt.xlabel("x (Tape Position)")
plt.ylabel("y")
plt.colorbar(label="Intensity")
plt.grid(False)
plt.show()

# 打印最终带/状态/头
tape = [1 if np.sum(np.abs(psi[:, :, i*5:(i*5)+5])**2) > 0.5 else 0 for i in range(psi.shape[2]//5)]
print("Final Tape:", tape)
print("Final State:", state)
print("Final Head Position:", head_pos)