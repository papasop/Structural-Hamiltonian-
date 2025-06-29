import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 系统的空间范围与网格点数
L = 20
N = 200
dx = L / N
x = np.linspace(0, L, N)

# 高斯波包的初始条件
x0 = 10
sigma = 0.5
k0 = 5.0
psi_0 = np.exp(-0.5 * ((x - x0) / sigma)**2) * np.exp(1j * k0 * x)

# 非厄米耗散项Gamma
eta = 0.1
Gamma = -eta * np.gradient(np.abs(psi_0), dx)

# 动态外势项phi_res
def phi_res(x, t, A=0.5, omega=2*np.pi, x0=10):
    return A * np.cos(omega * t) * np.exp(-(x - x0)**2)

# 计算哈密顿量的每一项
H0 = -0.5 * np.roll(np.eye(N), -1, axis=0) + np.roll(np.eye(N), 1, axis=0)
H0 = H0 / dx**2  # 规范化为二阶导数
V_res = np.zeros(N)  # 外势项，可以在此加入

# 定义哈密顿量函数
def Hamiltonian(psi, t):
    psi_real = np.real(psi)
    psi_imag = np.imag(psi)
    d2psi_real = np.roll(psi_real, -1) - 2 * psi_real + np.roll(psi_real, 1)
    d2psi_imag = np.roll(psi_imag, -1) - 2 * psi_imag + np.roll(psi_imag, 1)
    phi = phi_res(x, t)
    return -0.5 * (d2psi_real + d2psi_imag) / dx**2 + V_res * psi_real + 1j * Gamma * psi_imag + 1j * phi

# 定义时间演化方程
def time_derivative(t, psi):
    H_psi = Hamiltonian(psi, t)
    return -1j * H_psi

# 归一化波函数
def normalize_wavefunction(psi):
    norm = np.sum(np.abs(psi)**2 * dx)
    return psi / np.sqrt(norm)

# 增强吸收边界
def apply_absorbing_layer(psi, x, w=2.0):
    mask = np.ones_like(x)
    mask[x < 2] = np.exp(-w * (2 - x[x < 2])**2)
    mask[x > 18] = np.exp(-w * (x[x > 18] - 18)**2)
    return psi * mask

# 归一化误差
def normalization_error(psi):
    norm = np.sum(np.abs(psi)**2 * dx)
    return np.abs(norm - 1)

# 数值求解
def solve_with_fine_step():
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 500)
    
    # 使用solve_ivp进行时间演化，采用RK45方法
    sol = solve_ivp(time_derivative, t_span, psi_0, t_eval=t_eval, method='RK45', rtol=1e-6, max_step=0.001)
    
    # 存储归一化波函数与能量
    norm_errors = []
    for i in range(len(t_eval)):
        psi = sol.y[:, i]
        psi = normalize_wavefunction(psi)  # 归一化波函数
        norm_errors.append(normalization_error(psi))  # 计算归一化误差
        
        # 应用吸收边界
        psi = apply_absorbing_layer(psi, x)
        sol.y[:, i] = psi
    
    return sol, norm_errors

# 调用求解函数并输出归一化误差
sol, norm_errors = solve_with_fine_step()

# 输出最终归一化误差
print(f"最终归一化误差: {norm_errors[-1]}")

# 绘制归一化误差
plt.figure(figsize=(10, 6))
plt.plot(sol.t, norm_errors)
plt.xlabel('Time (t)')
plt.ylabel('Normalization Error')
plt.title('Normalization Error Evolution')
plt.show()