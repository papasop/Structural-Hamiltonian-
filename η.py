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
def gamma(psi, dx, eta):
    return -eta * np.gradient(np.abs(psi), dx)

# 动态外势项phi_res
def phi_res(x, t, A=0.5, omega=2*np.pi, x0=10):
    return A * np.cos(omega * t) * np.exp(-(x - x0)**2)

# 计算哈密顿量的每一项
H0 = -0.5 * np.roll(np.eye(N), -1, axis=0) + np.roll(np.eye(N), 1, axis=0)
H0 = H0 / dx**2  # 规范化为二阶导数
V_res = np.zeros(N)  # 外势项，可以在此加入

# 定义哈密顿量函数
def Hamiltonian(psi, t, eta):
    psi_real = np.real(psi)
    psi_imag = np.imag(psi)
    d2psi_real = np.roll(psi_real, -1) - 2 * psi_real + np.roll(psi_real, 1)
    d2psi_imag = np.roll(psi_imag, -1) - 2 * psi_imag + np.roll(psi_imag, 1)
    phi = phi_res(x, t)
    gamma_term = gamma(psi, dx, eta)
    return -0.5 * (d2psi_real + d2psi_imag) / dx**2 + V_res * psi_real + 1j * gamma_term * psi_imag + 1j * phi

# 定义时间演化方程
def time_derivative(t, psi, eta):
    H_psi = Hamiltonian(psi, t, eta)
    return -1j * H_psi

# 计算N(t)
def calculate_N(psi, dx):
    return np.trapz(np.abs(psi)**2, x)

# 数值求解
def solve_with_eta(eta):
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 500)

    # 使用solve_ivp进行时间演化，采用RK45方法
    sol = solve_ivp(time_derivative, t_span, psi_0, t_eval=t_eval, args=(eta,), method='RK45', rtol=1e-6, max_step=0.001)
    
    N_values = []
    for i in range(len(t_eval)):
        psi = sol.y[:, i]
        N = calculate_N(psi, dx)
        N_values.append(N)
        
    return sol, N_values

# 测试不同的eta值
eta_values = [0.1, 0.5, 1.0]

for eta in eta_values:
    sol, N_values = solve_with_eta(eta)
    plt.plot(sol.t, N_values, label=f'eta={eta}')

plt.xlabel('Time (t)')
plt.ylabel('Norm N(t)')
plt.legend()
plt.title('Norm N(t) Evolution for Different eta Values')
plt.show()
