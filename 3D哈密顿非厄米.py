# 运行此代码前，确保在Colab中运行：Runtime > Change runtime type > GPU (加速)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
import torch  # GPU加速

# 系统的空间范围与网格点数 (3D简单测试)
L = 5
N = 10  # 10x10x10网格，1000点
dx = L / N
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
z = np.linspace(0, L, N)
X, Y, Z = np.meshgrid(x, y, z)

# 高斯波包的初始条件 (3D)
x0 = 2.5
y0 = 2.5
z0 = 2.5
sigma = 0.5
k0x = 5.0
k0y = 5.0
k0z = 5.0
psi_0_3d = np.exp(-0.5 * (((X - x0) / sigma)**2 + ((Y - y0) / sigma)**2 + ((Z - z0) / sigma)**2)) * np.exp(1j * (k0x * X + k0y * Y + k0z * Z))
psi_0_flat = psi_0_3d.flatten()  # 扁平化用于solve_ivp

# 非厄米耗散项Gamma (3D, 简单平滑)
def gamma_3d(psi_flat, dx, eta, smooth_sigma=2.0):
    psi_3d = psi_flat.reshape((N, N, N))
    abs_psi = np.abs(psi_3d)
    smoothed_abs = gaussian_filter(abs_psi, sigma=smooth_sigma)  # 3D高斯平滑
    grad_x = np.gradient(smoothed_abs, dx, axis=0)
    grad_y = np.gradient(smoothed_abs, dx, axis=1)
    grad_z = np.gradient(smoothed_abs, dx, axis=2)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    return -eta * grad_mag.flatten()

# 动态外势项phi_res (3D)
def phi_res_3d(X, Y, Z, t, A=0.5, omega=2*np.pi, x0=2.5, y0=2.5, z0=2.5):
    return A * np.cos(omega * t) * np.exp(-((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2))

# 3D吸收层 (防止边界反射)
def apply_absorbing_layer_3d(psi_flat, x, y, z, w=2.0):
    psi_3d = psi_flat.reshape((N, N, N))
    mask = np.exp(-w * ((X - L/2)**2 + (Y - L/2)**2 + (Z - L/2)**2) / L**2)
    return (psi_3d * mask).flatten()

# 定义哈密顿量函数 (3D, 简化拉普拉斯)
def Hamiltonian_3d(psi_flat, t, eta):
    psi_3d = psi_flat.reshape((N, N, N))
    psi_3d_real = np.real(psi_3d)
    psi_3d_imag = np.imag(psi_3d)

    # 3D拉普拉斯 (二阶导，简化)
    d2_real_x = np.roll(psi_3d_real, -1, axis=0) - 2 * psi_3d_real + np.roll(psi_3d_real, 1, axis=0)
    d2_real_y = np.roll(psi_3d_real, -1, axis=1) - 2 * psi_3d_real + np.roll(psi_3d_real, 1, axis=1)
    d2_real_z = np.roll(psi_3d_real, -1, axis=2) - 2 * psi_3d_real + np.roll(psi_3d_real, 1, axis=2)
    d2_imag_x = np.roll(psi_3d_imag, -1, axis=0) - 2 * psi_3d_imag + np.roll(psi_3d_imag, 1, axis=0)
    d2_imag_y = np.roll(psi_3d_imag, -1, axis=1) - 2 * psi_3d_imag + np.roll(psi_3d_imag, 1, axis=1)
    d2_imag_z = np.roll(psi_3d_imag, -1, axis=2) - 2 * psi_3d_imag + np.roll(psi_3d_imag, 1, axis=2)

    lap_real = (d2_real_x + d2_real_y + d2_real_z) / dx**2
    lap_imag = (d2_imag_x + d2_imag_y + d2_imag_z) / dx**2

    phi = phi_res_3d(X, Y, Z, t)
    gamma_term = gamma_3d(psi_flat, dx, eta).reshape((N, N, N))

    H_real = -0.5 * lap_real + phi
    H_imag = -0.5 * lap_imag + gamma_term * psi_3d_imag

    H_3d = H_real + 1j * H_imag
    return H_3d.flatten()

# 定义时间演化方程
def time_derivative(t, psi_flat, eta):
    H_psi_flat = Hamiltonian_3d(psi_flat, t, eta)
    return -1j * H_psi_flat

# 计算N(t) (3D三重积分)
def calculate_N(psi_flat, dx):
    psi_3d = psi_flat.reshape((N, N, N))
    return np.trapz(np.trapz(np.trapz(np.abs(psi_3d)**2, dx=dx, axis=2), dx=dx, axis=1), dx=dx)

# 数值求解（加try-except捕获异常）
def solve_with_eta(eta):
    t_span = (0, 0.2)  # 超短时间减少计算
    t_eval = np.linspace(0, 0.2, 10)  # 减少点
    try:
        sol = solve_ivp(time_derivative, t_span, psi_0_flat, t_eval=t_eval, args=(eta,), method='BDF', rtol=1e-7, atol=1e-7, max_step=0.0001)
        if not sol.success:
            print(f"Warning: Solver failed for eta={eta}")
            return None, None
    except Exception as e:
        print(f"Error for eta={eta}: {e}")
        return None, None
    
    N_values = []
    for i in range(len(t_eval)):
        psi_flat = sol.y[:, i]
        psi_flat = apply_absorbing_layer_3d(psi_flat, x, y, z)  # 应用吸收层
        N = calculate_N(psi_flat, dx)
        N_values.append(N)
        
    return sol, N_values

# 测试不同的eta值
eta_values = [0.1, 0.5, 1.0]

for eta in eta_values:
    sol, N_values = solve_with_eta(eta)
    if sol is not None:
        plt.plot(sol.t, N_values, label=f'eta={eta}')

plt.xlabel('Time (t)')
plt.ylabel('Norm N(t)')
plt.legend()
plt.title('Norm N(t) Evolution for Different eta Values in 3D')
plt.show()