import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter

# 系统的空间范围与网格点数 (进一步减小以防内存问题)
L = 10
N = 20  # 减小网格点
dx = L / N
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# 高斯波包的初始条件 (2D)
x0 = 5
y0 = 5
sigma = 0.5
k0x = 5.0
k0y = 5.0
psi_0_2d = np.exp(-0.5 * (((X - x0) / sigma)**2 + ((Y - y0) / sigma)**2)) * np.exp(1j * (k0x * X + k0y * Y))
psi_0_flat = psi_0_2d.flatten()  # 扁平化用于solve_ivp

# 非厄米耗散项Gamma (2D, 加更强平滑)
def gamma_2d(psi_flat, dx, eta, smooth_sigma=3.0):
    psi_2d = psi_flat.reshape((N, N))
    abs_psi = np.abs(psi_2d)
    smoothed_abs = gaussian_filter(abs_psi, sigma=smooth_sigma)  # 2D高斯平滑增强
    grad_x = np.gradient(smoothed_abs, dx, axis=0)
    grad_y = np.gradient(smoothed_abs, dx, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return -eta * grad_mag

# 动态外势项phi_res (2D)
def phi_res_2d(X, Y, t, A=0.5, omega=2*np.pi, x0=5, y0=5):
    return A * np.cos(omega * t) * np.exp(-((X - x0)**2 + (Y - y0)**2))

# 定义哈密顿量函数 (2D)
def Hamiltonian_2d(psi_flat, t, eta):
    psi_2d_real = np.real(psi_flat.reshape((N, N)))
    psi_2d_imag = np.imag(psi_flat.reshape((N, N)))

    # 2D拉普拉斯 (二阶导)
    d2_real_x = np.roll(psi_2d_real, -1, axis=0) - 2 * psi_2d_real + np.roll(psi_2d_real, 1, axis=0)
    d2_real_y = np.roll(psi_2d_real, -1, axis=1) - 2 * psi_2d_real + np.roll(psi_2d_real, 1, axis=1)
    d2_imag_x = np.roll(psi_2d_imag, -1, axis=0) - 2 * psi_2d_imag + np.roll(psi_2d_imag, 1, axis=0)
    d2_imag_y = np.roll(psi_2d_imag, -1, axis=1) - 2 * psi_2d_imag + np.roll(psi_2d_imag, 1, axis=1)

    lap_real = (d2_real_x + d2_real_y) / dx**2
    lap_imag = (d2_imag_x + d2_imag_y) / dx**2

    phi = phi_res_2d(X, Y, t)
    gamma_term = gamma_2d(psi_flat, dx, eta)

    H_real = -0.5 * lap_real + phi  # 简化V_res=0
    H_imag = -0.5 * lap_imag + gamma_term * psi_2d_imag  # 加gamma到虚部

    H_2d = H_real + 1j * H_imag
    return H_2d.flatten()

# 定义时间演化方程
def time_derivative(t, psi_flat, eta):
    H_psi_flat = Hamiltonian_2d(psi_flat, t, eta)
    return -1j * H_psi_flat

# 计算N(t) (2D积分)
def calculate_N(psi_flat, dx):
    psi_2d = psi_flat.reshape((N, N))
    return np.trapz(np.trapz(np.abs(psi_2d)**2, dx=dx, axis=0), dx=dx)

# 数值求解（加try-except捕获异常）
def solve_with_eta(eta):
    t_span = (0, 0.5)  # 进一步缩短时间
    t_eval = np.linspace(0, 0.5, 10)  # 减少点
    try:
        sol = solve_ivp(time_derivative, t_span, psi_0_flat, t_eval=t_eval, args=(eta,), method='BDF', rtol=1e-7, atol=1e-7, max_step=0.0001)  # BDF for stiff, 缩小精度和步长
        if not sol.success:
            print(f"Warning: Solver failed for eta={eta}")
            return None, None
    except Exception as e:
        print(f"Error for eta={eta}: {e}")
        return None, None
    
    N_values = []
    for i in range(len(t_eval)):
        psi_flat = sol.y[:, i]
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
plt.title('Norm N(t) Evolution for Different eta Values in 2D')
plt.show()