import numpy as np
import matplotlib.pyplot as plt

L = 10
N = 20
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
z = np.linspace(0, L, N)
dx = x[1] - x[0]
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

dt = 0.005
steps = 500

def gaussian_3d(x0, y0, z0, sigma=0.8, amp=5.0):
    r2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
    return amp * np.exp(-r2 / (2 * sigma**2))

def phi_res_3d(t, omega=2*np.pi/5, amp=3.0):
    return amp * np.cos(omega * t) * np.exp(-((X - 5)**2 + (Y - 5)**2 + (Z - 5)**2)/4)

def smooth_gamma(psi, threshold=1.0, slope=5.0, base=6.0):
    grad_x = np.gradient(np.abs(psi), dx, axis=0)
    grad_y = np.gradient(np.abs(psi), dx, axis=1)
    grad_z = np.gradient(np.abs(psi), dx, axis=2)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    gamma = base / (1 + np.exp(-slope * (grad_mag - threshold)))
    return gamma, grad_mag

def nonlinear_feedback(psi, coeff=0.3):
    return coeff * np.abs(psi)**2 * psi

# 简单异常点状态跳转示范（两态切换）
def ep_state_update(state, max_gamma, gamma_threshold=0.1):
    # 当最大gamma超过阈值时，状态翻转
    if max_gamma > gamma_threshold:
        return 1 - state
    return state

# 参数扫描空间
thresholds = [0.5, 1.0, 1.5]
bases = [4.0, 6.0, 8.0]
slopes = [3.0, 5.0, 8.0]

for threshold in thresholds:
    for base in bases:
        for slope in slopes:
            print(f"Testing threshold={threshold}, base={base}, slope={slope}")
            psi = gaussian_3d(3,3,3,sigma=1.0,amp=5.0) + 1j*gaussian_3d(7,7,7,sigma=1.0,amp=5.0)
            ep_state = 0
            states = []
            max_gammas = []
            norms = []

            for step in range(steps):
                laplacian = (
                    np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
                    np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) +
                    np.roll(psi, 1, axis=2) + np.roll(psi, -1, axis=2) -
                    6 * psi
                ) / dx**2

                phi = phi_res_3d(step*dt)
                gamma, grad_mag = smooth_gamma(psi, threshold=threshold, slope=slope, base=base)
                nonlin = nonlinear_feedback(psi)

                dpsi_dt = -1j * (-0.5 * laplacian + phi * psi + nonlin) - gamma * psi
                psi += dpsi_dt * dt

                norm = np.sqrt(np.sum(np.abs(psi)**2) * dx**3)
                if norm > 1e-10:
                    psi /= norm

                max_gamma = np.max(gamma)
                ep_state = ep_state_update(ep_state, max_gamma)

                states.append(ep_state)
                max_gammas.append(max_gamma)
                norms.append(norm)

            print(f"Final EP state: {ep_state}, Max gamma last: {max_gammas[-1]:.3f}")
            plt.figure(figsize=(10,3))
            plt.subplot(1,2,1)
            plt.plot(states, label='EP State (0/1)')
            plt.xlabel('Step')
            plt.ylabel('State')
            plt.title('Anomaly Point State Evolution')
            plt.legend()

            plt.subplot(1,2,2)
            plt.plot(max_gammas, label='Max gamma')
            plt.xlabel('Step')
            plt.ylabel('Max gamma')
            plt.title('Max Gamma Evolution')
            plt.legend()
            plt.tight_layout()
            plt.show()
