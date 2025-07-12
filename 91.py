import numpy as np
import matplotlib.pyplot as plt

# 参数设置
L = 10
N = 30
dx = L / N
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y, indexing='ij')

dt = 0.005
steps = 3000

# 多符号波包
def gaussian_2d(x0, y0, sigma=0.4, amp=4.0):
    return amp * np.exp(-0.5 * (((X - x0)/sigma)**2 + ((Y - y0)/sigma)**2))

# 动态外势
def phi_res_2d(t, omega=2*np.pi/15, amp=3.0):
    return amp * np.cos(omega * t) * np.exp(-((X - 5)**2 + (Y - 5)**2)/5)

# 异常点耗散激活
def smooth_gamma(psi, threshold=1.0, slope=6.0, base=5.0):
    grad_x = np.gradient(np.abs(psi), dx, axis=0)
    grad_y = np.gradient(np.abs(psi), dx, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    gamma = base / (1 + np.exp(-slope * (grad_mag - threshold)))
    return gamma

# 非线性反馈
def nonlinear_feedback(psi, coeff=0.3):
    return coeff * np.abs(psi)**2 * psi

# 读写带配置
tape_size = 9
cell_size = N // tape_size

# 多头初始化（位置分散）
heads = [
    {'pos': [2, 3], 'state': 0},
    {'pos': [7, 6], 'state': 1}
]

symbols = [0, 1, 2, 3]
num_states = 6
state_counters = {s: 0 for s in range(num_states)}
max_stay = 50  # 状态最大停留步数限制

# 初始化波函数，随机符号填充
psi = np.zeros((N, N), dtype=np.complex128)
for i in range(tape_size):
    for j in range(tape_size):
        sym = np.random.choice(symbols)
        if sym != 0:
            psi += gaussian_2d((i + 0.5)*cell_size, (j + 0.5)*cell_size, sigma=cell_size/4, amp=4.0*sym)

# 复杂跳转规则 (state, symbol, neighbor_symbol) -> (new_state, write_symbol, move_x, move_y)
transition = {
    (0, 0, 0): (1, 1, 1, 0),
    (0, 1, 0): (2, 2, 0, 1),
    (1, 1, 1): (3, 0, -1, 0),
    (2, 2, 2): (4, 3, 0, -1),
    (3, 2, 3): (5, 1, -1, 1),
    (4, 3, 0): (0, 3, 1, -1),
    (5, 1, 2): (1, 2, 1, 1),
}

def read_symbol(psi, pos):
    x_idx, y_idx = pos
    xs = slice(x_idx*cell_size, (x_idx+1)*cell_size)
    ys = slice(y_idx*cell_size, (y_idx+1)*cell_size)
    intensities = [np.sum(np.abs(psi[xs, ys])**2) * s for s in symbols]
    max_idx = np.argmax(intensities)
    return symbols[max_idx]

def write_symbol(psi, pos, write_val):
    x_idx, y_idx = pos
    xs = slice(x_idx*cell_size, (x_idx+1)*cell_size)
    ys = slice(y_idx*cell_size, (y_idx+1)*cell_size)
    psi[xs, ys] *= 0.3  # 衰减旧符号
    if write_val != 0:
        psi += gaussian_2d((x_idx+0.5)*cell_size, (y_idx+0.5)*cell_size, sigma=cell_size/4, amp=3.5*write_val)
    return psi

def add_noise(psi, noise_level=0.01):
    noise = (np.random.randn(*psi.shape) + 1j * np.random.randn(*psi.shape)) * noise_level
    return psi + noise

def get_neighbor_symbol(psi, pos):
    x,y = pos
    neighbors = []
    for dx_ in [-1,0,1]:
        for dy_ in [-1,0,1]:
            if dx_ == 0 and dy_ == 0:
                continue
            nx, ny = x+dx_, y+dy_
            if 0 <= nx < tape_size and 0 <= ny < tape_size:
                neighbors.append(read_symbol(psi, [nx, ny]))
    return max(neighbors) if neighbors else 0

# 主循环
for step in range(steps):
    laplacian = (
        np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
        np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) -
        4 * psi
    ) / dx**2

    phi = phi_res_2d(step*dt)
    gamma = smooth_gamma(psi)
    nonlin = nonlinear_feedback(psi)

    noise_level = 0.005 if step > 400 else 0.0
    psi = add_noise(psi, noise_level)

    dpsi_dt = -1j * (-0.5 * laplacian + phi * psi + nonlin) - gamma * psi
    psi += dpsi_dt * dt

    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx**2)
    if norm > 1e-10:
        psi /= norm

    if step % 60 == 0:
        for head in heads:
            pos = head['pos']
            state = head['state']
            sym = read_symbol(psi, pos)
            nbsym = get_neighbor_symbol(psi, pos)
            key = (state, sym, nbsym)
            if key in transition:
                new_state, write_sym, move_x, move_y = transition[key]

                # 状态计数器防死循环
                state_counters[new_state] += 1
                if state_counters[new_state] > max_stay:
                    new_state = (new_state + 1) % num_states
                    state_counters[new_state] = 0

                psi = write_symbol(psi, pos, write_sym)
                head['pos'][0] = np.clip(pos[0] + move_x, 0, tape_size - 1)
                head['pos'][1] = np.clip(pos[1] + move_y, 0, tape_size - 1)
                head['state'] = new_state

    if step % 100 == 0:
        print(f"Step {step}, Norm: {norm:.4f}, Heads:")
        for i, head in enumerate(heads):
            print(f"  Head {i}: State={head['state']}, Pos={head['pos']}, Max gamma={np.max(gamma):.3f}")

plt.figure(figsize=(6,6))
plt.imshow(np.abs(psi)**2, extent=[0,L,0,L], origin='lower')
plt.colorbar(label='|ψ|² intensity')
plt.title('Wavefunction Intensity (2D Tape Multi-head)')
plt.show()
