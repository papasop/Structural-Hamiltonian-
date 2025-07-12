# 运行此代码前，确保在Colab中运行：Runtime > Change runtime type > GPU (加速)
import numpy as np
import matplotlib.pyplot as plt

# 参数定义
L = 10
N = 30  # 测试小网格，增大到100+观察GPU加速
dx = L / N
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
z = np.linspace(0, L, N)  # 为3D添加
X, Y = np.meshgrid(x, y, indexing='ij')
X3, Y3, Z3 = np.meshgrid(x, y, z, indexing='ij')  # 3D网格

dt = 0.005
steps = 5000  # 超长演化

num_bands = 3  # 多带TM：添加3带
use_3d_tape = True  # 启用3D tape扩展

# 3D/2D高斯波包
def gaussian_nd(x0, y0, z0=None, sigma=0.4, amp=0.1, dim=2):  # 增amp=0.1*val
    if dim == 2:
        return amp * np.exp(-0.5 * (((X - x0)/sigma)**2 + ((Y - y0)/sigma)**2))
    else:
        return amp * np.exp(-0.5 * (((X3 - x0)/sigma)**2 + ((Y3 - y0)/sigma)**2 + ((Z3 - z0)/sigma)**2))

# 动态外势（适应3D）
def phi_res_nd(t, omega=2*np.pi/15, amp=3.0, dim=2):
    if dim == 2:
        return amp * np.cos(omega * t) * np.exp(-((X - 5)**2 + (Y - 5)**2)/5)
    else:
        return amp * np.cos(omega * t) * np.exp(-((X3 - 5)**2 + (Y3 - 5)**2 + (Z3 - 5)**2)/5)

# 光滑gamma（适应3D梯度）
def smooth_gamma(psi, threshold=1.0, slope=6.0, base=5.0, dim=2):
    if dim == 2:
        grad_x = np.gradient(np.abs(psi), dx, axis=0)
        grad_y = np.gradient(np.abs(psi), dx, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    else:
        grad_x = np.gradient(np.abs(psi), dx, axis=0)
        grad_y = np.gradient(np.abs(psi), dx, axis=1)
        grad_z = np.gradient(np.abs(psi), dx, axis=2)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    return base / (1 + np.exp(-slope * (grad_mag - threshold)))

# 非线性反馈
def nonlinear_feedback(psi, coeff=0.15):
    return coeff * np.abs(psi)**2 * psi

# 符号与带
symbols = [0, 1, 2, 3]
tape_size = 9
cell_size = N // tape_size

# 读取符号（适应多带/3D，使用阈值防止偏向高sym）
def read_symbol(psi, pos, band=0, dim=2):
    if dim == 2:
        x_idx, y_idx = pos
        xs = slice(x_idx*cell_size, (x_idx+1)*cell_size)
        ys = slice(y_idx*cell_size, (y_idx+1)*cell_size)
        sum_intensity = np.sum(np.abs(psi[xs, ys])**2)
    else:
        x_idx, y_idx, z_idx = pos
        xs = slice(x_idx*cell_size, (x_idx+1)*cell_size)
        ys = slice(y_idx*cell_size, (y_idx+1)*cell_size)
        zs = slice(z_idx*cell_size, (z_idx+1)*cell_size)
        sum_intensity = np.sum(np.abs(psi[xs, ys, zs])**2)
    # 减阈值[0,0.01,0.02,0.03]，debug print intensity
    thresholds = [0, 0.01, 0.02, 0.03]  # 减阈值
    sym = 0
    for th in thresholds[1:]:
        if sum_intensity >= th:
            sym += 1
    # Debug intensity
    if np.random.rand() < 0.1:  # 随机10%打印
        print(f"Debug intensity: band={band}, pos={pos}, sum={sum_intensity:.3f}, sym={sym}")
    return min(sym, 3)

# 读取邻居（适应多带/3D）
def get_neighbor_symbol(psi, pos, band=0, dim=2):
    neighbors = []
    if dim == 2:
        x, y = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < tape_size and 0 <= ny < tape_size:
                    neighbors.append(read_symbol(psi, [nx, ny], band, dim))
    else:
        x, y, z = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0: continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if all(0 <= n < tape_size for n in [nx, ny, nz]):
                        neighbors.append(read_symbol(psi, [nx, ny, nz], band, dim))
    return max(neighbors) if neighbors else 0

# 写符号（适应多带/3D，移除*0.3阻尼，靠全局norm）
def write_symbol(psi, pos, val, band=0, dim=2):
    if dim == 2:
        x_idx, y_idx = pos
        xs = slice(x_idx*cell_size, (x_idx+1)*cell_size)
        ys = slice(y_idx*cell_size, (y_idx+1)*cell_size)
        # 移除*0.3
        if val != 0:
            psi += gaussian_nd((x_idx+0.5)*cell_size, (y_idx+0.5)*cell_size, amp=0.1*val)
    else:
        x_idx, y_idx, z_idx = pos
        xs = slice(x_idx*cell_size, (x_idx+1)*cell_size)
        ys = slice(y_idx*cell_size, (y_idx+1)*cell_size)
        zs = slice(z_idx*cell_size, (z_idx+1)*cell_size)
        # 移除*0.3
        if val != 0:
            psi += gaussian_nd((x_idx+0.5)*cell_size, (y_idx+0.5)*cell_size, (z_idx+0.5)*cell_size, amp=0.1*val, dim=3)
    return psi

# 噪声
def add_noise(psi, noise_level=0.01):
    shape = psi.shape
    noise = (np.random.randn(*shape) + 1j*np.random.randn(*shape)) * noise_level
    return psi + noise

# 多控制单元
controllers = [
    {'heads': [{'pos': [2, 3, tape_size//2] if use_3d_tape else [2, 3], 'state': 0}], 'id': 0},
    {'heads': [{'pos': [7, 6, tape_size//2] if use_3d_tape else [7, 6], 'state': 1}], 'id': 1},
]

num_states = 6
state_counters = [0] * len(controllers)
max_stay = 50

# transition扩展dz_move基于sym规则（if sym > 1 then dz=1 else -1）
def get_dz_move(sym):
    return 1 if sym > 1 else -1

transition = {
    (0, 0, 0): (1, 1, 1, 0, get_dz_move(0)),
    (0, 1, 0): (2, 2, 0, 1, get_dz_move(1)),
    (1, 1, 1): (3, 0, -1, 0, get_dz_move(1)),
    (2, 2, 2): (4, 3, 0, -1, get_dz_move(2)),
    (3, 2, 3): (5, 1, -1, 1, get_dz_move(2)),
    (4, 3, 0): (0, 3, 1, -1, get_dz_move(3)),
    (5, 1, 2): (1, 2, 1, 1, get_dz_move(1)),
    # UTM示例：加法扩展循环
    (0, 1, 1): (0, 2, 1, 0, get_dz_move(1)),
    (0, 2, 2): (0, 3, 1, 0, get_dz_move(2)),
    (0, 3, 3): (0, 1, 1, 0, get_dz_move(3)),  # 循环加1
    # 添加更多key覆盖随机sym (e.g., for sym=3 nbsym=3)
    (0, 3, 3): (1, 0, -1, 0, get_dz_move(3)),  # 新添加覆盖(0,3,3)
    (1, 3, 3): (2, 1, 0, -1, get_dz_move(3)),  # 新添加覆盖(1,3,3)
    (2, 3, 3): (3, 2, 1, 0, get_dz_move(3)),
    (3, 3, 3): (4, 3, -1, 1, get_dz_move(3)),
    (4, 3, 3): (5, 0, 0, -1, get_dz_move(3)),
    (5, 3, 3): (0, 1, 1, 0, get_dz_move(3)),
    # 更多覆盖 (添加覆盖0-3所有组合)
    (0, 0, 1): (1, 2, -1, 1, get_dz_move(0)),
    (0, 0, 2): (2, 3, 0, -1, get_dz_move(0)),
    (0, 0, 3): (3, 1, 1, 0, get_dz_move(0)),
    (1, 0, 0): (4, 2, -1, 1, get_dz_move(0)),
    (1, 0, 1): (5, 3, 0, -1, get_dz_move(0)),
    (1, 0, 2): (0, 1, 1, 0, get_dz_move(0)),
    (1, 0, 3): (1, 2, -1, 1, get_dz_move(0)),
    # ... (类似添加所有 (state, sym, nbsym) 组合，确保100%覆盖)
    (5, 3, 0): (0, 3, 1, -1, get_dz_move(3)),
    (5, 3, 1): (1, 0, -1, 0, get_dz_move(3)),
    (5, 3, 2): (2, 1, 0, 1, get_dz_move(3)),
}

# 并行TM
parallel_tms = [
    {'controller_id': 0, 'tape_view': slice(0,1)},
    {'controller_id': 1, 'tape_view': slice(1,3)},
]

# 初始化psi，并初始化tape[0]为程序符号扩展 e.g., [[1,1,0,2,0]]模拟加法循环 (在[0,0-4,z]写1,1,0,2,0)
dim = 3 if use_3d_tape else 2
if dim == 2:
    psi = np.zeros((N, N), dtype=np.complex128)
else:
    psi = np.zeros((N, N, N), dtype=np.complex128)
# 随机初始化其他band
for band in range(1, num_bands):
    for i in range(tape_size):
        for j in range(tape_size):
            sym = np.random.choice(symbols)
            if sym != 0:
                if dim == 2:
                    psi += gaussian_nd((i+0.5)*cell_size, (j+0.5)*cell_size, amp=0.1*sym)
                else:
                    psi += gaussian_nd((i+0.5)*cell_size, (j+0.5)*cell_size, (band+0.5)*cell_size, amp=0.1*sym, dim=3)
# 初始化band 0程序: [1,1,0,2,0] at [0,0,z] to [0,4,z]
z_mid = tape_size//2
program = [1,1,0,2,0]
for j, sym in enumerate(program):
    psi = write_symbol(psi, [0, j, z_mid], sym, 0, dim)
# 初始化band 1数据: e.g., [1,0,z]=2测试加法 (预期变3)
psi = write_symbol(psi, [1, 0, z_mid], 2, 1, dim)
for j in range(1, tape_size):
    sym = np.random.choice(symbols)
    psi = write_symbol(psi, [1, j, z_mid], sym, 1, dim)
norm = np.sqrt(np.sum(np.abs(psi)**2) * dx**dim)
if norm > 1e-10:
    psi /= norm

# 打印初始tape检查
print("Initial Band 0 Tape:")
tape_band = [[read_symbol(psi, [i, j, z_mid] if dim==3 else [i, j], 0, dim) for j in range(tape_size)] for i in range(tape_size)]
for row in tape_band:
    print(row)
print("Initial Band 1 Tape:")
tape_band = [[read_symbol(psi, [i, j, z_mid] if dim==3 else [i, j], 1, dim) for j in range(tape_size)] for i in range(tape_size)]
for row in tape_band:
    print(row)

# 主演化循环
for step in range(steps):
    if dim == 2:
        laplacian = (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
                     np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi) / dx**2
    else:
        laplacian = (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
                     np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) +
                     np.roll(psi, 1, axis=2) + np.roll(psi, -1, axis=2) - 6 * psi) / dx**2
    phi = phi_res_nd(step * dt, dim=dim)
    gamma = smooth_gamma(psi, dim=dim)
    nonlin = nonlinear_feedback(psi)
    noise_level = 0.01 if step > 400 else 0.0
    psi = add_noise(psi, noise_level)
    dpsi_dt = -1j * (-0.5 * laplacian + phi * psi + nonlin) - gamma * psi
    psi += dpsi_dt * dt
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx**dim)
    if norm > 1e-10:
        psi /= norm

    if step % 60 == 0:
        for tm in parallel_tms:
            controller = controllers[tm['controller_id']]
            for head in controller['heads']:
                pos = head['pos']
                state = head['state']
                band_slice = tm['tape_view']
                sym = read_symbol(psi, pos, band_slice.start, dim)
                nbsym = get_neighbor_symbol(psi, pos, band_slice.start, dim)
                key = (state, sym, nbsym)
                # Debug print每100步
                if step % 100 == 0:
                    print(f"Debug: Controller {controller['id']}, sym={sym}, nbsym={nbsym}, key={key}")
                if key in transition:
                    new_state, write_sym, dx_move, dy_move, dz_move = transition[key]
                    if band_slice.start == 0 and write_sym == 1:  # UTM加法
                        data_sym = read_symbol(psi, pos, band_slice.stop - 1, dim)
                        write_sym = (data_sym + 1) % 4
                    state_counters[controller['id']] += 1
                    if state_counters[controller['id']] > max_stay:
                        new_state = (new_state + 1) % num_states
                        state_counters[controller['id']] = 0
                    psi = write_symbol(psi, pos, write_sym, band_slice.start, dim)
                    new_x = np.clip(pos[0] + dx_move, 0, tape_size - 1)
                    new_y = np.clip(pos[1] + dy_move, 0, tape_size - 1)
                    new_pos = [new_x, new_y]
                    if dim == 3:
                        new_z = np.clip(pos[2] + dz_move, 0, tape_size - 1)
                        new_pos.append(new_z)
                    head['pos'] = new_pos
                    head['state'] = new_state

    if step % 100 == 0:
        print(f"Step {step}, Norm: {norm:.4f}, Heads:")
        for controller in controllers:
            for i, head in enumerate(controller['heads']):
                print(f"  Controller {controller['id']} Head {i}: State={head['state']}, Pos={head['pos']}, Max gamma={np.max(gamma):.3f}")

# 可视化
if dim == 2:
    plt.imshow(np.abs(psi)**2, extent=[0, L, 0, L], origin='lower', cmap='inferno')
else:
    plt.imshow(np.abs(psi[:,:,tape_size//2])**2, extent=[0, L, 0, L], origin='lower', cmap='inferno')  # 3D切片
plt.colorbar(label='|ψ|² intensity')
plt.title('Optimized Non-Hermitian Multi-Module TM Evolution')
plt.show()

# 打印最终带
for band in range(num_bands):
    print(f"Band {band} Tape:")
    tape_band = [[read_symbol(psi, [i, j, tape_size//2] if dim==3 else [i, j], band, dim) for j in range(tape_size)] for i in range(tape_size)]
    for row in tape_band:
        print(row)
for controller in controllers:
    for i, head in enumerate(controller['heads']):
        print(f"Final Controller {controller['id']} Head {i}: State={head['state']}, Pos={head['pos']}")