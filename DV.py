import numpy as np
import scipy.linalg as la
from typing import Tuple, List

###############################
# 1. 基础：Struct-+1 单 qudit 门
###############################

def struct_plus_one_matrix() -> np.ndarray:
    """
    Struct-+1 on a 4-level system (qudit):
        |0> -> |1>
        |1> -> |2>
        |2> -> |3>
        |3> -> |0>
    我们用 2 个 qubit 编码 4 个基态：
        0 -> |00>, 1 -> |01>, 2 -> |10>, 3 -> |11>
    """
    U = np.zeros((4, 4), dtype=complex)
    for s in range(4):
        U[(s + 1) % 4, s] = 1.0
    return U

def is_unitary(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """检查矩阵是否为酉矩阵"""
    n = matrix.shape[0]
    return np.allclose(matrix @ matrix.conj().T, np.eye(n), atol=tol)

###############################
# 2. 高效的受控门实现
###############################

def encode_state(c0: int, c1: int, t0: int, t1: int) -> int:
    """
    将4-qubit状态编码为索引
    约定: |c0 c1 t0 t1> = |wire0 wire1 wire2 wire3>
    PennyLane使用小端序: wire0是最低位
    所以索引 = c0 * 2^0 + c1 * 2^1 + t0 * 2^2 + t1 * 2^3
    """
    return (c0 << 0) | (c1 << 1) | (t0 << 2) | (t1 << 3)

def decode_index(idx: int) -> Tuple[int, int, int, int]:
    """将索引解码为qubit状态"""
    c0 = (idx >> 0) & 1  # wire0
    c1 = (idx >> 1) & 1  # wire1
    t0 = (idx >> 2) & 1  # wire2
    t1 = (idx >> 3) & 1  # wire3
    return c0, c1, t0, t1

def qudit_to_qubits(s: int) -> Tuple[int, int]:
    """将qudit状态转换为两个qubit状态"""
    return ((s >> 1) & 1, s & 1)

def qubits_to_qudit(q0: int, q1: int) -> int:
    """将两个qubit状态转换为qudit状态"""
    return (q0 << 1) | q1

def controlled_struct_plus_one_matrix_corrected() -> np.ndarray:
    """
    修正版的受控 Struct-+1 门
    - 总系统: 4 qubits = 2 qudits
    - 控制qudit: wires [0,1] (wire0, wire1)
    - 目标qudit: wires [2,3] (wire2, wire3)
    - 当控制qudit处于逻辑值1时，对目标qudit施加Struct-+1
    """
    n_total = 16  # 4 qubits
    U = np.eye(n_total, dtype=complex)
    
    # Struct-+1门
    U_s = struct_plus_one_matrix()
    
    # 构建受控门
    for idx in range(n_total):
        c0, c1, t0, t1 = decode_index(idx)
        
        # 提取控制qudit状态
        control_qudit = qubits_to_qudit(c0, c1)
        
        # 如果控制qudit = 1 (二进制01)
        if control_qudit == 1:
            # 提取目标qudit状态
            target_qudit = qubits_to_qudit(t0, t1)
            
            # 应用Struct-+1
            new_target_qudit = (target_qudit + 1) % 4
            new_t0, new_t1 = qudit_to_qubits(new_target_qudit)
            
            # 计算新的索引
            new_idx = encode_state(c0, c1, new_t0, new_t1)
            
            # 在酉矩阵中设置
            U[new_idx, idx] = 1.0
            U[idx, idx] = 0.0  # 清空原位置
    
    return U

###############################
# 3. 量子态操作和测量
###############################

def prepare_initial_state() -> np.ndarray:
    """
    准备初始态: (|0> + |1>)_control / sqrt(2) ⊗ |0>_target
    编码:
        control qudit: wires [0,1]
        target qudit: wires [2,3]
    """
    psi = np.zeros(16, dtype=complex)
    
    # |0>_control ⊗ |0>_target
    idx00 = encode_state(0, 0, 0, 0)  # |0000>
    
    # |1>_control ⊗ |0>_target
    # control qudit=1 对应 qubits |01> = wire0=1, wire1=0
    idx10 = encode_state(1, 0, 0, 0)  # |1000>? 需要验证!
    
    # 修正: control qudit=1 应该是 |01> = wire0=1, wire1=0
    # 但根据小端序，wire0是最低位，所以应该是:
    # 状态 |wire3 wire2 wire1 wire0> = |t1 t0 c1 c0>
    # control qudit=1 -> c0=1, c1=0
    # 所以正确的索引:
    idx00 = encode_state(1, 0, 0, 0)  # c0=1, c1=0, t0=0, t1=0 -> 二进制 0001?
    idx10 = encode_state(0, 1, 0, 0)  # c0=0, c1=1, t0=0, t1=0 -> 二进制 0010?
    
    # 让我重新检查编码...
    print("\n=== 验证编码 ===")
    print(f"encode_state(1,0,0,0) = {encode_state(1,0,0,0)} -> {decode_index(encode_state(1,0,0,0))}")
    print(f"encode_state(0,1,0,0) = {encode_state(0,1,0,0)} -> {decode_index(encode_state(0,1,0,0))}")
    
    # 实际上，我们需要明确约定
    # 让我们使用更清晰的编码
    
    return psi

def apply_phase_noise(state: np.ndarray, thetas: List[float]) -> np.ndarray:
    """
    对4个qubit分别施加RZ相位噪声
    RZ(theta) = [[exp(-i*theta/2), 0], [0, exp(i*theta/2)]]
    """
    noisy_state = state.copy()
    
    # 对每个qubit应用RZ
    for qubit in range(4):
        # 构建RZ对角矩阵
        rz_diag = np.array([np.exp(-1j * thetas[qubit] / 2), 
                           np.exp(1j * thetas[qubit] / 2)])
        
        # 将RZ应用到对应qubit
        n = 16
        rz_full = np.ones(n, dtype=complex)
        
        # 遍历所有基态，应用相位
        for idx in range(n):
            bits = decode_index(idx)
            qubit_value = bits[qubit]  # 0或1
            rz_full[idx] = rz_diag[qubit_value]
        
        noisy_state *= rz_full
    
    # 归一化
    noisy_state /= np.linalg.norm(noisy_state)
    return noisy_state

###############################
# 4. 密度矩阵和纠缠度量
###############################

def state_to_density_matrix(state: np.ndarray) -> np.ndarray:
    """纯态转密度矩阵"""
    return np.outer(state, state.conj())

def partial_trace(rho_total: np.ndarray, keep_qubits: List[int]) -> np.ndarray:
    """
    对4-qubit系统做偏迹
    keep_qubits: 保留的qubit索引
    """
    n_total = 4
    n_keep = len(keep_qubits)
    n_trace = n_total - n_keep
    
    # 创建映射
    dim = 2**n_total
    dim_keep = 2**n_keep
    dim_trace = 2**n_trace
    
    # 重排密度矩阵
    rho_reshaped = rho_total.reshape([2,2,2,2,2,2,2,2])
    
    # 计算偏迹
    axes_to_trace = [i for i in range(n_total) if i not in keep_qubits]
    axes_to_keep = [i for i in range(n_total) if i in keep_qubits]
    
    # 更简单的方法：直接计算
    rho_reduced = np.zeros((dim_keep, dim_keep), dtype=complex)
    
    for i in range(dim_keep):
        for j in range(dim_keep):
            # 需要将i,j映射回完整索引
            total_sum = 0
            for k in range(dim_trace):
                # 计算完整索引
                idx_i = 0
                idx_j = 0
                
                # 这个计算比较复杂，我们换一种方法
                pass
    
    # 简化：使用向量方法
    return partial_trace_simple(rho_total, keep_qubits)

def partial_trace_simple(rho: np.ndarray, keep_qubits: List[int]) -> np.ndarray:
    """简化的偏迹计算"""
    n_total = 4
    n_keep = len(keep_qubits)
    dim_keep = 2 ** n_keep
    
    # 构建qubit到位置的映射
    rho_keep = np.zeros((dim_keep, dim_keep), dtype=complex)
    
    # 遍历所有基态
    for i_full in range(16):
        for j_full in range(16):
            # 提取保留qubit的值
            i_bits = decode_index(i_full)
            j_bits = decode_index(j_full)
            
            # 检查被trace掉的qubit是否相等
            trace_ok = True
            for q in range(n_total):
                if q not in keep_qubits:
                    if i_bits[q] != j_bits[q]:
                        trace_ok = False
                        break
            
            if trace_ok:
                # 计算约化密度矩阵的索引
                i_keep = 0
                j_keep = 0
                pos = 0
                for q in keep_qubits:
                    i_keep |= (i_bits[q] << pos)
                    j_keep |= (j_bits[q] << pos)
                    pos += 1
                
                rho_keep[i_keep, j_keep] += rho[i_full, j_full]
    
    return rho_keep

def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    """计算冯·诺依曼熵 (以2为底，单位bits)"""
    # 确保密度矩阵是厄米的
    rho = (rho + rho.conj().T) / 2
    
    # 计算特征值
    eigvals = la.eigvalsh(rho)
    eigvals = np.real(eigvals)  # 确保是实数
    eigvals = np.clip(eigvals, 0.0, 1.0)
    
    # 计算熵
    entropy = 0.0
    for lam in eigvals:
        if lam > eps:
            entropy -= lam * np.log2(lam)
    
    return entropy

def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """计算两个纯态之间的保真度"""
    overlap = np.vdot(state1, state2)
    return np.abs(overlap) ** 2

###############################
# 5. 主测试函数
###############################

def test_basic_gates():
    print("=== 测试基本门 ===")
    
    # 测试Struct-+1门
    U_s = struct_plus_one_matrix()
    print(f"Struct-+1矩阵:\n{U_s}")
    print(f"是否为酉矩阵: {is_unitary(U_s)}")
    
    # 测试受控门
    U_c = controlled_struct_plus_one_matrix_corrected()
    print(f"\n受控Struct-+1矩阵维度: {U_c.shape}")
    print(f"是否为酉矩阵: {is_unitary(U_c)}")
    
    # 验证受控逻辑
    print("\n验证受控逻辑:")
    test_states = [
        (0, 0, 0, 0),  # control=0, target=0
        (1, 0, 0, 0),  # control=1, target=0
        (0, 1, 2, 0),  # control=2, target=2
        (1, 0, 3, 0),  # control=1, target=3
    ]
    
    for c0, c1, t_qudit, dummy in test_states:
        t0, t1 = qudit_to_qubits(t_qudit)
        idx_in = encode_state(c0, c1, t0, t1)
        
        # 应用受控门
        state_in = np.zeros(16, dtype=complex)
        state_in[idx_in] = 1.0
        state_out = U_c @ state_in
        
        # 找到输出态
        idx_out = np.argmax(np.abs(state_out))
        c0_out, c1_out, t0_out, t1_out = decode_index(idx_out)
        control_out = qubits_to_qudit(c0_out, c1_out)
        target_out = qubits_to_qudit(t0_out, t1_out)
        
        control_in = qubits_to_qudit(c0, c1)
        
        print(f"输入: control={control_in}, target={t_qudit}")
        print(f"输出: control={control_out}, target={target_out}")
        
        # 验证逻辑
        if control_in == 1:
            expected = (t_qudit + 1) % 4
            correct = (target_out == expected)
        else:
            correct = (target_out == t_qudit)
        
        print(f"逻辑正确: {'✓' if correct else '✗'}\n")

def test_entanglement_and_noise():
    print("\n=== 测试纠缠和噪声鲁棒性 ===")
    
    # 构建受控门
    U_c = controlled_struct_plus_one_matrix_corrected()
    
    # 准备初始态: (|0> + |1>)_control / sqrt(2) ⊗ |0>_target
    psi0 = np.zeros(16, dtype=complex)
    
    # control=0, target=0
    idx0 = encode_state(1, 0, 0, 0)  # c0=1, c1=0, t0=0, t1=0 -> control qudit=1? 需要修正!
    
    # 让我们重新仔细定义
    print("\n明确编码约定:")
    print("control qudit 状态: |c1 c0> (wire1 wire0)")
    print("  qudit=0 -> |00> = wire0=0, wire1=0")
    print("  qudit=1 -> |01> = wire0=1, wire1=0")
    print("  qudit=2 -> |10> = wire0=0, wire1=1")
    print("  qudit=3 -> |11> = wire0=1, wire1=1")
    
    # 修正encode_state函数
    def encode_state_corrected(c_qudit: int, t_qudit: int) -> int:
        """正确的编码: c_qudit, t_qudit -> 索引"""
        c0, c1 = qudit_to_qubits(c_qudit)  # c0=LSB, c1=MSB
        t0, t1 = qudit_to_qubits(t_qudit)  # t0=LSB, t1=MSB
        
        # PennyLane小端序: wire0是最低位
        # |wire3 wire2 wire1 wire0> = |t1 t0 c1 c0>
        return (c0 << 0) | (c1 << 1) | (t0 << 2) | (t1 << 3)
    
    # 测试新的编码
    print(f"\n测试编码:")
    for c in range(4):
        for t in range(4):
            idx = encode_state_corrected(c, t)
            print(f"control={c}, target={t} -> idx={idx:04b}")
    
    # 准备正确的初始态
    psi0 = np.zeros(16, dtype=complex)
    
    # control=0, target=0
    idx00 = encode_state_corrected(0, 0)
    
    # control=1, target=0
    idx10 = encode_state_corrected(1, 0)
    
    psi0[idx00] = 1.0 / np.sqrt(2)
    psi0[idx10] = 1.0 / np.sqrt(2)
    
    # 应用受控门
    psi_ent = U_c @ psi0
    
    # 计算密度矩阵
    rho_total = state_to_density_matrix(psi_ent)
    
    # 计算约化密度矩阵
    # control qudit: wires [0,1] -> qubits 0,1
    rho_control = partial_trace_simple(rho_total, [0, 1])
    
    # target qudit: wires [2,3] -> qubits 2,3
    rho_target = partial_trace_simple(rho_total, [2, 3])
    
    # 计算纠缠熵
    S_control = von_neumann_entropy(rho_control)
    S_target = von_neumann_entropy(rho_target)
    
    print(f"\n纠缠熵:")
    print(f"S(control) = {S_control:.6f} bits")
    print(f"S(target)  = {S_target:.6f} bits")
    print(f"纠缠存在: {'✓' if S_control > 0.01 else '✗'} (非零熵)")
    
    # 测试相位噪声鲁棒性
    print("\n=== 相位噪声鲁棒性测试 ===")
    
    sigmas = [0.0, 0.02, 0.05, 0.1, 0.2]
    np.random.seed(1234)
    
    for sigma in sigmas:
        fidelities = []
        for _ in range(50):
            # 生成随机相位噪声
            thetas = np.random.normal(0, sigma, 4)
            
            # 应用噪声
            psi_noisy = apply_phase_noise(psi_ent, thetas)
            
            # 计算保真度
            f = fidelity(psi_ent, psi_noisy)
            fidelities.append(f)
        
        f_mean = np.mean(fidelities)
        f_std = np.std(fidelities)
        print(f"σ={sigma:.4f} rad: 平均保真度 = {f_mean:.4f} ± {f_std:.4f}")
        
        # 特别关注0.2 rad的情况
        if abs(sigma - 0.2) < 1e-6:
            print(f"  论文宣称: 98.4%")
            print(f"  实际得到: {f_mean*100:.2f}%")
            print(f"  差异: {abs(0.984 - f_mean)*100:.2f}%")

###############################
# 6. 修复后的主函数
###############################

if __name__ == "__main__":
    print("量子形态计算架构 - 代码验证")
    print("=" * 50)
    
    # 测试基本门
    test_basic_gates()
    
    # 测试纠缠和噪声鲁棒性
    test_entanglement_and_noise()
    
    print("\n" + "=" * 50)
    print("验证完成!")
