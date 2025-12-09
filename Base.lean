import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Basic

/-
ShellProfile：一维“壳层振幅”抽象模型

* psi n           : 第 n 层的振幅（实数）
* C, r            : 几何衰减常数
* hC_nonneg       : C ≥ 0
* hr_pow_nonneg   : ∀ n, r^n ≥ 0
* hgeom           : ∀ n, |psi n| ≤ C * r^n
-/
structure ShellProfile where
  psi           : Nat → ℝ
  C             : ℝ
  r             : ℝ
  hC_nonneg     : 0 ≤ C
  hr_pow_nonneg : ∀ n : Nat, 0 ≤ r ^ n
  hgeom         : ∀ n : Nat, |psi n| ≤ C * r ^ n

namespace ShellProfile

/-- 泄漏：P.leakage L = ∑_{n=0}^{L-1} (psi n)²（递归定义） -/
def leakage (P : ShellProfile) : Nat → ℝ
  | 0     => 0
  | n + 1 => P.leakage n + (P.psi n) ^ 2

/-- 几何上界：P.geomBound L = ∑_{n=0}^{L-1} (C r^n)²（递归定义） -/
def geomBound (P : ShellProfile) : Nat → ℝ
  | 0     => 0
  | n + 1 => P.geomBound n + (P.C * P.r ^ n) ^ 2

/--
核心引理：如果对所有 n 有 |psi n| ≤ C r^n，
则对所有 L，有 P.leakage L ≤ P.geomBound L。
-/
lemma leakage_le_geomBound (P : ShellProfile) :
    ∀ L : Nat, P.leakage L ≤ P.geomBound L := by
  intro L
  -- 对 L 归纳
  induction L with
  | zero =>
      -- L = 0
      simp [ShellProfile.leakage, ShellProfile.geomBound]
  | succ n ih =>
      -- 归纳步：假设 ih : P.leakage n ≤ P.geomBound n
      -- 目标：P.leakage (n+1) ≤ P.geomBound (n+1)
      -- 先把递归展开
      simp [ShellProfile.leakage, ShellProfile.geomBound] at ih ⊢

      -- 物理输入：局域几何约束 |psi n| ≤ C * r^n
      have h_geom : |P.psi n| ≤ P.C * P.r ^ n := P.hgeom n

      -- r^n ≥ 0
      have hrn_nonneg : 0 ≤ P.r ^ n := P.hr_pow_nonneg n

      -- C * r^n ≥ 0
      have hCrn_nonneg : 0 ≤ P.C * P.r ^ n := by
        have := mul_nonneg P.hC_nonneg hrn_nonneg
        simpa using this

      -- 因为 C r^n ≥ 0，所以 |C r^n| = C r^n
      have h_abs_rhs : |P.C * P.r ^ n| = P.C * P.r ^ n := by
        simpa [abs_of_nonneg hCrn_nonneg]

      -- 把右边也写成绝对值形式
      have h_abs_ineq : |P.psi n| ≤ |P.C * P.r ^ n| := by
        simpa [h_abs_rhs] using h_geom

      -- 利用 sq_le_sq：|a| ≤ |b| ⇒ a² ≤ b²
      have h_sq : (P.psi n) ^ 2 ≤ (P.C * P.r ^ n) ^ 2 := by
        have := sq_le_sq.mpr h_abs_ineq
        -- 在 ℝ 上 pow_two = (^2)
        simpa [pow_two] using this

      -- 归纳假设 ih：P.leakage n ≤ P.geomBound n
      -- 单点不等式 h_sq： (psi n)² ≤ (C r^n)²
      -- 相加得到：
      have h_add :
          P.leakage n + (P.psi n) ^ 2
        ≤ P.geomBound n + (P.C * P.r ^ n) ^ 2 :=
        add_le_add ih h_sq

      exact h_add

end ShellProfile

