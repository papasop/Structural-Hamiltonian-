import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Basic

noncomputable section

/-
Route C：从几何泄漏上界，形式化推出寿命的下界。

ShellProfile:
  * leakage L    : 泄漏率 Γ(L)
  * geomBound L  : 几何上界 G(L)
  * leakage_nonneg  : Γ(L) ≥ 0
  * geomBound_pos   : G(L) > 0
  * leakage_le_geom : Γ(L) ≤ G(L)

ExpGeomProfile 额外假设：
  * G(L) ≤ A * α^(L^2)  （面积律形式），A>0, 0<α<1

核心结论：
  τ(L) = 1 / Γ(L) ≥ 1 / (A * α^(L^2))  （在 Γ(L) > 0 时）
-/

namespace NonHermitianAreaLaw

/-- 抽象的“壳层泄漏模型” -/
structure ShellProfile where
  leakage    : ℕ → ℝ     -- Γ(L)
  geomBound  : ℕ → ℝ     -- G(L)
  leakage_nonneg  : ∀ L, 0 ≤ leakage L
  geomBound_pos   : ∀ L, 0 < geomBound L
  leakage_le_geom : ∀ L, leakage L ≤ geomBound L

namespace ShellProfile

variable (P : ShellProfile)

/--
寿命定义：τ(L) = 1 / Γ(L)；
约定：如果 leakage L = 0，则 τ(L) = 0。
-/
def lifetime (L : ℕ) : ℝ :=
  if h : P.leakage L = 0 then 0 else 1 / P.leakage L

/-- 寿命总是非负。 -/
lemma lifetime_nonneg (L : ℕ) : 0 ≤ P.lifetime L := by
  unfold ShellProfile.lifetime
  split_ifs with h
  · -- case: leakage L = 0
    simp [h]
  · -- case: leakage L ≠ 0
    have hnn : 0 ≤ P.leakage L := P.leakage_nonneg L
    have hne : P.leakage L ≠ 0 := h
    have hpos : 0 < P.leakage L :=
      lt_of_le_of_ne hnn (Ne.symm hne)
    have h_inv_nonneg : 0 ≤ (P.leakage L)⁻¹ :=
      inv_nonneg.mpr (le_of_lt hpos)
    simpa [one_div] using h_inv_nonneg

/--
**Route C，Level 1：**

假设 `leakage(L) > 0` 且 `leakage(L) ≤ geomBound(L)`，
则
  τ(L) ≥ 1 / geomBound(L)。
-/
lemma lifetime_ge_inv_geomBound
    (L : ℕ)
    (h_pos : 0 < P.leakage L) :
    1 / P.geomBound L ≤ P.lifetime L := by
  have h_leak_ne : P.leakage L ≠ 0 := ne_of_gt h_pos
  -- 已知：leakage(L) ≤ geomBound(L)
  have h_le : P.leakage L ≤ P.geomBound L := P.leakage_le_geom L

  -- 使用 1/x 在 (0,+∞) 的单调性：
  -- one_div_le_one_div_of_le : 0 < a → a ≤ b → 1 / b ≤ 1 / a
  -- 这里取 a = leakage(L), b = geomBound(L)
  have h_div : 1 / P.geomBound L ≤ 1 / P.leakage L := by
    have := one_div_le_one_div_of_le h_pos h_le
    simpa [one_div] using this

  -- 展开 lifetime，leakage ≠ 0 时：lifetime = 1 / leakage
  unfold ShellProfile.lifetime
  simpa [h_leak_ne] using h_div

end ShellProfile

/--
带有指数几何上界的壳层模型：

在 ShellProfile 基础上，额外假设常数 A, α, L0 满足
  * A > 0
  * 0 < α < 1
  * ∀ L ≥ L0, geomBound(L) ≤ A * α^(L^2)
-/
structure ExpGeomProfile extends ShellProfile where
  A      : ℝ
  α      : ℝ
  L0     : ℕ
  hA_pos : 0 < A
  hα_pos : 0 < α
  hα_lt_one : α < 1
  geom_le_exp :
    ∀ ⦃L : ℕ⦄, L ≥ L0 → geomBound L ≤ A * α^(L^2)

namespace ExpGeomProfile

variable (P : ExpGeomProfile)

/--
**Route C，Level 2：指数几何上界 ⇒ 寿命指数下界**

若对所有 L ≥ L0 有
  geomBound(L) ≤ A * α^(L^2)，且 0 < α < 1, A > 0，
并且在该 L 有 leakage(L) > 0，
则
  τ(L) ≥ 1 / (A * α^(L^2))。
-/
lemma lifetime_ge_exp
    (L : ℕ)
    (hL : L ≥ P.L0)
    (h_leak_pos : 0 < P.leakage L) :
    1 / (P.A * P.α^(L^2)) ≤
      ShellProfile.lifetime (P.toShellProfile) L := by
  -- 1) geomBound(L) > 0
  have h_geom_pos : 0 < P.geomBound L := P.geomBound_pos L
  -- 2) geomBound(L) ≤ A * α^(L^2)
  have h_geom_le_exp : P.geomBound L ≤ P.A * P.α^(L^2) :=
    P.geom_le_exp (L := L) hL

  -- 3) 由 0 < geom ≤ A α^(L²) 得到 1 / (A α^(L²)) ≤ 1 / geom
  have h_one_div :
      1 / (P.A * P.α^(L^2)) ≤ 1 / P.geomBound L := by
    -- one_div_le_one_div_of_le : 0 < a → a ≤ b → 1 / b ≤ 1 / a
    -- 在这里 a = geomBound(L), b = A * α^(L²)
    have := one_div_le_one_div_of_le h_geom_pos h_geom_le_exp
    simpa [one_div] using this

  -- 4) Route C Level 1：1 / geomBound ≤ lifetime
  have h_life :
      1 / P.geomBound L ≤
        ShellProfile.lifetime P.toShellProfile L :=
    ShellProfile.lifetime_ge_inv_geomBound
      (P := P.toShellProfile) L h_leak_pos

  -- 5) 链接两个不等式
  exact le_trans h_one_div h_life

end ExpGeomProfile

end NonHermitianAreaLaw

