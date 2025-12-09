import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Basic

noncomputable section

namespace NonHermitianAreaLaw

/-- 
Level 1：抽象壳层剖面（只看系统尺寸 L 的泄漏率与几何上界）。
-/
structure ShellProfile where
  leakage    : ℕ → ℝ
  geomBound  : ℕ → ℝ
  leakage_nonneg  : ∀ L, 0 ≤ leakage L
  geomBound_pos   : ∀ L, 0 < geomBound L
  leakage_le_geom : ∀ L, leakage L ≤ geomBound L

namespace ShellProfile

variable (P : ShellProfile)

/-- 寿命 τ(L) := 1 / leakage(L)，若 leakage(L)=0 则定义为 0。 -/
def lifetime (L : ℕ) : ℝ :=
  if h : P.leakage L = 0 then 0 else 1 / P.leakage L

/-- 寿命总是非负。 -/
lemma lifetime_nonneg (L : ℕ) : 0 ≤ P.lifetime L := by
  unfold lifetime
  split_ifs with h
  · -- leakage L = 0
    simp [h]
  · -- leakage L ≠ 0
    have hpos : 0 < P.leakage L := by
      have hnn : 0 ≤ P.leakage L := P.leakage_nonneg L
      have hne : P.leakage L ≠ 0 := h
      exact lt_of_le_of_ne hnn (Ne.symm hne)
    have h_inv : 0 ≤ (P.leakage L)⁻¹ :=
      inv_nonneg.mpr (le_of_lt hpos)
    simpa [one_div, h] using h_inv

/--
核心 Route C（Level 1）定理：

若 `0 < leakage(L)`，则
  τ(L) ≥ 1 / geomBound(L)。
-/
lemma lifetime_ge_inv_geomBound
    (L : ℕ)
    (h_pos : 0 < P.leakage L) :
    1 / P.geomBound L ≤ P.lifetime L := by
  -- leakage(L) ≠ 0
  have h_leak_ne : P.leakage L ≠ 0 := ne_of_gt h_pos
  -- 从 Γ(L) ≤ Γ_geom(L) 推出 1/Γ_geom(L) ≤ 1/Γ(L)
  have h_le : P.leakage L ≤ P.geomBound L := P.leakage_le_geom L
  have h_div : 1 / P.geomBound L ≤ 1 / P.leakage L := by
    -- 这里 a = leakage(L), b = geomBound(L)
    have := one_div_le_one_div_of_le h_pos h_le
    simpa [one_div] using this
  -- 展开 lifetime，利用 leakage ≠ 0
  unfold lifetime
  simpa [h_leak_ne, one_div] using h_div

end ShellProfile

end NonHermitianAreaLaw
