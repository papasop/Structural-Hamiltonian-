import Mathlib.Data.Real.Basic

noncomputable section

namespace NonHermitianAreaLaw

/-- 
一个抽象的“壳层泄漏”模型：

* `leakage L`    : 系统尺寸为 `L` 时的泄漏率 Γ(L)
* `geomBound L`  : 来自几何/微扰估计的泄漏上界
* `leakage_nonneg`  : Γ(L) ≥ 0
* `geomBound_pos`   : 几何上界严格为正
* `leakage_le_geom` : Γ(L) ≤ 几何上界
-/
structure ShellProfile where
  leakage    : ℕ → ℝ
  geomBound  : ℕ → ℝ
  leakage_nonneg  : ∀ L, 0 ≤ leakage L
  geomBound_pos   : ∀ L, 0 < geomBound L
  leakage_le_geom : ∀ L, leakage L ≤ geomBound L

namespace ShellProfile

variable (P : ShellProfile)

/-- 
寿命定义：`τ(L) = 1 / leakage(L)`；  
为避免除以 0，当 `leakage(L) = 0` 时约定 τ(L)=0。
-/
def lifetime (L : ℕ) : ℝ :=
  if h : P.leakage L = 0 then 0 else 1 / P.leakage L

/-- 寿命总是非负：τ(L) ≥ 0。 -/
lemma lifetime_nonneg (L : ℕ) : 0 ≤ P.lifetime L := by
  unfold ShellProfile.lifetime
  split_ifs with h
  · -- 情形 leakage(L)=0
    simp [h]
  · -- 情形 leakage(L) ≠ 0
    have hnn : 0 ≤ P.leakage L := P.leakage_nonneg L
    have hne : P.leakage L ≠ 0 := h
    have hpos : 0 < P.leakage L := lt_of_le_of_ne hnn (Ne.symm hne)
    have h_inv : 0 ≤ (P.leakage L)⁻¹ :=
      inv_nonneg.mpr (le_of_lt hpos)
    simpa [h, one_div] using h_inv

/--
**Route C, Level 1：几何泄漏上界 ⇒ 寿命下界**

假设某个尺寸 `L` 下泄漏率满足 `leakage(L) > 0`，并且  
`leakage(L) ≤ geomBound(L)`，则有  

\[
  \tau(L) \;\ge\; \frac{1}{\mathrm{geomBound}(L)}.
\]

这是从 Γ ≤ Γ\_geom 推出 τ ≥ 1/Γ\_geom 的形式化版本。
-/
lemma lifetime_ge_inv_geomBound
    (L : ℕ) (h_pos : 0 < P.leakage L) :
    1 / P.geomBound L ≤ P.lifetime L := by
  -- 1) 几何上界：leakage(L) ≤ geomBound(L)
  have h_le : P.leakage L ≤ P.geomBound L := P.leakage_le_geom L
  -- 2) 由 0<leakage≤geomBound，倒数单调性：1/geomBound ≤ 1/leakage
  have h_div : 1 / P.geomBound L ≤ 1 / P.leakage L := by
    have := one_div_le_one_div_of_le h_pos h_le
    simpa [one_div] using this
  -- 3) 用寿命定义：τ(L) = 1 / leakage(L)
  unfold ShellProfile.lifetime
  have h_ne : P.leakage L ≠ 0 := ne_of_gt h_pos
  simpa [h_ne] using h_div

end ShellProfile

/-- 
带“指数几何上界”的版本：在 `ShellProfile` 基础上再加上面积律几何界。

假设存在常数 `A>0`，`0<α<1`，以及阈值尺寸 `L0`，使得  
对所有 `L ≥ L0` 有

\[
  \mathrm{geomBound}(L) \;\le\; A \, α^{L^2}.
\]

这是你物理解读里的“面积律几何泄漏上界”。
-/
structure ExpGeomProfile extends ShellProfile where
  A      : ℝ
  α      : ℝ
  L0     : ℕ
  hA_pos : 0 < A
  hα_pos : 0 < α
  hα_lt_one : α < 1
  geom_le_exp : ∀ {L : ℕ}, L ≥ L0 → geomBound L ≤ A * α^(L^2)

namespace ExpGeomProfile

variable (P : ExpGeomProfile)

/--
**Route C, Level 2：指数几何上界 ⇒ 指数寿命下界**

若对所有 `L ≥ L0` 都有
\[
  \mathrm{geomBound}(L) \;\le\; A \, α^{L^2},
\]
并且该尺寸下泄漏率满足 `leakage(L) > 0`，  
则对所有 `L ≥ L0` 有

\[
  \tau(L) \;\ge\; \frac{1}{A\,α^{L^2}}.
\]

这就是“面积律寿命下界”的抽象数学表述：
寿命至少是 \(A^{-1} α^{-L^2}\) 量级（α<1）。
-/
lemma lifetime_ge_exp
    (L : ℕ) (hL : L ≥ P.L0) (h_leak_pos : 0 < P.leakage L) :
    1 / (P.A * P.α^(L^2))
      ≤ ShellProfile.lifetime P.toShellProfile L := by
  -- Step 1：先把几何上界倒数化：1/(A α^{L²}) ≤ 1/geomBound(L)
  have h_geom_pos : 0 < P.geomBound L := P.geomBound_pos L
  have h_geom_le_exp : P.geomBound L ≤ P.A * P.α^(L^2) :=
    P.geom_le_exp (L := L) hL
  have h_one_div :
      1 / (P.A * P.α^(L^2)) ≤ 1 / P.geomBound L := by
    -- 注意这里 a = geomBound(L), b = A α^{L²}：
    -- 0 < a, a ≤ b ⇒ 1/b ≤ 1/a
    have := one_div_le_one_div_of_le h_geom_pos h_geom_le_exp
    simpa [one_div] using this

  -- Step 2：再用 ShellProfile 的引理：1/geomBound(L) ≤ τ(L)
  have h_life :
      1 / P.geomBound L ≤
        ShellProfile.lifetime P.toShellProfile L :=
    ShellProfile.lifetime_ge_inv_geomBound
      (P := P.toShellProfile) L h_leak_pos

  -- Step 3：串起来
  exact le_trans h_one_div h_life

end ExpGeomProfile

end NonHermitianAreaLaw


