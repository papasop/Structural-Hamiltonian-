import Mathlib.Data.Real.Basic

noncomputable section

/-!
# Non-Hermitian Area-Law: Abstract Route C Formalization

This file encodes the purely algebraic core of your “Route C” argument:

* Level 1 (`ShellProfile`):
  - We have a leakage rate `Γ(L) = leakage L ≥ 0`.
  - We have a geometric upper bound `geomBound L > 0`.
  - And we assume `leakage L ≤ geomBound L`.
  - Lifetime is defined as `τ(L) = 1 / leakage(L)` (with τ(L)=0 if leakage(L)=0).
  - We prove: if `leakage(L) > 0`, then `τ(L) ≥ 1 / geomBound(L)`.

* Level 2 (`ExpGeomProfile`):
  - We further assume an area-law-type bound
      `geomBound L ≤ A * α^(L^2)` for all `L ≥ L0`,
      with `A > 0` and `0 < α < 1`.
  - Then we prove an exponential lower bound on the lifetime:
      `τ(L) ≥ 1 / (A * α^(L^2))`.

This is exactly the abstract mathematical backbone behind the
“area-law lifetime scaling” in your non-Hermitian corner gain model.
-/

namespace NonHermitianAreaLaw

/-- 
An abstract “shell profile” for a family of systems labeled by size `L : ℕ`.

* `leakage L`    : leakage rate Γ(L) from the attractor at system size `L`;
* `geomBound L`  : geometric upper bound on this leakage;
* `leakage_nonneg  : ∀ L, 0 ≤ leakage L`;
* `geomBound_pos   : ∀ L, 0 < geomBound L`;
* `leakage_le_geom : ∀ L, leakage L ≤ geomBound L`.
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
Lifetime at size `L`, defined as τ(L) = 1 / Γ(L).

To avoid division by zero, we adopt the convention
`lifetime L = 0` when `leakage L = 0`.
This is enough for inequality reasoning.
-/
def lifetime (L : ℕ) : ℝ :=
  if h : P.leakage L = 0 then 0 else 1 / P.leakage L

/-- Lifetime is always nonnegative: τ(L) ≥ 0. -/
lemma lifetime_nonneg (L : ℕ) : 0 ≤ P.lifetime L := by
  unfold ShellProfile.lifetime
  split_ifs with h
  · -- case leakage(L)=0
    simp [h]
  · -- case leakage(L) ≠ 0
    have hnn : 0 ≤ P.leakage L := P.leakage_nonneg L
    have hne : P.leakage L ≠ 0 := h
    have hpos : 0 < P.leakage L := lt_of_le_of_ne hnn (Ne.symm hne)
    have h_inv : 0 ≤ (P.leakage L)⁻¹ :=
      inv_nonneg.mpr (le_of_lt hpos)
    simpa [h, one_div] using h_inv

/--
**Route C, Level 1.**

If the leakage rate is strictly positive and bounded above by a geometric
bound, then the lifetime is bounded below by the reciprocal of that bound:

Assumptions:
* `0 < leakage L`,
* `leakage L ≤ geomBound L`.

Conclusion:
\[
  τ(L) \;\ge\; \frac{1}{\mathrm{geomBound}(L)}.
\]
-/
lemma lifetime_ge_inv_geomBound
    (L : ℕ) (h_pos : 0 < P.leakage L) :
    1 / P.geomBound L ≤ P.lifetime L := by
  -- Geometric bound: leakage(L) ≤ geomBound(L).
  have h_le : P.leakage L ≤ P.geomBound L := P.leakage_le_geom L
  -- From 0 < leakage ≤ geomBound, monotonicity of 1/x gives:
  --   1 / geomBound ≤ 1 / leakage.
  have h_div : 1 / P.geomBound L ≤ 1 / P.leakage L := by
    have := one_div_le_one_div_of_le h_pos h_le
    simpa [one_div] using this
  -- By definition, τ(L) = 1 / leakage(L) when leakage(L) ≠ 0.
  unfold ShellProfile.lifetime
  have h_ne : P.leakage L ≠ 0 := ne_of_gt h_pos
  simpa [h_ne] using h_div

end ShellProfile

/-- 
An “exponentially bounded” shell profile, encoding an area-law-type
upper bound on the leakage.

In addition to `ShellProfile`, we assume:

* A constant `A > 0`,
* A constant `α` with `0 < α < 1`,
* A threshold size `L0`,
* And for all `L ≥ L0`:
    `geomBound L ≤ A * α^(L^2)`.

This is the abstract form of an “area-law geometric leakage bound”.
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
**Route C, Level 2.  Exponential geometric bound ⇒ exponential lifetime lower bound.**

Assume:
* `geomBound L ≤ A * α^(L^2)` for all `L ≥ L0`,
* `0 < α < 1`,
* `0 < A`,
* and a strictly positive leakage `0 < leakage L`.

Then for all `L ≥ L0` we have

\[
  τ(L) \;\ge\; \frac{1}{A\,α^{L^2}}.
\]

This is the abstract, mathematically precise statement of an
“area-law exponential lower bound” on the lifetime.
-/
lemma lifetime_ge_exp
    (L : ℕ) (hL : L ≥ P.L0) (h_leak_pos : 0 < P.leakage L) :
    1 / (P.A * P.α^(L^2))
      ≤ ShellProfile.lifetime P.toShellProfile L := by
  -- Step 1: From geomBound(L) ≤ A α^(L²), with geomBound(L) > 0,
  --         we get 1 / (A α^(L²)) ≤ 1 / geomBound(L).
  have h_geom_pos : 0 < P.geomBound L := P.geomBound_pos L
  have h_geom_le_exp : P.geomBound L ≤ P.A * P.α^(L^2) :=
    P.geom_le_exp (L := L) hL
  have h_one_div :
      1 / (P.A * P.α^(L^2)) ≤ 1 / P.geomBound L := by
    have := one_div_le_one_div_of_le h_geom_pos h_geom_le_exp
    simpa [one_div] using this

  -- Step 2: From ShellProfile lemma:
  --         1 / geomBound(L) ≤ lifetime(L).
  have h_life :
      1 / P.geomBound L ≤
        ShellProfile.lifetime P.toShellProfile L :=
    ShellProfile.lifetime_ge_inv_geomBound
      (P := P.toShellProfile) L h_leak_pos

  -- Step 3: Chain the inequalities.
  exact le_trans h_one_div h_life

/-
Remark (for the paper, not formalized here):

From the Lean-proved inequality
  τ(L) ≥ 1 / (A * α^(L^2))
with 0 < α < 1, one immediately obtains, at the level of real analysis,

  log τ(L) ≥ -log A + (log(1/α)) · L^2.

Thus there exist constants c > 0 and b ∈ ℝ such that
  log τ(L) ≥ c · L^2 + b
for all sufficiently large L.

We do not formalize Real.log here to keep the dependency minimal;
this final logarithmic rewriting is left to the mathematical text.
-/

end ExpGeomProfile

end NonHermitianAreaLaw



