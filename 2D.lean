import Mathlib.Data.Real.Basic

/-!
Stage 2: 从 2D 曼哈顿晶格到 1D 壳层模型的形式化连接

目标：如果在 2D 晶格上有
  |ψ(x,y)| ≤ C * r^(x+y)
则在任意给定的壳层 n（即 x+y = n）上，都有
  |ψ(x,y)| ≤ C * r^n

这一步把“曼哈顿距离 d=x+y” 和 “壳层编号 n” 严格对齐。
-/

/-- 2D 曼哈顿晶格上的波函数剖面。

- `L` : 晶格线性尺寸（0 ≤ x,y < L）
- `psi x y` : 点 (x,y) 处的（实）振幅
  （如果你以后想用复数，可以把 `ℝ` 换成 `ℂ`）
-/
structure ManhattanProfile where
  L   : ℕ
  psi : ℕ → ℕ → ℝ

namespace ManhattanProfile

/-- 曼哈顿距离：d(x,y) = x + y -/
def manhattan (x y : ℕ) : ℕ := x + y

/-- 
假设：在整个晶格上存在全局指数衰减界
  |ψ(x,y)| ≤ C * r^(x+y)
那么在任意“壳层” n（即 x+y = n）上，可以把指数直接写成 n。
这对应物理上的：“壳层按曼哈顿距离编号是自洽的”。
-/
lemma amplitude_bound_on_shell
  (P : ManhattanProfile)
  (C r : ℝ)
  (hdecay : ∀ x y, |P.psi x y| ≤ C * r^(x + y)) :
  ∀ {n x y}, x < P.L → y < P.L → x + y = n →
    |P.psi x y| ≤ C * r^n := by
  intro n x y hx hy hxy
  -- 直接用全局指数界，把 x+y 换成 n 即可
  have h := hdecay x y
  simpa [manhattan, hxy] using h

/-- 
从 2D 曼哈顿剖面诱导出的“1D 壳层剖面”。

在阶段 2 我们先不去取真正的 max 或 sum，
而是把理论上的壳层上界写成一个显式函数：
  shell(n) = C * r^n

在后续阶段（面积律、求和等）可以用这个 `ShellProfile`
作为 1D 抽象模型继续推。
-/
structure ShellProfile where
  shell : ℕ → ℝ

/--
给定一个 2D 剖面 `P`，以及理论上从局域约束得到的
指数衰减常数 `C, r`，我们可以构造一个“理想壳层模型”：
  shell(n) = C * r^n
注意：这里并没有声称 `shell(n)` 真的是
  max_{x+y=n} |ψ(x,y)|，
只是给出了一个在所有 (x,y) 上都成立的上界函数。
-/
def inducedShell (P : ManhattanProfile) (C r : ℝ) : ShellProfile :=
  { shell := fun n => C * r^n }

/--
把上界形式化成一句话：如果全局有
  |ψ(x,y)| ≤ C * r^(x+y)，
那么对所有满足 x+y=n 的点都有
  |ψ(x,y)| ≤ inducedShell(P,C,r).shell n = C * r^n。
-/
lemma abs_psi_le_shell
  (P : ManhattanProfile)
  (C r : ℝ)
  (hdecay : ∀ x y, |P.psi x y| ≤ C * r^(x + y)) :
  ∀ {n x y}, x < P.L → y < P.L → x + y = n →
    |P.psi x y| ≤ (inducedShell P C r).shell n := by
  intro n x y hx hy hxy
  -- 应用刚刚证明的 amplitude_bound_on_shell 即可
  have h := amplitude_bound_on_shell P C r hdecay hx hy hxy
  simpa [inducedShell] using h

end ManhattanProfile
