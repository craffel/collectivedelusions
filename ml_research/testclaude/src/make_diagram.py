"""Method diagram for SCALE-Merge.

Emphasizes two things:
  (1) TIES-style preprocessing of task vectors (trim + sign-election), and
  (2) the activation-aware quadratic form Ĉ_t = Δ̃_tᵀ Δ̃_t that enters the
      RegMean/MaTS linear system. This quadratic form is the step that makes
      SCALE *activation-aware* (and data-free), even though, per the §5.2
      diagnostic, it does NOT approximate the data-based C_t well — it routes
      per-task mass into task-relevant columns (routing view).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 5.0))
ax.set_xlim(0, 12); ax.set_ylim(0, 5.0); ax.axis("off")

box_style = dict(boxstyle="round,pad=0.35", linewidth=1.2, alpha=0.95)

# Column x-centers
X = dict(W=0.7, D=2.2, Dhat=3.7, Dtilde=5.3, Cq=7.2, Solve=9.1, Out=10.8)

# ---- Column headers (top row, small annotations) ----
headers = [
    (X["W"],       "Fine-tuned\n$W_t$"),
    (X["D"],       "Task vectors\n$\\Delta_t = W_t{-}W_0$"),
    (X["Dhat"],    "Trim top-$k$%\n(drop redundant)"),
    (X["Dtilde"],  "Elect sign $\\gamma$\n(drop sign-conflicts)"),
    (X["Cq"],      "Quadratic form\n(activation-aware)"),
    (X["Solve"],   "Linear solve\n(closed-form, data-free)"),
    (X["Out"],     "Merged model"),
]
for x, txt in headers:
    ax.text(x, 4.75, txt, ha="center", va="bottom", fontsize=9.6)

# Row y-centers for the T task rows
def row_y(i):
    return 3.5 - i * 0.85  # 3 rows at y = 3.5, 2.65, 1.8

ROW_LBL = [r"$W_1$", r"$W_2$", r"$W_T$"]
TV_LBL = [r"$\Delta_1$", r"$\Delta_2$", r"$\Delta_T$"]
TV_HAT = [r"$\hat\Delta_1$", r"$\hat\Delta_2$", r"$\hat\Delta_T$"]
TV_TILDE = [r"$\tilde\Delta_1$", r"$\tilde\Delta_2$", r"$\tilde\Delta_T$"]

# 1. Fine-tuned weights W_t
for i, name in enumerate(ROW_LBL):
    y = row_y(i)
    ax.add_patch(patches.FancyBboxPatch((X["W"]-0.4, y-0.28), 0.8, 0.56,
                                        edgecolor="#333", facecolor="#e5e5e5", **box_style))
    ax.text(X["W"], y, name, ha="center", va="center", fontsize=11)
ax.text(X["W"], row_y(1)-0.48, r"$\ldots$", ha="center", va="center", fontsize=14)

# 2. Task vectors Δ_t
for i, name in enumerate(TV_LBL):
    y = row_y(i)
    ax.add_patch(patches.FancyBboxPatch((X["D"]-0.4, y-0.28), 0.8, 0.56,
                                        edgecolor="#333", facecolor="#cce5ff", **box_style))
    ax.text(X["D"], y, name, ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(X["D"]-0.45, y), xytext=(X["W"]+0.45, y),
                arrowprops=dict(arrowstyle="->", lw=1.0))

# 3. Trimmed Δ̂_t
for i, name in enumerate(TV_HAT):
    y = row_y(i)
    ax.add_patch(patches.FancyBboxPatch((X["Dhat"]-0.4, y-0.28), 0.8, 0.56,
                                        edgecolor="#333", facecolor="#b3d9ff", **box_style))
    ax.text(X["Dhat"], y, name, ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(X["Dhat"]-0.45, y), xytext=(X["D"]+0.45, y),
                arrowprops=dict(arrowstyle="->", lw=1.0))

# 4. Sign-elected Δ̃_t
for i, name in enumerate(TV_TILDE):
    y = row_y(i)
    ax.add_patch(patches.FancyBboxPatch((X["Dtilde"]-0.4, y-0.28), 0.8, 0.56,
                                        edgecolor="#333", facecolor="#80bfff", **box_style))
    ax.text(X["Dtilde"], y, name, ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(X["Dtilde"]-0.45, y), xytext=(X["Dhat"]+0.45, y),
                arrowprops=dict(arrowstyle="->", lw=1.0))

# 5. Quadratic form (emphasized — this is what makes SCALE activation-aware)
ax.add_patch(patches.FancyBboxPatch((X["Cq"]-0.9, 1.25), 1.8, 2.6,
                                    edgecolor="#b8860b", facecolor="#fff3b0",
                                    boxstyle="round,pad=0.4", linewidth=1.8, alpha=0.98))
ax.text(X["Cq"], 3.4, r"$\hat C_t \;=\;\tilde\Delta_t^{\!\top}\tilde\Delta_t$",
        ha="center", va="center", fontsize=13)
ax.text(X["Cq"], 2.75, r"data-free", ha="center", va="center", fontsize=9, style="italic")
ax.text(X["Cq"], 2.45, r"activation-aware", ha="center", va="center", fontsize=9, style="italic")
ax.text(X["Cq"], 2.15, r"quadratic form", ha="center", va="center", fontsize=9, style="italic")
# Routing tagline
ax.text(X["Cq"], 1.55, r"$\Rightarrow$ task-relevance", ha="center", va="center", fontsize=9.5)
ax.text(X["Cq"], 1.30, r"weighting (\S 5.2)", ha="center", va="center", fontsize=9.5)
# Arrows from each Δ̃_t into the quadratic box
for i in range(3):
    y = row_y(i)
    ax.annotate("", xy=(X["Cq"]-0.9, 2.55), xytext=(X["Dtilde"]+0.45, y),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#666",
                                connectionstyle="arc3,rad=0.0"))

# 6. Solve block
ax.add_patch(patches.FancyBboxPatch((X["Solve"]-0.95, 1.45), 1.9, 2.1,
                                    edgecolor="#333", facecolor="#e0f0e0", **box_style))
ax.text(X["Solve"], 3.25, r"$W^\star = \left(\sum_t \hat C_t\right)^{-1}$",
        ha="center", va="center", fontsize=10.5)
ax.text(X["Solve"], 2.75, r"$\quad \cdot\sum_t W_t\,\hat C_t$",
        ha="center", va="center", fontsize=10.5)
ax.text(X["Solve"], 2.2, r"(RegMean / MaTS", ha="center", va="center", fontsize=9, style="italic")
ax.text(X["Solve"], 1.95, r"closed form)", ha="center", va="center", fontsize=9, style="italic")
ax.annotate("", xy=(X["Solve"]-0.95, 2.5), xytext=(X["Cq"]+0.9, 2.5),
            arrowprops=dict(arrowstyle="->", lw=1.2))

# 7. Merged output
ax.add_patch(patches.FancyBboxPatch((X["Out"]-0.55, 2.1), 1.1, 0.9,
                                    edgecolor="#2e7d32", facecolor="#c2f0c2",
                                    boxstyle="round,pad=0.3", linewidth=1.6))
ax.text(X["Out"], 2.55, r"$W^\star$", ha="center", va="center", fontsize=15)
ax.annotate("", xy=(X["Out"]-0.55, 2.55), xytext=(X["Solve"]+0.95, 2.55),
            arrowprops=dict(arrowstyle="->", lw=1.3))

# Bottom caption-like annotation linking the two halves of the pipeline.
ax.annotate("", xy=(X["Dtilde"]+0.55, 0.6), xytext=(X["D"]-0.05, 0.6),
            arrowprops=dict(arrowstyle="-", lw=0.8, color="#666", linestyle=":"))
ax.text((X["D"]+X["Dtilde"])/2, 0.4, "TIES-style preprocessing",
        ha="center", va="center", fontsize=9.5, color="#333", style="italic")
ax.annotate("", xy=(X["Solve"]+0.95, 0.6), xytext=(X["Cq"]-0.85, 0.6),
            arrowprops=dict(arrowstyle="-", lw=0.8, color="#666", linestyle=":"))
ax.text((X["Cq"]+X["Solve"])/2, 0.4, "activation-aware linear system",
        ha="center", va="center", fontsize=9.5, color="#333", style="italic")

plt.tight_layout()
plt.savefig("/fsx/craffel/collectivedelusions/ml_research/testclaude/results/method_diagram.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("diagram saved")
