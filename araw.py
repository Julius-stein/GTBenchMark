import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv("results_flex_vs_sdpa.csv")

# 只保留 Flex Attention 的数据
# flex_df = df[df["winner"] == "flex"]

# 按 L 分组画图
plt.figure(figsize=(8,6))

for L, group in df.groupby("L"):
    plt.plot(
        group["density_actual"], 
        group["t_flex"], 
        marker="o", 
        linewidth=2, 
        label=f"L={L}"
    )

plt.xlabel("Density (nominal)")
plt.ylabel("Flex Attention Time (s)")
plt.title("Flex Attention Runtime vs Density (Different L)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Sequence Length (L)")
plt.tight_layout()
plt.savefig("flex")
