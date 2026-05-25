import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Static', 'PROTO-TTMM', 'IGGS-OW', 'TT-Diag-Fisher', 'KT-Fisher (Ours)']
fashion_acc = [15.89, 12.66, 77.24, 88.49, 85.83]
overall_acc = [46.65, 61.30, 82.83, 86.58, 85.69]

x = np.arange(len(methods))
width = 0.35

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 5))

rects1 = ax.bar(x - width/2, fashion_acc, width, label='FashionMNIST (Novel Domain)', color='#e74c3c')
rects2 = ax.bar(x + width/2, overall_acc, width, label='Overall Stream Accuracy', color='#3498db')

# Add labels and titles
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Test-Time Model Merging Performance on Non-Stationary Streams', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylim(0, 110)
ax.legend(loc='upper left', fontsize=11)

# Annotate bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='semibold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('results_plot.pdf', bbox_inches='tight')
print("Successfully saved plot to results_plot.pdf")
