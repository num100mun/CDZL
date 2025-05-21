import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.arange(0, 10, 1)
y1 = np.array([2, 3, 5, 8, 10, 12, 9, 6, 4, 3])
y2 = np.array([3, 5, 7, 9, 11, 10, 8, 6, 4, 3])
y3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 画三条折线
ax.plot(x, y1, label='线1', color='b', marker='o', linestyle='-', linewidth=2)
ax.plot(x, y2, label='线2', color='g', marker='s', linestyle='--', linewidth=2)
ax.plot(x, y3, label='线3', color='r', marker='^', linestyle='-.', linewidth=2)

# 添加横向参考线
reference_line_y = 7
ax.axhline(y=reference_line_y, color='k', linestyle='--', label='横向参考线')

# 标题和标签
ax.set_title('折线图示例')
ax.set_xlabel('X轴标签')
ax.set_ylabel('Y轴标签')

# 添加图例
ax.legend()

# 显示网格
ax.grid(True, linestyle='--', alpha=0.6)

# 美观的设置刻度
ax.set_xticks(x)
ax.set_yticks(np.arange(0, 15, 2))

# 设置坐标轴范围
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)

# 在图上添加数据点标签
for i, j in zip(x, y1):
    ax.annotate(f'{j}', (i, j), textcoords="offset points", xytext=(0, 10), ha='center')
for i, j in zip(x, y2):
    ax.annotate(f'{j}', (i, j), textcoords="offset points", xytext=(0, 10), ha='center')
for i, j in zip(x, y3):
    ax.annotate(f'{j}', (i, j), textcoords="offset points", xytext=(0, 10), ha='center')

# 保存图像
plt.savefig('line_plot.png', dpi=300)

# 显示图像
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# # 模拟 x 轴（如训练轮数、实验编号、或某种超参数取值）
# x = np.array([1, 2, 3, 4, 5])
#
# # 下方四组 y 数据仅做演示，您可替换为自己论文中的指标结果
# # 例如，这里我们假设是某项评价指标随 x 的变化
# baseline = np.array([0.65, 0.68, 0.72, 0.74, 0.73])   # Baseline
# model_a  = np.array([0.62, 0.66, 0.70, 0.77, 0.78])   # 模型A
# model_b  = np.array([0.58, 0.64, 0.71, 0.76, 0.77])   # 模型B
# model_c  = np.array([0.60, 0.65, 0.69, 0.72, 0.75])   # 模型C
#
# # 设置图像大小（可根据需要调整）
# plt.figure(figsize=(7, 5), dpi=100)
#
# # 绘制 Baseline，设置线型为虚线、颜色为红色，并加粗线条
# plt.plot(x, baseline,
#          label='Baseline',
#          color='red',
#          linestyle='--',
#          linewidth=2.5,
#          marker='o',
#          markersize=8,
#          markerfacecolor='white')
#
# # 绘制其他模型折线
# plt.plot(x, model_a,
#          label='Model A',
#          color='blue',
#          linestyle='-',
#          linewidth=2,
#          marker='s',
#          markersize=6)
# plt.plot(x, model_b,
#          label='Model B',
#          color='green',
#          linestyle='-',
#          linewidth=2,
#          marker='^',
#          markersize=6)
# plt.plot(x, model_c,
#          label='Model C',
#          color='orange',
#          linestyle='-',
#          linewidth=2,
#          marker='D',
#          markersize=6)
#
# # 添加网格线
# plt.grid(True, linestyle='--', alpha=0.7)
#
# # 设定坐标轴标签和标题
# plt.xlabel('X axis (e.g., Epoch / Iteration)', fontsize=12)
# plt.ylabel('Evaluation Score', fontsize=12)
# plt.title('Performance Comparison of Different Models', fontsize=14, fontweight='bold')
#
# # 设置坐标轴刻度字体大小
# plt.xticks(fontsize=11)
# plt.yticks(fontsize=11)
#
# # 图例放置在最佳位置
# plt.legend(loc='best', fontsize=10)
#
# # 适当留白，避免元素重叠
# plt.tight_layout()
#
# # 显示图像
# plt.show()


#
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# #-----------------------------------------
# #         1. 数据准备（示例）
# #-----------------------------------------
# x = np.array([1, 2, 3, 4, 5])
# # 下方仅为示例数据：可替换为您自己的实验结果
# baseline = np.array([0.60, 0.63, 0.65, 0.66, 0.68])
# model_a  = np.array([0.61, 0.65, 0.67, 0.69, 0.72])
# model_b  = np.array([0.57, 0.64, 0.66, 0.68, 0.71])
#
# #-----------------------------------------
# #         2. 样式与画布初始化
# #-----------------------------------------
# # 选择Seaborn白色背景风格并设定字体大小
# sns.set_style("white")
# sns.set_context("talk", font_scale=1.2)
#
# # 创建画布
# plt.figure(figsize=(7, 5), dpi=120)
#
# #-----------------------------------------
# #         3. 绘制线条
# #-----------------------------------------
# # 绘制 baseline，设置虚线、稍粗线宽，并用醒目的标记
# plt.plot(x, baseline, label='Baseline',
#          color='#2D3142',   # 深色
#          linestyle='--',
#          linewidth=2.5,
#          marker='o',
#          markersize=7)
#
# # 绘制 Model A
# plt.plot(x, model_a, label='Model A',
#          color='#EF8354',   # 橙色
#          linestyle='-',
#          linewidth=2,
#          marker='^',
#          markersize=7)
#
# # 绘制 Model B
# plt.plot(x, model_b, label='Model B',
#          color='#4F5D75',   # 灰蓝色
#          linestyle='-',
#          linewidth=2,
#          marker='s',
#          markersize=7)
#
# #-----------------------------------------
# #         4. 视觉增强：填充与网格
# #-----------------------------------------
# # 在baseline与Model A/B之间进行轻度填充，打造对比效果
# plt.fill_between(x, baseline, model_a,
#                  color='#EF8354', alpha=0.1)
# plt.fill_between(x, baseline, model_b,
#                  color='#4F5D75', alpha=0.1)
#
# # 如果喜欢网格，可启用(可选)：
# # plt.grid(True, linestyle='--', alpha=0.7)
#
# #-----------------------------------------
# #         5. 标题、坐标轴与装饰
# #-----------------------------------------
# plt.title("Performance Comparison", fontsize=16, fontweight='bold')
# plt.xlabel("Epoch", fontsize=13)
# plt.ylabel("Score", fontsize=13)
#
# # 设置x刻度为整点，并可根据需要设置y刻度范围
# plt.xticks(x)
# plt.yticks(np.arange(0.55, 0.76, 0.05))
#
# # 移除图像上方和右侧边框(更简洁)
# sns.despine()
#
# # 显示图例
# plt.legend(loc='lower right', frameon=False)  # 也可修改loc='best'
#
# # 保证布局紧凑
# plt.tight_layout()
#
# #-----------------------------------------
# #         6. 显示或保存图像
# #-----------------------------------------
# # 如果需要保存到文件请取消下面注释，并设置合适的dpi
# # plt.savefig('my_line_chart.png', dpi=300, bbox_inches='tight')
#
# plt.show()
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 设置 Seaborn 样式
# sns.set_style("whitegrid")
# sns.set_context("talk", font_scale=1.2)
#
# #------------------------------
# #   1. 准备示例数据
# #------------------------------
#
# # 假设有4个模型，每个模型在2个指标(BLEU, CIDEr)上的平均得分
# models = ["Baseline", "Model A", "Model B", "Model C"]
# bleu_scores = [0.65, 0.68, 0.70, 0.72]
# cider_scores = [0.95, 1.02, 1.10, 1.12]
#
# # 为图1准备分组柱状图的数据
# df_bar = pd.DataFrame({
#     "Model": models,
#     "BLEU": bleu_scores,
#     "CIDEr": cider_scores
# })
#
# # 为图2准备箱线图的数据
# # 假设在某项指标(如 BLEU)下，各模型进行了多次试验(10次)
# # 这里随机模拟数据，仅供演示
# np.random.seed(42)
# data_box = {
#     "Model": [],
#     "BLEU": []
# }
# for m in models:
#     # 在这里生成一些随机值模拟多次实验结果，可替换为真实数据
#     scores = np.random.normal(loc=bleu_scores[models.index(m)], scale=0.02, size=10)
#     data_box["Model"].extend([m]*10)
#     data_box["BLEU"].extend(scores)
# df_box = pd.DataFrame(data_box)
#
# #------------------------------
# #   2. 图1：分组柱状图
# #------------------------------
# plt.figure(figsize=(7,5), dpi=120)
#
# bar_width = 0.35
# x = np.arange(len(models))
#
# # 绘制 BLEU 柱
# plt.bar(x - bar_width/2, df_bar["BLEU"],
#         width=bar_width,
#         label="BLEU",
#         color="#EF8354",
#         edgecolor="white")
#
# # 绘制 CIDEr 柱
# plt.bar(x + bar_width/2, df_bar["CIDEr"],
#         width=bar_width,
#         label="CIDEr",
#         color="#4F5D75",
#         edgecolor="white")
#
# # 设置 x 轴刻度与标签
# plt.xticks(x, models)
# plt.ylabel("Score", fontsize=12)
# plt.title("Figure 1: Comparison of BLEU & CIDEr", fontsize=15, fontweight='bold')
#
# # 显示图例
# plt.legend(frameon=False, fontsize=11)
#
# # 微调布局
# plt.tight_layout()
#
# # 若需要保存图像可取消下行注释
# # plt.savefig("bar_chart_comparison.png", dpi=300, bbox_inches='tight')
#
# #------------------------------
# #   3. 图2：箱线图
# #------------------------------
# plt.figure(figsize=(7,5), dpi=120)
#
# sns.boxplot(x="Model", y="BLEU", data=df_box,
#             palette=["#2D3142","#EF8354","#4F5D75","#BFC0C0"],  # 可自定义配色
#             width=0.6)
#
# # 为箱线图增加散点（可选），使分布更直观
# sns.stripplot(x="Model", y="BLEU", data=df_box,
#               color="black", size=4, alpha=0.5)
#
# plt.title("Figure 2: Distribution of BLEU Scores across Models",
#           fontsize=15, fontweight='bold')
# plt.ylabel("BLEU Score", fontsize=12)
#
# plt.tight_layout()
# # plt.savefig("boxplot_distribution.png", dpi=300, bbox_inches='tight')
#
# plt.show()







#
# import numpy as np
# import matplotlib.pyplot as plt
#
# #-----------------------------
# #     1. 数据示例（请替换）
# #-----------------------------
# models = ["Baseline", "Model A", "Model B", "Model C"]
# bleu_scores  = [0.62, 0.68, 0.71, 0.73]  # 用于柱状图
# cider_scores = [0.95, 1.10, 1.15, 1.20]  # 用于折线图
#
# x = np.arange(len(models))  # x轴刻度
#
# #-----------------------------
# #     2. 画布初始化
# #-----------------------------
# fig, ax1 = plt.subplots(figsize=(7,5), dpi=120)
# # 创建共享X轴，但独立Y轴的第二个坐标系
# ax2 = ax1.twinx()
#
# #-----------------------------
# #     3. 绘制柱状图（BLEU）
# #-----------------------------
# width = 0.4
# bars = ax1.bar(x, bleu_scores,
#                width=width,
#                color="#4F5D75",
#                edgecolor="white",
#                label="BLEU")
#
# # 设置 ax1 的y轴标签与范围
# ax1.set_ylabel("BLEU Score", fontsize=12, color="#4F5D75")
# # (可根据实际值调整 Y 轴范围，比如)
# ax1.set_ylim(0, 1.0)
#
# #-----------------------------
# #     4. 绘制折线图（CIDEr）
# #-----------------------------
# line = ax2.plot(x, cider_scores,
#                 color="#EF8354",
#                 linewidth=2.5,
#                 marker='o',
#                 markersize=8,
#                 label="CIDEr")
#
# # 设置 ax2 的y轴标签与范围
# ax2.set_ylabel("CIDEr Score", fontsize=12, color="#EF8354")
# ax2.set_ylim(0, 1.5)  # 根据实际数值适当设置
#
# #-----------------------------
# #     5. 设置X轴刻度和标题
# #-----------------------------
# plt.xticks(x, models, fontsize=11)
# plt.title("Comparison of BLEU (Bar) and CIDEr (Line)",
#           fontsize=15, fontweight='bold')
#
# #-----------------------------
# #     6. 添加图例
# #-----------------------------
# # 分别获取两个坐标轴的图例句柄和标签
# bars_handles, bars_labels   = ax1.get_legend_handles_labels()
# line_handles, line_labels   = ax2.get_legend_handles_labels()
#
# # 合并图例并放在一个位置
# ax1.legend(bars_handles + line_handles,
#            bars_labels + line_labels,
#            loc="upper left", frameon=False)
#
# # 细调布局
# plt.tight_layout()
#
# # 如果需要保存图片到文件(如论文插图),可使用：
# # plt.savefig("bar_line_combined.png", dpi=300, bbox_inches='tight')
#
# plt.show()
