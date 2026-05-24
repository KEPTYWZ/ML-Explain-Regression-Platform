# 特征选择工具使用说明

## 安装依赖

首先安装所需的Python包：

```bash
pip install pandas numpy scikit-learn lightgbm
```

## 基本使用

### 命令格式

```bash
python feature_selector.py \
  --dataset <数据集路径> \
  --target <目标变量名> \
  --mandatory <必选变量1> <必选变量2> ... \
  --min-vars <最小变量数> \
  --max-vars <最大变量数> \
  --strategy <搜索策略>
```

### 参数说明

| 参数 | 必需 | 说明 | 示例 |
|------|------|------|------|
| `--dataset` | 是 | CSV数据集文件路径 | `data.csv` |
| `--target` | 是 | 目标变量（因变量）的列名 | `fc` |
| `--mandatory` | 否 | 必须包含的变量列表 | `C W BFS` |
| `--min-vars` | 是 | 最少选择的变量数量 | `3` |
| `--max-vars` | 是 | 最多选择的变量数量（≤30） | `8` |
| `--strategy` | 是 | 搜索策略：`exhaustive` 或 `forward` | `forward` |
| `--test-size` | 否 | 测试集比例（默认0.2） | `0.2` |
| `--seed` | 否 | 随机种子（默认42） | `42` |
| `--output` | 否 | 结果输出文件（默认results.json） | `results.json` |

### 搜索策略选择

**1. 穷举搜索 (exhaustive)**
- 评估所有可能的变量组合
- 保证找到最优解
- 适合：变量总数≤15，max-vars≤8
- 计算量：组合数可能很大

**2. 前向选择 (forward)**
- 从必选变量开始，逐步添加最优变量
- 速度快，但不保证全局最优
- 适合：变量总数较多，或max-vars较大
- 计算量：相对较小

## 使用示例

### 示例1：小规模穷举搜索

假设你的数据集有10个变量，想找出3-5个变量的最优组合：

```bash
python feature_selector.py \
  --dataset data_cleaned_20260220_234443.csv \
  --target fc \
  --mandatory C W \
  --min-vars 3 \
  --max-vars 5 \
  --strategy exhaustive
```

**说明**：
- 数据集：`data_cleaned_20260220_234443.csv`
- 目标变量：`fc`（混凝土抗压强度）
- 必选变量：`C`（水泥）和 `W`（水）必须包含
- 变量范围：3到5个变量
- 策略：穷举搜索所有组合

### 示例2：大规模前向选择

假设你的数据集有20个变量，想找出5-10个变量的最优组合：

```bash
python feature_selector.py \
  --dataset data_cleaned_20260220_234443.csv \
  --target fc \
  --mandatory C W BFS \
  --min-vars 5 \
  --max-vars 10 \
  --strategy forward
```

**说明**：
- 必选变量：`C`、`W`、`BFS` 三个变量必须包含
- 变量范围：5到10个变量
- 策略：前向选择（速度更快）

### 示例3：自定义参数

```bash
python feature_selector.py \
  --dataset data_cleaned_20260220_234443.csv \
  --target fc \
  --mandatory C \
  --min-vars 4 \
  --max-vars 8 \
  --strategy forward \
  --test-size 0.25 \
  --seed 123 \
  --output my_results.json
```

**说明**：
- 测试集比例：25%（默认是20%）
- 随机种子：123（用于结果可重现）
- 输出文件：`my_results.json`

## 输出结果

程序运行后会：

1. **在控制台显示进度**：
   ```
   === Forward Selection ===
   Starting with mandatory variables: ['C', 'W']
   [1] Initial R² = 0.8234 with 2 variables
   [15] Added 'BFS': R² = 0.8756 (+0.0522)
   [28] Added 'FA': R² = 0.9012 (+0.0256)
   ...
   ✓ Forward selection complete!
   Best R² score: 0.9234
   Best combination (5 variables): ['C', 'W', 'BFS', 'FA', 'CA']
   ```

2. **保存JSON结果文件**（`results.json`）：
   ```json
   {
     "optimal_combination": ["C", "W", "BFS", "FA", "CA"],
     "performance_metric": {
       "name": "R²",
       "value": 0.9234
     },
     "search_summary": {
       "strategy": "forward",
       "combinations_evaluated": 45,
       "total_variables_available": 12
     },
     "configuration": {
       "mandatory_variables": ["C", "W"],
       "min_vars": 3,
       "max_vars": 8,
       "test_size": 0.2,
       "random_seed": 42
     }
   }
   ```

## 实际使用建议

### 针对你的混凝土数据集

假设你的数据集包含以下变量：
- 原始变量：`C`, `W`, `BFS`, `S`, `CA`, `FA`, `SP`, `Age`
- 衍生变量：`W_B`, `WB`, `Agg_B`, `C_B` 等

**推荐使用流程**：

1. **第一步：快速探索（前向选择）**
   ```bash
   python feature_selector.py \
     --dataset data_cleaned_20260220_234443.csv \
     --target fc \
     --mandatory C W \
     --min-vars 5 \
     --max-vars 10 \
     --strategy forward
   ```
   这会快速给你一个不错的变量组合。

2. **第二步：精细搜索（穷举搜索）**
   
   如果前向选择找到了8个变量，你可以在这8个变量附近做穷举搜索：
   ```bash
   python feature_selector.py \
     --dataset data_cleaned_20260220_234443.csv \
     --target fc \
     --mandatory C W \
     --min-vars 6 \
     --max-vars 8 \
     --strategy exhaustive
   ```

3. **第三步：验证结果**
   
   使用不同的随机种子验证结果的稳定性：
   ```bash
   python feature_selector.py \
     --dataset data_cleaned_20260220_234443.csv \
     --target fc \
     --mandatory C W \
     --min-vars 6 \
     --max-vars 8 \
     --strategy forward \
     --seed 123
   ```

## 常见问题

**Q: 穷举搜索需要多长时间？**

A: 取决于组合数量。例如：
- 从10个变量中选3-5个：约100-300个组合，几分钟
- 从15个变量中选5-8个：约5000-10000个组合，可能需要30分钟到1小时
- 从20个变量中选8-10个：组合数巨大，不推荐使用穷举

**Q: 如何知道我的数据集有哪些变量？**

A: 运行时如果目标变量或必选变量不存在，程序会显示所有可用的列名。

**Q: 必选变量可以为空吗？**

A: 可以。如果没有必选变量，不要加 `--mandatory` 参数即可。

**Q: 结果可重现吗？**

A: 是的。使用相同的 `--seed` 参数会得到完全相同的结果。

## 下一步

运行完成后，你可以：
1. 查看 `results.json` 获取最优变量组合
2. 在你的主程序（`app.py`）中使用这些变量训练模型
3. 比较不同策略和参数设置的结果
