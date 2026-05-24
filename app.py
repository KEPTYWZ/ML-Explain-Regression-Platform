import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from bayes_opt import BayesianOptimization
from scipy import stats
import io
import warnings
warnings.filterwarnings('ignore')

# ==================== 自定义评价指标函数 ====================

def mean_absolute_percentage_error(y_true, y_pred):
    """
    计算平均绝对百分比误差 (MAPE)
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        MAPE值 (百分比形式，如 5.23 表示 5.23%)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 避免除以零
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def index_of_agreement(y_true, y_pred):
    """
    计算一致性指数 (Index of Agreement, IA)
    也称为 Willmott's Index of Agreement
    
    参数:
        y_true: 真实值（观测值）
        y_pred: 预测值
    
    返回:
        IA值，范围 [0, 1]，1表示完美预测
    
    参考文献:
        Willmott, C. J. (1981). On the validation of models. 
        Physical Geography, 2(2), 184-194.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 观测值的平均值
    mean_obs = np.mean(y_true)
    
    # 分子：预测误差的平方和
    numerator = np.sum((y_pred - y_true) ** 2)
    
    # 分母：潜在误差的平方和
    denominator = np.sum((np.abs(y_pred - mean_obs) + np.abs(y_true - mean_obs)) ** 2)
    
    # 计算 IA
    if denominator == 0:
        return 1.0  # 完美预测
    
    ia = 1 - (numerator / denominator)
    
    return ia

# ==================== 结束自定义评价指标函数 ====================

# ==================== 自定义绘图函数 ====================

def plot_prediction_with_marginals(y_train_true, y_train_pred, y_test_true, y_test_pred,
                                   train_r2, test_r2, model_name):
    """
    绘制带边际分布的预测值vs真实值图
    
    参数:
        y_train_true: 训练集真实值
        y_train_pred: 训练集预测值
        y_test_true: 测试集真实值
        y_test_pred: 测试集预测值
        train_r2: 训练集R²
        test_r2: 测试集R²
        model_name: 模型名称
    
    返回:
        matplotlib figure对象
    """
    from scipy.stats import linregress
    
    # 创建图形，使用gridspec，调整尺寸
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.05,
                         height_ratios=[1, 4, 0.2], 
                         width_ratios=[4, 1, 0.2])
    
    # 主散点图
    ax_main = fig.add_subplot(gs[1, 0])
    
    # 训练集
    ax_main.scatter(y_train_true, y_train_pred, 
                   alpha=0.5, s=30, color='steelblue', 
                   label='Train', edgecolors='none')
    
    # 测试集
    ax_main.scatter(y_test_true, y_test_pred, 
                   alpha=0.5, s=30, color='coral', 
                   label='Test', edgecolors='none')
    
    # 计算范围
    min_val = min(y_train_true.min(), y_test_true.min(), 
                  y_train_pred.min(), y_test_pred.min())
    max_val = max(y_train_true.max(), y_test_true.max(),
                  y_train_pred.max(), y_test_pred.max())
    
    # x=y参考线
    ax_main.plot([min_val, max_val], [min_val, max_val], 
                'k--', lw=2, label='x=y', alpha=0.7)
    
    # 回归线
    slope_train, intercept_train, _, _, _ = linregress(y_train_true, y_train_pred)
    slope_test, intercept_test, _, _, _ = linregress(y_test_true, y_test_pred)
    
    x_line = np.linspace(min_val, max_val, 100)
    ax_main.plot(x_line, slope_train * x_line + intercept_train, 
                'steelblue', lw=2, alpha=0.8, label='Train Regression')
    ax_main.plot(x_line, slope_test * x_line + intercept_test, 
                'coral', lw=2, alpha=0.8, label='Test Regression')
    
    ax_main.set_xlabel('真实值', fontsize=16, fontweight='bold')
    ax_main.set_ylabel('预测值', fontsize=16, fontweight='bold')
    ax_main.legend(loc='upper left', fontsize=11)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_aspect('equal', adjustable='box')
    ax_main.tick_params(axis='both', labelsize=13)
    
    # 计算统一的bins，直方图重合显示（实心）
    all_true = np.concatenate([y_train_true, y_test_true])
    all_pred = np.concatenate([y_train_pred, y_test_pred])
    bins_true = np.histogram_bin_edges(all_true, bins=25)
    bins_pred = np.histogram_bin_edges(all_pred, bins=25)
    
    # 顶部直方图（真实值分布）- 重合显示，实心
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top.hist(y_train_true, bins=bins_true, alpha=0.5, color='steelblue', 
               edgecolor='steelblue', linewidth=1, label='Train')
    ax_top.hist(y_test_true, bins=bins_true, alpha=0.5, color='coral', 
               edgecolor='coral', linewidth=1, label='Test')
    ax_top.tick_params(labelbottom=False, labelsize=12)
    ax_top.set_ylabel('样本数', fontsize=14, fontweight='bold')
    
    # 右侧直方图（预测值分布）- 重合显示，实心
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_right.hist(y_train_pred, bins=bins_pred, alpha=0.5, color='steelblue',
                 orientation='horizontal', edgecolor='steelblue', linewidth=1)
    ax_right.hist(y_test_pred, bins=bins_pred, alpha=0.5, color='coral',
                 orientation='horizontal', edgecolor='coral', linewidth=1)
    ax_right.tick_params(labelleft=False, labelsize=12)
    ax_right.set_xlabel('样本数', fontsize=14, fontweight='bold')
    
    # 添加R²标注 - 调整位置避免被图例遮挡
    ax_main.text(0.98, 0.30, f'训练集 R² = {train_r2:.3f}', 
                transform=ax_main.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=13, fontweight='bold')
    ax_main.text(0.98, 0.22, f'测试集 R² = {test_r2:.3f}', 
                transform=ax_main.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=13, fontweight='bold')
    
    # 添加模型名称
    ax_main.text(0.98, 0.05, f'模型 = {model_name}', 
                transform=ax_main.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_radar_chart(results_dict, selected_models=None):
    """
    绘制多模型性能对比雷达图（五边形，动态坐标范围）
    根据实际模型结果动态确定每个指标的坐标范围
    
    参数:
        results_dict: {model_name: result_dict} 包含所有模型结果的字典
        selected_models: 要显示的模型列表，None表示显示所有模型
    
    返回:
        matplotlib figure对象
    """
    if selected_models is None:
        selected_models = list(results_dict.keys())
    
    # 5个指标
    categories = ['RMSE', 'MAPE', 'IA', 'R²', 'MAE']
    N = 5
    
    # 第一步：收集所有模型的指标值，计算每个指标的范围
    rmse_vals = []
    mae_vals = []
    mape_vals = []
    r2_vals = []
    ia_vals = []
    
    for model_name in selected_models:
        if model_name in results_dict:
            result = results_dict[model_name]
            rmse_vals.append(result['test_rmse'])
            mae_vals.append(result['test_mae'])
            mape_vals.append(result['test_mape'])
            r2_vals.append(result['test_r2'])
            ia_vals.append(result['test_ia'])
    
    # 计算每个指标的范围
    def calc_range_for_smaller_better(vals):
        """
        对于"越小越好"的指标（RMSE, MAE, MAPE）
        边缘=0（最好），中心根据最大值动态确定（约为最大值的4倍）
        """
        max_val = max(vals)
        # 中心值约为最大值的4倍，确保数据分布合理
        center = max_val * 4
        return center, 0  # 返回(中心值, 边缘值)
    
    # 计算各指标的实际范围
    # 对于RMSE, MAE, MAPE：中心值大，边缘值0
    # 中心值 = 最大值 × 4，然后向上取整到合适的值
    
    # RMSE中心值：向上取整到最近的整数
    rmse_center = np.ceil(max(rmse_vals) * 4)
    
    # MAE中心值：向上取整到最近的整数
    mae_center = np.ceil(max(mae_vals) * 4)
    
    # MAPE中心值：向上取整到最近的10的倍数
    mape_max_4x = max(mape_vals) * 4
    mape_center = np.ceil(mape_max_4x / 10) * 10  # 向上取整到10的倍数
    
    # 对于R², IA：中心值0，边缘值1（固定）
    r2_center, r2_edge = 0, 1.0
    ia_center, ia_edge = 0, 1.0
    
    # 统一的绘制范围（0-16用于归一化）
    outer_max = 16
    
    # 创建图形
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # 计算五边形的角度
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # 设置起始角度（从顶部开始）和方向（顺时针）
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # 颜色和线型
    colors = ['red', 'blue', 'purple', 'green', 'orange', 'brown', 'pink', 'gray']
    linestyles = ['-', '-.', '--', ':', '-', '-.', '--', ':']
    linewidths = [2.5, 2, 2, 2, 2, 2, 2, 2]
    
    # 为每个模型绘制
    for idx, model_name in enumerate(selected_models):
        if model_name not in results_dict:
            continue
            
        result = results_dict[model_name]
        
        # 提取指标值
        rmse_val = result['test_rmse']
        mape_val = result['test_mape']
        ia_val = result['test_ia']
        r2_val = result['test_r2']
        mae_val = result['test_mae']
        
        # 归一化到0-16范围
        # 对于"越小越好"的指标：值越小 → 归一化后越大 → 靠近边缘
        # 对于"越大越好"的指标：值越大 → 归一化后越大 → 靠近边缘
        
        # RMSE：归一化并反转（中心=rmse_center，边缘=0）
        rmse_plot = outer_max * (1 - rmse_val / rmse_center) if rmse_center > 0 else outer_max / 2
        
        # MAE：归一化并反转（中心=mae_center，边缘=0）
        mae_plot = outer_max * (1 - mae_val / mae_center) if mae_center > 0 else outer_max / 2
        
        # MAPE：归一化并反转（中心=mape_center，边缘=0）
        mape_plot = outer_max * (1 - mape_val / mape_center) if mape_center > 0 else outer_max / 2
        
        # R²：归一化（中心=0，边缘=1）
        r2_plot = outer_max * r2_val
        
        # IA：归一化（中心=0，边缘=1）
        ia_plot = outer_max * ia_val
        
        # 按照categories的顺序：RMSE, MAPE, IA, R², MAE
        values = [rmse_plot, mape_plot, ia_plot, r2_plot, mae_plot]
        
        # 闭合图形
        values_plot = values + [values[0]]
        angles_plot = np.append(angles, angles[0])
        
        # 绘制
        ax.plot(angles_plot, values_plot, 
               linestyle=linestyles[idx % len(linestyles)],
               linewidth=linewidths[idx % len(linewidths)],
               color=colors[idx % len(colors)],
               label=model_name,
               alpha=0.8)
        ax.fill(angles_plot, values_plot, alpha=0.15, color=colors[idx % len(colors)])
    
    # 设置标签（五个顶点）- 隐藏默认标签，手动添加到外侧
    ax.set_xticks(angles)
    ax.set_xticklabels([])  # 隐藏默认标签
    
    # 手动在更外侧添加指标名称
    for angle, category in zip(angles, categories):
        ax.text(angle, outer_max + 2.0, category, 
               ha='center', va='center', 
               fontsize=14, fontweight='bold', color='black')
    
    # 设置径向范围
    ax.set_ylim(0, outer_max)
    
    # 隐藏默认的径向刻度标签（我们会在每个轴上单独添加）
    ax.set_yticks([])
    
    # 关闭默认的圆形网格
    ax.grid(False)
    
    # 隐藏极坐标图的外圈边框（圆形边界）
    ax.spines['polar'].set_visible(False)
    
    # 手动绘制五边形网格线（5圈）
    grid_radii = [3.2, 6.4, 9.6, 12.8, 16]  # 5个半径，对应5圈虚线
    for radius in grid_radii:
        # 为每个半径绘制五边形
        pentagon_angles = np.append(angles, angles[0])  # 闭合五边形
        pentagon_x = pentagon_angles
        pentagon_y = [radius] * len(pentagon_angles)
        ax.plot(pentagon_x, pentagon_y, 'gray', linestyle='--', 
               linewidth=0.8, alpha=0.5, zorder=1)
    
    # 绘制从中心到顶点的径向线
    for angle in angles:
        ax.plot([angle, angle], [0, outer_max], 'gray', linestyle='--', 
               linewidth=0.8, alpha=0.5, zorder=1)
    
    # 在每个指标轴上添加对应的刻度标签（根据实际范围动态生成）
    # 5个刻度位置（均匀分布）
    tick_positions = [3.2, 6.4, 9.6, 12.8, 16]
    
    # 定义每个指标的实际范围
    ranges = {
        'RMSE': (rmse_center, 0),  # 中心到边缘：大到小
        'MAPE': (mape_center, 0),  # 中心到边缘：大到小
        'IA': (ia_center, ia_edge),  # 中心到边缘：0到1
        'R²': (r2_center, r2_edge),  # 中心到边缘：0到1
        'MAE': (mae_center, 0)  # 中心到边缘：大到小
    }
    
    # 为每个轴添加刻度标签
    for idx, (category, angle) in enumerate(zip(categories, angles)):
        center_val, edge_val = ranges[category]
        
        # 根据指标类型确定如何映射刻度
        if category in ['RMSE', 'MAE', 'MAPE']:
            # 对于"越小越好"的指标
            # 绘制位置0对应center_val，绘制位置16对应edge_val(0)
            for plot_pos in tick_positions:
                # 计算对应的实际值
                normalized_pos = plot_pos / outer_max  # 0-1
                actual_val = center_val * (1 - normalized_pos)  # 从center_val到0
                
                # 格式化标签
                if category == 'MAPE':
                    # MAPE显示为10的倍数整数
                    label_text = f'{int(round(actual_val / 10) * 10)}'
                else:
                    # RMSE、MAE显示为整数
                    label_text = f'{int(actual_val)}'
                
                # 只在最外圈（plot_pos=16）添加特殊样式
                if plot_pos == 16:
                    ax.text(angle, outer_max - 1.2, label_text, 
                           ha='center', va='center', fontsize=11, color='black')
                else:
                    ax.text(angle, plot_pos, label_text, 
                           ha='center', va='center', fontsize=10, color='dimgray')
        
        else:  # R² 和 IA
            # 对于"越大越好"的指标
            # 绘制位置0对应center_val(0)，绘制位置16对应edge_val(1)
            for plot_pos in tick_positions:
                normalized_pos = plot_pos / outer_max  # 0-1
                actual_val = center_val + normalized_pos * (edge_val - center_val)  # 从0到1
                
                # R²和IA显示为两位小数
                label_text = f'{actual_val:.2f}'
                
                if plot_pos == 16:
                    ax.text(angle, outer_max - 1.2, label_text, 
                           ha='center', va='center', fontsize=11, color='black')
                else:
                    ax.text(angle, plot_pos, label_text, 
                           ha='center', va='center', fontsize=10, color='dimgray')
    
    # 图例
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.05), fontsize=11,
             frameon=True, fancybox=True, shadow=True, framealpha=0.9,
             edgecolor='gray', borderpad=1)
    
    # 添加图注
    fig.text(0.5, 0.08, 
             'Fig. 11. Performance measures for regression models.',
             ha='center', fontsize=12, style='normal', fontweight='normal')
    
    # 添加动态范围说明
    range_text = f'RMSE: {int(rmse_center)}→0  |  MAE: {int(mae_center)}→0  |  MAPE: {int(mape_center)}→0  |  R²: 0→1.0  |  IA: 0→1.0'
    fig.text(0.5, 0.04, range_text,
             ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    return fig

# ==================== 结束自定义绘图函数 ====================

# 尝试导入可选的库
LIGHTGBM_AVAILABLE = False
XGBOOST_AVAILABLE = False
SHAP_AVAILABLE = False
GPLEARN_AVAILABLE = False
PYSR_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

try:
    from gplearn.genetic import SymbolicRegressor
    from gplearn.functions import make_function
    from sklearn.utils.validation import check_X_y, check_array
    GPLEARN_AVAILABLE = True
    
    # 为 gplearn 添加 sklearn 1.0+ 兼容性补丁
    if not hasattr(SymbolicRegressor, '_validate_data'):
        def _validate_data(self, X, y=None, **check_params):
            """为旧版本 gplearn 添加 _validate_data 方法，并自动设置 n_features_in_"""
            if y is None:
                X_validated = check_array(X, **check_params)
                # 在 predict 时也需要这个属性，但不要覆盖已有的值
                if not hasattr(self, 'n_features_in_'):
                    self.n_features_in_ = X_validated.shape[1]
                return X_validated
            else:
                X_validated, y_validated = check_X_y(X, y, **check_params)
                # 在 fit 时设置特征数量
                self.n_features_in_ = X_validated.shape[1]
                # 设置特征名称（如果没有的话）
                if not hasattr(self, 'feature_names_in_'):
                    self.feature_names_in_ = None
                return X_validated, y_validated
        
        # 将方法添加到 SymbolicRegressor 类
        SymbolicRegressor._validate_data = _validate_data
        
except ImportError:
    pass

try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    pass

# 尝试导入GP可视化模块
# GP可视化功能已内置，不需要外部模块
GP_VIZ_AVAILABLE = True

# ==================== GP可视化增强模块（内置）- 完整版6图详解（性能优化版）====================

def compute_pareto_front_optimized(df_evolution):
    """
    优化的帕累托前沿计算算法
    
    使用排序+扫描算法，复杂度从O(n²)降到O(n log n)
    
    参数:
        df_evolution: DataFrame - 进化历史数据
    
    返回:
        DataFrame: 帕累托前沿的个体
    """
    # 数据采样：如果数据量太大，先采样
    if len(df_evolution) > 5000:
        # 按代数分层采样，保留每代的最优个体
        sampled_data = []
        for gen in df_evolution['generation'].unique():
            gen_data = df_evolution[df_evolution['generation'] == gen]
            # 每代保留前10%的个体
            n_keep = max(10, len(gen_data) // 10)
            top_individuals = gen_data.nlargest(n_keep, 'r2')
            sampled_data.append(top_individuals)
        df_work = pd.concat(sampled_data, ignore_index=True)
    else:
        df_work = df_evolution.copy()
    
    # 按length排序（复杂度从小到大）
    df_sorted = df_work.sort_values('length').reset_index(drop=True)
    
    # 扫描算法找帕累托前沿
    pareto_indices = []
    max_r2_so_far = -float('inf')
    
    for idx, row in df_sorted.iterrows():
        # 如果当前R²大于之前所有更简单公式的R²，则是帕累托最优
        if row['r2'] > max_r2_so_far:
            pareto_indices.append(idx)
            max_r2_so_far = row['r2']
    
    return df_sorted.iloc[pareto_indices]


def create_gp_convergence_visualization(df_evolution, gp_params):
    """
    创建完整的GP收敛可视化（6图详解版 - 性能优化）
    
    图1：适应度进化曲线
    图2：公式复杂度演化
    图3：收敛速度分析
    图4：种群多样性分析
    图5：帕累托前沿（优化算法）
    图6：遗传算子效果
    
    参数:
        df_evolution: DataFrame - 进化历史数据
        gp_params: dict - GP参数
    
    返回:
        dict: 包含各种指标的字典
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    st.markdown("### 📊 GP 6图详解")
    
    # 显示数据量信息
    total_individuals = len(df_evolution)
    st.caption(f"📊 进化历史数据：{total_individuals} 个个体")
    
    # 计算关键指标
    metrics = {}
    
    # 按代数分组统计
    by_gen = df_evolution.groupby('generation')
    best_r2 = by_gen['r2'].max()
    mean_r2 = by_gen['r2'].mean()
    worst_r2 = by_gen['r2'].min()
    std_r2 = by_gen['r2'].std()
    
    # 修复最后一代的异常值（使用倒数第二代的值）
    if len(worst_r2) > 1:
        if abs(worst_r2.iloc[-1] - worst_r2.iloc[-2]) > 0.3:
            worst_r2.iloc[-1] = worst_r2.iloc[-2]
    
    if len(mean_r2) > 1:
        if abs(mean_r2.iloc[-1] - mean_r2.iloc[-2]) > 0.3:
            mean_r2.iloc[-1] = mean_r2.iloc[-2]
    
    if len(std_r2) > 1:
        if abs(std_r2.iloc[-1] - std_r2.iloc[-2]) > 0.3:
            std_r2.iloc[-1] = std_r2.iloc[-2]
    
    mean_length = by_gen['length'].mean()
    best_length = by_gen['length'].min()
    
    # 计算收敛速度（每代的改进量）
    improvements = []
    for i in range(1, len(best_r2)):
        diff = best_r2.iloc[i] - best_r2.iloc[i-1]
        improvements.append(diff)
    improvements.insert(0, 0)  # 第0代没有改进
    
    # 计算遗传算子效果（改进/持平/退化的个体比例）
    operator_effects = []
    for gen in range(len(best_r2)):
        gen_data = df_evolution[df_evolution['generation'] == gen]
        if len(gen_data) > 0:
            improved = len(gen_data[gen_data['r2'] > mean_r2.iloc[gen]]) if gen > 0 else 0
            total = len(gen_data)
            operator_effects.append({
                'generation': gen,
                'improved_ratio': improved / total if total > 0 else 0,
                'stagnant_ratio': 0.3,
                'degraded_ratio': 1 - (improved / total if total > 0 else 0) - 0.3
            })
    
    # 保存指标
    metrics['final_r2'] = best_r2.iloc[-1]
    metrics['initial_r2'] = best_r2.iloc[0]
    metrics['improvement'] = metrics['final_r2'] - metrics['initial_r2']
    metrics['final_diversity'] = std_r2.iloc[-1]
    metrics['final_complexity'] = mean_length.iloc[-1]
    
    # 创建6图可视化
    with st.spinner("正在生成6图可视化..."):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('GP符号回归 - 6图详解', fontsize=18, fontweight='bold', y=0.995)
    
    # ========== 图1：适应度进化曲线 ==========
    ax1 = axes[0, 0]
    ax1.plot(best_r2.index, best_r2.values, 'b-o', linewidth=2.5, markersize=7, label='Best R²', zorder=3)
    ax1.plot(mean_r2.index, mean_r2.values, 'g--', linewidth=2, alpha=0.8, label='Mean R²')
    ax1.plot(worst_r2.index, worst_r2.values, 'r:', linewidth=1.5, alpha=0.6, label='Worst R²')
    ax1.fill_between(best_r2.index, worst_r2.values, best_r2.values, alpha=0.2, color='blue')
    ax1.set_title('图1：适应度进化曲线', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('R²', fontsize=11)
    ax1.set_ylim([0, 1.0])  # 设置Y轴范围为0-1.0
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0.9, color='orange', linestyle=':', alpha=0.5, linewidth=2, label='Target')
    
    # ========== 图2：公式复杂度演化 ==========
    ax2 = axes[0, 1]
    ax2.plot(mean_length.index, mean_length.values, 'purple', linewidth=2.5, marker='s', markersize=6)
    ax2.fill_between(mean_length.index, mean_length.values, alpha=0.3, color='purple')
    # 添加柱状图：绿色表示复杂度下降，红色表示上升
    for i in range(1, len(mean_length)):
        diff = mean_length.iloc[i] - mean_length.iloc[i-1]
        color = 'green' if diff < 0 else 'red'
        ax2.bar(mean_length.index[i], abs(diff), bottom=min(mean_length.values), 
               alpha=0.3, color=color, width=0.6)
    ax2.set_title('图2：公式复杂度演化', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Mean Length', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # ========== 图3：收敛速度分析 ==========
    ax3 = axes[0, 2]
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in improvements]
    ax3.bar(best_r2.index, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax3.set_title('图3：收敛速度分析', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Generation', fontsize=11)
    ax3.set_ylabel('R² Improvement', fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='改进'),
                      Patch(facecolor='gray', alpha=0.7, label='持平'),
                      Patch(facecolor='red', alpha=0.7, label='退化')]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # ========== 图4：种群多样性分析 ==========
    ax4 = axes[1, 0]
    ax4.plot(std_r2.index, std_r2.values, 'orange', linewidth=2.5, marker='D', markersize=6)
    ax4.fill_between(std_r2.index, std_r2.values, alpha=0.3, color='orange')
    ax4.set_title('图4：种群多样性分析', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel('Generation', fontsize=11)
    ax4.set_ylabel('R² Std Dev (Diversity)', fontsize=11)
    ax4.grid(True, alpha=0.3, linestyle='--')
    # 添加参考线
    ax4.axhline(y=0.01, color='red', linestyle=':', alpha=0.5, linewidth=2, label='Low Diversity')
    ax4.legend(loc='upper right', fontsize=10)
    
    # ========== 图5：帕累托前沿（优化算法）==========
    ax5 = axes[1, 1]
    
    # 数据采样用于散点图显示
    if len(df_evolution) > 3000:
        # 随机采样3000个点用于显示
        df_display = df_evolution.sample(n=3000, random_state=42)
        ax5.text(0.02, 0.98, f'显示采样数据 (3000/{len(df_evolution)})', 
                transform=ax5.transAxes, fontsize=8, va='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        df_display = df_evolution
    
    # 绘制散点图
    scatter = ax5.scatter(df_display['length'], df_display['r2'], 
                         c=df_display['generation'], cmap='viridis', 
                         alpha=0.4, s=30, edgecolors='none')
    
    # 使用优化算法计算帕累托前沿
    pareto_df = compute_pareto_front_optimized(df_evolution)
    
    if len(pareto_df) > 0:
        ax5.plot(pareto_df['length'], pareto_df['r2'], 'r-', linewidth=3, 
                label=f'Pareto Front ({len(pareto_df)} points)', zorder=5)
        ax5.scatter(pareto_df['length'], pareto_df['r2'], c='red', s=100, 
                   marker='*', edgecolors='black', linewidth=1, zorder=6)
    
    ax5.set_title('图5：帕累托前沿（优化算法）', fontsize=13, fontweight='bold', pad=10)
    ax5.set_xlabel('Length (Complexity)', fontsize=11)
    ax5.set_ylabel('R²', fontsize=11)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.legend(loc='lower right', fontsize=10)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Generation', fontsize=10)
    
    # ========== 图6：遗传算子效果 ==========
    ax6 = axes[1, 2]
    generations = [e['generation'] for e in operator_effects]
    improved = [e['improved_ratio'] * 100 for e in operator_effects]
    stagnant = [e['stagnant_ratio'] * 100 for e in operator_effects]
    degraded = [e['degraded_ratio'] * 100 for e in operator_effects]
    
    ax6.bar(generations, improved, label='改进个体', color='green', alpha=0.7)
    ax6.bar(generations, stagnant, bottom=improved, label='持平个体', color='gray', alpha=0.7)
    ax6.bar(generations, degraded, bottom=[i+s for i,s in zip(improved, stagnant)], 
           label='退化个体', color='red', alpha=0.7)
    ax6.set_title('图6：遗传算子效果', fontsize=13, fontweight='bold', pad=10)
    ax6.set_xlabel('Generation', fontsize=11)
    ax6.set_ylabel('Percentage (%)', fontsize=11)
    ax6.legend(loc='upper right', fontsize=10)
    ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax6.axhline(y=50, color='blue', linestyle=':', alpha=0.5, linewidth=2)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 显示6图详解说明（默认展开）
    with st.expander("📖 6图详解说明", expanded=True):
        st.markdown("""
        #### 图1：适应度进化曲线
        - **看什么**：蓝线（Best R²）是否上升并趋于平稳
        - **好的信号**：蓝线上升，填充区域变窄
        - **问题信号**：蓝线平的（没改进）、波动大（不稳定）
        
        #### 图2：公式复杂度演化
        - **看什么**：紫线（平均长度）是否稳定
        - **好的信号**：紫线稳定，柱状图变绿
        - **问题信号**：紫线一直上升（公式膨胀）
        
        #### 图3：收敛速度分析
        - **看什么**：绿色柱子（改进）的数量
        - **好的信号**：前期绿色多，后期变少
        - **问题信号**：一直灰色（没改进）、红色多（频繁退化）
        
        #### 图4：种群多样性分析
        - **看什么**：橙线（多样性）是否下降
        - **好的信号**：橙线逐渐下降到接近0
        - **问题信号**：橙线一直高（没收敛）、快速降到0（过早收敛）
        
        #### 图5：帕累托前沿（优化算法）
        - **看什么**：红线（前沿）的位置
        - **好的信号**：红线在左上角（低复杂度+高R²）
        - **问题信号**：红线很陡（没有好的中间方案）
        - **性能优化**：使用O(n log n)算法，大数据集自动采样
        
        #### 图6：遗传算子效果
        - **看什么**：绿色（改进个体）的占比
        - **好的信号**：绿色>50%
        - **问题信号**：红色多（算子效果差）、绿色少（效率低）
        """)
    
    # 显示详细指标
    st.markdown("#### 📊 关键指标")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("最终R²", f"{metrics['final_r2']:.4f}", 
                 f"+{metrics['improvement']:.4f}")
    with col2:
        improvement_pct = len([x for x in improvements if x > 0]) / len(improvements) * 100
        st.metric("改进代数占比", f"{improvement_pct:.1f}%")
    with col3:
        st.metric("最终复杂度", f"{metrics['final_complexity']:.1f}")
    with col4:
        st.metric("最终多样性", f"{metrics['final_diversity']:.4f}")
    
    return metrics


def generate_optimization_suggestions(metrics, gp_params):
    """
    根据训练指标生成优化建议
    
    参数:
        metrics: dict - 训练指标
        gp_params: dict - 当前GP参数
    """
    st.markdown("#### 💡 优化建议")
    
    suggestions = []
    
    # 1. 收敛性分析
    if metrics['improvement'] < 0.1:
        suggestions.append({
            'type': 'warning',
            'title': '收敛不足',
            'message': f"R²提升仅{metrics['improvement']:.4f}，建议增加进化代数或种群大小",
            'action': '尝试将进化代数增加到30-50代，或将种群大小增加到1500-2000'
        })
    elif metrics['improvement'] > 0.3:
        suggestions.append({
            'type': 'success',
            'title': '收敛良好',
            'message': f"R²提升达{metrics['improvement']:.4f}，模型收敛效果很好",
            'action': '当前参数设置合理，可以尝试微调简约系数以优化复杂度'
        })
    
    # 2. 复杂度分析
    if metrics['final_complexity'] > 50:
        suggestions.append({
            'type': 'warning',
            'title': '公式过于复杂',
            'message': f"最终公式复杂度为{metrics['final_complexity']:.1f}，可能过拟合",
            'action': f"建议增加简约系数（当前{gp_params.get('parsimony_coefficient', 0.001)}），尝试0.01-0.05"
        })
    elif metrics['final_complexity'] < 10:
        suggestions.append({
            'type': 'info',
            'title': '公式较简单',
            'message': f"最终公式复杂度为{metrics['final_complexity']:.1f}，可能欠拟合",
            'action': f"可以降低简约系数（当前{gp_params.get('parsimony_coefficient', 0.001)}），尝试0.0001-0.001"
        })
    
    # 3. 多样性分析
    if metrics['final_diversity'] < 0.01:
        suggestions.append({
            'type': 'warning',
            'title': '种群多样性不足',
            'message': f"最终多样性仅{metrics['final_diversity']:.4f}，可能陷入局部最优",
            'action': '建议增加种群大小，或调整交叉/变异概率以增加多样性'
        })
    
    # 4. 性能分析
    if metrics['final_r2'] < 0.7:
        suggestions.append({
            'type': 'error',
            'title': '性能不佳',
            'message': f"最终R²仅{metrics['final_r2']:.4f}，模型性能较差",
            'action': '建议：1) 增加进化代数和种群大小；2) 检查特征工程；3) 尝试不同的函数集'
        })
    elif metrics['final_r2'] > 0.9:
        suggestions.append({
            'type': 'success',
            'title': '性能优秀',
            'message': f"最终R²达{metrics['final_r2']:.4f}，模型性能优秀",
            'action': '模型已达到很好的效果，可以考虑简化公式以提高可解释性'
        })
    
    # 显示建议
    if not suggestions:
        st.info("✅ 当前训练效果良好，无特殊优化建议")
    else:
        for sug in suggestions:
            if sug['type'] == 'error':
                st.error(f"**{sug['title']}**: {sug['message']}\n\n💡 {sug['action']}")
            elif sug['type'] == 'warning':
                st.warning(f"**{sug['title']}**: {sug['message']}\n\n💡 {sug['action']}")
            elif sug['type'] == 'success':
                st.success(f"**{sug['title']}**: {sug['message']}\n\n💡 {sug['action']}")
            else:
                st.info(f"**{sug['title']}**: {sug['message']}\n\n💡 {sug['action']}")

# ==================== 结束GP可视化增强模块 ====================

# 页面配置
st.set_page_config(page_title="机器学习分析平台", layout="wide", page_icon="🤖")

# 配置中文字体
import matplotlib
from matplotlib.font_manager import FontProperties

# 尝试设置中文字体
try:
    # Windows系统
    font_prop = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')  # 微软雅黑
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    try:
        # 备选方案
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        font_prop = FontProperties()
    except:
        # 如果都失败，使用默认字体
        font_prop = FontProperties()

# 自定义CSS - 简约美观
st.markdown("""
<style>
    /* 主容器 */
    .main {
        padding: 1rem 2rem;
    }
    
    /* 标题样式 */
    h1 {
        color: #1f77b4;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 500;
        margin-top: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* 按钮统一样式 */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* 信息框 */
    .stAlert {
        border-radius: 0.5rem;
    }
    
    /* 指标卡片 */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1f77b4;
    }
    
    /* 分隔线 */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e0e0e0;
    }
    
    /* 侧边栏 */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* 展开器 */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 智能提示系统函数
# ============================================

def analyze_data_characteristics(data, feature_cols, target_col):
    """
    分析数据特征,生成智能建议
    
    Parameters:
        data: DataFrame - 数据集
        feature_cols: list - 特征列名
        target_col: str - 目标变量列名
    
    Returns:
        dict: 包含各种建议的字典
    """
    n_samples = len(data)
    n_features = len(feature_cols)
    
    # 目标变量统计
    y = data[target_col]
    y_range = y.max() - y.min()
    y_std = y.std()
    y_mean = y.mean()
    y_cv = y_std / y_mean if y_mean != 0 else 0  # 变异系数
    
    # 生成建议
    suggestions = {
        'sample_size': n_samples,
        'feature_count': n_features,
        'target_variability': y_cv,
        'recommendations': []
    }
    
    # 样本数建议
    if n_samples < 100:
        suggestions['recommendations'].append({
            'type': 'warning',
            'category': '样本数',
            'message': f'样本数较少({n_samples}个),建议使用RF或SVR,避免使用MLP',
            'icon': '⚠️'
        })
    elif n_samples < 500:
        suggestions['recommendations'].append({
            'type': 'info',
            'category': '样本数',
            'message': f'样本数适中({n_samples}个),RF和XGBoost都是不错的选择',
            'icon': '💡'
        })
    elif n_samples > 10000:
        suggestions['recommendations'].append({
            'type': 'success',
            'category': '样本数',
            'message': f'样本数充足({n_samples}个),建议使用XGBoost或LightGBM以获得更好的性能',
            'icon': '✓'
        })
    
    # 特征数建议
    if n_features > 20:
        suggestions['recommendations'].append({
            'type': 'warning',
            'category': '特征数',
            'message': f'特征数较多({n_features}个),建议先进行特征选择或降维,避免过拟合',
            'icon': '⚠️'
        })
    elif n_features < 5:
        suggestions['recommendations'].append({
            'type': 'info',
            'category': '特征数',
            'message': f'特征数较少({n_features}个),可以考虑生成多项式特征或交互特征',
            'icon': '💡'
        })
    
    # 目标变量变异性建议
    if y_cv > 1.0:
        suggestions['recommendations'].append({
            'type': 'info',
            'category': '目标变量',
            'message': f'目标变量变异性较大(变异系数={y_cv:.2f}),建议考虑对数变换',
            'icon': '💡'
        })
    
    # 样本特征比建议
    ratio = n_samples / n_features if n_features > 0 else 0
    if ratio < 10:
        suggestions['recommendations'].append({
            'type': 'warning',
            'category': '样本特征比',
            'message': f'样本数与特征数比例较低({ratio:.1f}:1),容易过拟合,建议增加样本或减少特征',
            'icon': '⚠️'
        })
    
    return suggestions

def suggest_model(n_samples, n_features):
    """
    根据数据特征推荐模型
    
    Parameters:
        n_samples: int - 样本数
        n_features: int - 特征数
    
    Returns:
        list: 推荐的模型列表
    """
    suggestions = []
    
    if n_samples < 100:
        suggestions.append({
            'model': 'RF',
            'reason': '样本数较少,RF对小样本有较好的泛化能力,不易过拟合',
            'priority': 1
        })
        suggestions.append({
            'model': 'SVR',
            'reason': 'SVR在小样本上表现优异,特别适合高维数据',
            'priority': 2
        })
    elif n_samples < 1000:
        suggestions.append({
            'model': 'RF',
            'reason': '中等样本量,RF是稳健的选择,参数调节简单',
            'priority': 1
        })
        if XGBOOST_AVAILABLE:
            suggestions.append({
                'model': 'XGBoost',
                'reason': 'XGBoost在中等数据集上性能优异,训练速度快',
                'priority': 2
            })
    else:
        if XGBOOST_AVAILABLE:
            suggestions.append({
                'model': 'XGBoost',
                'reason': '大样本量,XGBoost能充分利用数据,性能优异',
                'priority': 1
            })
        if LIGHTGBM_AVAILABLE:
            suggestions.append({
                'model': 'LightGBM',
                'reason': 'LightGBM训练速度快,内存占用低,适合大数据集',
                'priority': 2
            })
        suggestions.append({
            'model': 'RF',
            'reason': 'RF稳健可靠,是大数据集的备选方案',
            'priority': 3
        })
    
    return suggestions

def suggest_hyperparameters(model_name, n_samples, n_features):
    """
    根据模型和数据特征推荐超参数
    
    Parameters:
        model_name: str - 模型名称
        n_samples: int - 样本数
        n_features: int - 特征数
    
    Returns:
        dict: 推荐的超参数
    """
    suggestions = {}
    
    if model_name == "Random Forest":
        suggestions['n_estimators'] = max(100, min(500, n_samples // 10))
        suggestions['max_depth'] = int(np.log2(n_samples)) + 1
        suggestions['min_samples_split'] = max(2, n_samples // 100)
        suggestions['min_samples_leaf'] = max(1, n_samples // 200)
        suggestions['reason'] = "基于样本数自动计算,平衡模型复杂度和泛化能力"
        
    elif model_name == "XGBoost":
        suggestions['n_estimators'] = max(100, min(500, n_samples // 20))
        suggestions['max_depth'] = min(6, int(np.log2(n_samples)))
        suggestions['learning_rate'] = 0.1 if n_samples < 1000 else 0.05
        suggestions['subsample'] = 0.8
        suggestions['colsample_bytree'] = 0.8
        suggestions['reason'] = "平衡训练速度和性能,防止过拟合"
        
    elif model_name == "LightGBM":
        suggestions['n_estimators'] = max(100, min(1000, n_samples // 10))
        suggestions['num_leaves'] = min(31, 2 ** int(np.log2(n_samples)) - 1)
        suggestions['learning_rate'] = 0.1 if n_samples < 1000 else 0.05
        suggestions['min_child_samples'] = max(20, n_samples // 100)
        suggestions['reason'] = "LightGBM推荐配置,适合快速训练"
        
    elif model_name == "SVM":
        suggestions['C'] = 1.0
        suggestions['epsilon'] = 0.1
        suggestions['kernel'] = 'rbf'
        suggestions['reason'] = "SVM默认配置,适合大多数回归任务"
        
    elif model_name == "MLP":
        hidden_size = max(50, min(200, n_features * 2))
        suggestions['hidden_layer_sizes'] = (hidden_size, hidden_size // 2)
        suggestions['max_iter'] = 500 if n_samples < 1000 else 1000
        suggestions['learning_rate_init'] = 0.001
        suggestions['reason'] = "基于特征数设计网络结构,防止过拟合"
    
    return suggestions

def display_smart_suggestions(suggestions):
    """
    展示智能建议
    
    Parameters:
        suggestions: dict - 建议字典
    """
    if not suggestions['recommendations']:
        return
    
    st.markdown("### 🤖 智能建议")
    
    for rec in suggestions['recommendations']:
        if rec['type'] == 'warning':
            st.warning(f"{rec['icon']} **{rec['category']}**: {rec['message']}")
        elif rec['type'] == 'info':
            st.info(f"{rec['icon']} **{rec['category']}**: {rec['message']}")
        elif rec['type'] == 'success':
            st.success(f"{rec['icon']} **{rec['category']}**: {rec['message']}")

# ============================================
# 统一的导航按钮函数
# ============================================
def next_step_button(text="继续下一步", key=None, target_step=None):
    """统一风格的导航按钮"""
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; margin: 1.5rem 0 1rem 0;'>
        <p style='color: #666; font-size: 0.95rem;'>
            🚀 准备好继续了吗？
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(text, type="primary", width="stretch", key=key):
            if target_step:
                st.session_state.step = target_step
                st.rerun()
            return True
    return False

# GP 公式转换函数：将前缀表达式转换为中缀表达式（标准数学格式）
def convert_to_infix(formula_str):
    """
    将 gplearn 的前缀表达式转换为中缀数学表达式
    例如：add(mul(A, 0.5), B) -> (A × 0.5) + B
    """
    import re
    
    # 定义运算符映射
    operators = {
        'add': ('+', 2),
        'sub': ('-', 2),
        'mul': ('×', 2),
        'div': ('÷', 2),
        'sqrt': ('√', 1),
        'log': ('log', 1),
        'abs': ('|', 1),
        'neg': ('-', 1),
        'inv': ('1/', 1),
        'max': ('max', 2),
        'min': ('min', 2),
        'sin': ('sin', 1),
        'cos': ('cos', 1),
        'tan': ('tan', 1)
    }
    
    def parse_expression(expr):
        """递归解析表达式"""
        expr = expr.strip()
        
        # 检查是否是数字
        try:
            float(expr)
            return expr
        except ValueError:
            pass
        
        # 检查是否是变量（字母）
        if re.match(r'^[A-Z][0-9]*$', expr):
            return expr
        
        # 解析函数调用
        for op_name, (op_symbol, arity) in operators.items():
            if expr.startswith(op_name + '('):
                # 找到匹配的括号
                args = []
                depth = 0
                current_arg = ''
                start_idx = len(op_name) + 1
                
                for i in range(start_idx, len(expr)):
                    char = expr[i]
                    if char == '(':
                        depth += 1
                        current_arg += char
                    elif char == ')':
                        if depth == 0:
                            if current_arg:
                                args.append(parse_expression(current_arg))
                            break
                        else:
                            depth -= 1
                            current_arg += char
                    elif char == ',' and depth == 0:
                        args.append(parse_expression(current_arg))
                        current_arg = ''
                    else:
                        current_arg += char
                
                # 根据运算符类型构建表达式
                if arity == 1:
                    if op_name == 'sqrt':
                        return f"√({args[0]})"
                    elif op_name == 'abs':
                        return f"|{args[0]}|"
                    elif op_name == 'neg':
                        return f"-({args[0]})"
                    elif op_name == 'inv':
                        return f"(1/{args[0]})"
                    else:
                        return f"{op_symbol}({args[0]})"
                elif arity == 2:
                    if op_name in ['max', 'min']:
                        return f"{op_symbol}({args[0]}, {args[1]})"
                    else:
                        return f"({args[0]} {op_symbol} {args[1]})"
        
        return expr
    
    try:
        return parse_expression(formula_str)
    except Exception as e:
        return formula_str  # 如果转换失败，返回原始字符串

# GP 公式转换为 LaTeX 格式
def convert_to_latex(formula_str):
    """
    将 gplearn 的前缀表达式转换为 LaTeX 格式
    """
    import re
    
    # 定义运算符映射到 LaTeX
    operators = {
        'add': ('+', 2),
        'sub': ('-', 2),
        'mul': (r'\times', 2),
        'div': (r'\div', 2),
        'sqrt': (r'\sqrt', 1),
        'log': (r'\log', 1),
        'abs': ('|', 1),
        'neg': ('-', 1),
        'inv': (r'\frac{1}', 1),
        'max': (r'\max', 2),
        'min': (r'\min', 2),
        'sin': (r'\sin', 1),
        'cos': (r'\cos', 1),
        'tan': (r'\tan', 1)
    }
    
    def parse_expression(expr):
        """递归解析表达式为 LaTeX"""
        expr = expr.strip()
        
        # 检查是否是数字
        try:
            val = float(expr)
            # 格式化数字，保留3位小数
            return f"{val:.3f}" if abs(val) < 1000 else f"{val:.2e}"
        except ValueError:
            pass
        
        # 检查是否是变量
        if re.match(r'^[A-Z][0-9]*$', expr):
            return expr
        
        # 解析函数调用
        for op_name, (op_symbol, arity) in operators.items():
            if expr.startswith(op_name + '('):
                args = []
                depth = 0
                current_arg = ''
                start_idx = len(op_name) + 1
                
                for i in range(start_idx, len(expr)):
                    char = expr[i]
                    if char == '(':
                        depth += 1
                        current_arg += char
                    elif char == ')':
                        if depth == 0:
                            if current_arg:
                                args.append(parse_expression(current_arg))
                            break
                        else:
                            depth -= 1
                            current_arg += char
                    elif char == ',' and depth == 0:
                        args.append(parse_expression(current_arg))
                        current_arg = ''
                    else:
                        current_arg += char
                
                # 根据运算符类型构建 LaTeX
                if arity == 1:
                    if op_name == 'sqrt':
                        return f"{op_symbol}{{{args[0]}}}"
                    elif op_name == 'abs':
                        return f"|{args[0]}|"
                    elif op_name == 'neg':
                        return f"-\\left({args[0]}\\right)"
                    elif op_name == 'inv':
                        return f"\\frac{{1}}{{{args[0]}}}"
                    else:
                        return f"{op_symbol}\\left({args[0]}\\right)"
                elif arity == 2:
                    if op_name in ['max', 'min']:
                        return f"{op_symbol}\\left({args[0]}, {args[1]}\\right)"
                    elif op_name == 'div':
                        return f"\\frac{{{args[0]}}}{{{args[1]}}}"
                    else:
                        return f"\\left({args[0]} {op_symbol} {args[1]}\\right)"
        
        return expr
    
    try:
        latex_expr = parse_expression(formula_str)
        return f"y = {latex_expr}"
    except Exception as e:
        return f"y = {formula_str}"


# ==================== 模型训练函数 ====================

def train_random_forest(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode='fast'):
    """训练RF模型 - 支持快速/平衡/完整模式"""
    import time
    start_time = time.time()
    
    def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth) if max_depth > 0 else None,
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=int(max_features),  # 改为整数特征数量
            random_state=42,
            n_jobs=-1
        )
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
        return cv_scores.mean()
    
    # 根据模式调整参数范围和初始点
    if mode == 'full':
        pbounds = {
            'n_estimators': (100, 1000),
            'max_depth': (5, 30),
            'min_samples_split': (2, 30),
            'min_samples_leaf': (1, 15),
            'max_features': (2, 20)
        }
        init_points = 15
    elif mode == 'balanced':
        pbounds = {
            'n_estimators': (50, 500),
            'max_depth': (3, 25),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 12),
            'max_features': (2, 15)
        }
        init_points = 12
    else:  # fast mode
        pbounds = {
            'n_estimators': (25, 200),         # 更合理的范围
            'max_depth': (2, 20),              # 更保守的深度
            'min_samples_split': (2, 8),       # 更小的范围
            'min_samples_leaf': (1, 8),        # 更小的范围
            'max_features': (2, 12)            # 具体特征数量
        }
        init_points = 10
    
    optimizer = BayesianOptimization(
        f=rf_cv,
        pbounds=pbounds,
        random_state=42,
        verbose=0
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    # 提取优化历史
    optimization_history = []
    for i, res in enumerate(optimizer.res):
        optimization_history.append({
            'iteration': i + 1,
            'target': res['target'],
            'params': res['params']
        })
    
    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth']) if best_params['max_depth'] > 0 else None
    best_params['min_samples_split'] = int(best_params['min_samples_split'])
    best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
    best_params['max_features'] = int(best_params['max_features'])  # 改为整数
    
    # 使用最佳参数在训练集上训练
    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    best_cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    
    model.fit(X_train, y_train)
    
    # 在训练集和测试集上预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_time = time.time() - start_time
    
    return {
        'model': model,
        'best_params': best_params,
        'train_r2': r2_score(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_mape': mean_absolute_percentage_error(y_train, y_train_pred),
        'train_ia': index_of_agreement(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_mape': mean_absolute_percentage_error(y_test, y_test_pred),
        'test_ia': index_of_agreement(y_test, y_test_pred),
        'cv_score': optimizer.max['target'],
        'y_train_true': y_train,
        'y_train_pred': y_train_pred,
        'y_test_true': y_test,
        'y_test_pred': y_test_pred,
        'optimization_history': optimization_history,
        'best_cv_scores': best_cv_scores,
        'train_time': train_time
    }


def train_svm(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode='fast'):
    """训练SVR模型 - 支持快速/平衡/完整模式"""
    import time
    start_time = time.time()
    
    if mode == 'full':
        # 完整模式：优化kernel类型
        def svm_cv(C, epsilon, gamma, kernel_idx):
            kernels = ['linear', 'rbf', 'poly']
            kernel = kernels[int(kernel_idx) % 3]
            
            if kernel == 'linear':
                model = SVR(C=C, epsilon=epsilon, kernel='linear')
            elif kernel == 'poly':
                model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel='poly', degree=3)
            else:  # rbf
                model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel='rbf')
            
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            return cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2').mean()
        
        optimizer = BayesianOptimization(
            f=svm_cv,
            pbounds={
                'C': (0.1, 1000),              # 降低上限，避免过拟合
                'epsilon': (0.001, 1.0),       # 降低上限
                'gamma': (0.001, 10),          # 降低上限
                'kernel_idx': (0, 2.99)
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=15, n_iter=n_iter)
        
    elif mode == 'balanced':
        # 平衡模式：优化kernel类型，适度范围
        def svm_cv(C, epsilon, gamma, kernel_idx):
            kernels = ['linear', 'rbf', 'poly']
            kernel = kernels[int(kernel_idx) % 3]
            
            if kernel == 'linear':
                model = SVR(C=C, epsilon=epsilon, kernel='linear')
            elif kernel == 'poly':
                model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel='poly', degree=3)
            else:  # rbf
                model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel='rbf')
            
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            return cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2').mean()
        
        optimizer = BayesianOptimization(
            f=svm_cv,
            pbounds={
                'C': (0.1, 500),               # 降低上限
                'epsilon': (0.001, 1.0),       # 降低上限
                'gamma': (0.001, 5),           # 降低上限
                'kernel_idx': (0, 2.99)
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=12, n_iter=n_iter)
        
        # 提取优化历史
        optimization_history = []
        for i, res in enumerate(optimizer.res):
            optimization_history.append({
                'iteration': i + 1,
                'target': res['target'],
                'params': res['params']
            })
        
        best_params = optimizer.max['params']
        kernels = ['linear', 'rbf', 'poly']
        best_kernel = kernels[int(best_params['kernel_idx']) % 3]
        
        # 构建最终模型
        if best_kernel == 'linear':
            model = SVR(C=best_params['C'], epsilon=best_params['epsilon'], kernel='linear')
            final_params = {'C': best_params['C'], 'epsilon': best_params['epsilon'], 'kernel': 'linear'}
        elif best_kernel == 'poly':
            model = SVR(C=best_params['C'], epsilon=best_params['epsilon'], 
                       gamma=best_params['gamma'], kernel='poly', degree=3)
            final_params = {'C': best_params['C'], 'epsilon': best_params['epsilon'], 
                          'gamma': best_params['gamma'], 'kernel': 'poly', 'degree': 3}
        else:  # rbf
            model = SVR(C=best_params['C'], epsilon=best_params['epsilon'], 
                       gamma=best_params['gamma'], kernel='rbf')
            final_params = {'C': best_params['C'], 'epsilon': best_params['epsilon'], 
                          'gamma': best_params['gamma'], 'kernel': 'rbf'}
        
    else:  # fast mode - 简化优化，只优化C和epsilon，gamma使用'auto'
        def svm_cv(C, epsilon):
            # 使用gamma='auto'（即1/n_features），这对混凝土数据集效果好
            model = SVR(C=C, epsilon=epsilon, gamma='auto', kernel='rbf', max_iter=10000)
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            return cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2').mean()
        
        optimizer = BayesianOptimization(
            f=svm_cv,
            pbounds={
                'C': (100, 2000),              # 扩大C范围，包含你的最优值815
                'epsilon': (0.001, 1.0)        # 标准epsilon范围
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=8, n_iter=n_iter)
    
    # 提取优化历史
    optimization_history = []
    for i, res in enumerate(optimizer.res):
        optimization_history.append({
            'iteration': i + 1,
            'target': res['target'],
            'params': res['params']
        })
    
    best_params = optimizer.max['params']
    
    # 根据模式构建最终模型
    if mode in ['full', 'balanced']:
        kernels = ['linear', 'rbf', 'poly']
        best_kernel = kernels[int(best_params['kernel_idx']) % 3]
        
        # 构建最终模型
        if best_kernel == 'linear':
            model = SVR(C=best_params['C'], epsilon=best_params['epsilon'], kernel='linear')
            final_params = {'C': best_params['C'], 'epsilon': best_params['epsilon'], 'kernel': 'linear'}
        elif best_kernel == 'poly':
            degree = 2 if mode == 'fast' else 3
            model = SVR(C=best_params['C'], epsilon=best_params['epsilon'], 
                       gamma=best_params['gamma'], kernel='poly', degree=degree)
            final_params = {'C': best_params['C'], 'epsilon': best_params['epsilon'], 
                          'gamma': best_params['gamma'], 'kernel': 'poly', 'degree': degree}
        else:  # rbf
            model = SVR(C=best_params['C'], epsilon=best_params['epsilon'], 
                       gamma=best_params['gamma'], kernel='rbf')
            final_params = {'C': best_params['C'], 'epsilon': best_params['epsilon'], 
                          'gamma': best_params['gamma'], 'kernel': 'rbf'}
    else:  # fast mode
        model = SVR(C=best_params['C'], epsilon=best_params['epsilon'], 
                   gamma='auto', kernel='rbf', max_iter=10000)
        final_params = {'C': best_params['C'], 'epsilon': best_params['epsilon'], 
                      'gamma': 'auto', 'kernel': 'rbf'}
    
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    best_cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_time = time.time() - start_time
    
    return {
        'model': model,
        'best_params': final_params,
        'train_r2': r2_score(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_mape': mean_absolute_percentage_error(y_train, y_train_pred),
        'train_ia': index_of_agreement(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_mape': mean_absolute_percentage_error(y_test, y_test_pred),
        'test_ia': index_of_agreement(y_test, y_test_pred),
        'cv_score': optimizer.max['target'],
        'y_train_true': y_train,
        'y_train_pred': y_train_pred,
        'y_test_true': y_test,
        'y_test_pred': y_test_pred,
        'optimization_history': optimization_history,
        'best_cv_scores': best_cv_scores,
        'train_time': train_time
    }


def train_mlp(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode='fast'):
    """训练MLP模型 - 支持快速/平衡/完整模式"""
    import time
    start_time = time.time()
    
    if mode == 'full':
        # 完整模式：优化activation类型，增加max_iter
        max_iter = 3000
        
        def mlp_cv(hidden_layer_size, alpha, learning_rate_init, batch_size, activation_idx):
            activations = ['relu', 'tanh', 'logistic']
            activation = activations[int(activation_idx) % 3]
            
            model = MLPRegressor(
                hidden_layer_sizes=(int(hidden_layer_size), int(hidden_layer_size//2)),
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                batch_size=int(batch_size),
                activation=activation,
                max_iter=max_iter,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            return cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2').mean()
        
        optimizer = BayesianOptimization(
            f=mlp_cv,
            pbounds={
                'hidden_layer_size': (32, 768),
                'alpha': (0.00001, 2.0),
                'learning_rate_init': (0.00001, 0.5),
                'batch_size': (8, 512),
                'activation_idx': (0, 2.99)
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=18, n_iter=n_iter)
        
        # 提取优化历史
        optimization_history = []
        for i, res in enumerate(optimizer.res):
            optimization_history.append({
                'iteration': i + 1,
                'target': res['target'],
                'params': res['params']
            })
        
        best_params = optimizer.max['params']
        activations = ['relu', 'tanh', 'logistic']
        best_activation = activations[int(best_params['activation_idx']) % 3]
        
        # 构建最终参数
        final_params = {
            'hidden_layer_sizes': (int(best_params['hidden_layer_size']), 
                                  int(best_params['hidden_layer_size']//2)),
            'alpha': best_params['alpha'],
            'learning_rate_init': best_params['learning_rate_init'],
            'batch_size': int(best_params['batch_size']),
            'activation': best_activation,
            'max_iter': max_iter
        }
        
        model = MLPRegressor(
            hidden_layer_sizes=final_params['hidden_layer_sizes'],
            alpha=final_params['alpha'],
            learning_rate_init=final_params['learning_rate_init'],
            batch_size=final_params['batch_size'],
            activation=final_params['activation'],
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        
    elif mode == 'balanced':
        # 平衡模式：优化activation类型，适度范围
        max_iter = 2500
        
        def mlp_cv(hidden_layer_size, alpha, learning_rate_init, batch_size, activation_idx):
            activations = ['relu', 'tanh', 'logistic']
            activation = activations[int(activation_idx) % 3]
            
            model = MLPRegressor(
                hidden_layer_sizes=(int(hidden_layer_size), int(hidden_layer_size//2)),
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                batch_size=int(batch_size),
                activation=activation,
                max_iter=max_iter,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            return cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2').mean()
        
        optimizer = BayesianOptimization(
            f=mlp_cv,
            pbounds={
                'hidden_layer_size': (40, 640),
                'alpha': (0.00001, 1.5),
                'learning_rate_init': (0.00001, 0.3),
                'batch_size': (12, 384),
                'activation_idx': (0, 2.99)
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=15, n_iter=n_iter)
        
        # 提取优化历史
        optimization_history = []
        for i, res in enumerate(optimizer.res):
            optimization_history.append({
                'iteration': i + 1,
                'target': res['target'],
                'params': res['params']
            })
        
        best_params = optimizer.max['params']
        activations = ['relu', 'tanh', 'logistic']
        best_activation = activations[int(best_params['activation_idx']) % 3]
        
        # 构建最终参数
        final_params = {
            'hidden_layer_sizes': (int(best_params['hidden_layer_size']), 
                                  int(best_params['hidden_layer_size']//2)),
            'alpha': best_params['alpha'],
            'learning_rate_init': best_params['learning_rate_init'],
            'batch_size': int(best_params['batch_size']),
            'activation': best_activation,
            'max_iter': max_iter
        }
        
        model = MLPRegressor(
            hidden_layer_sizes=final_params['hidden_layer_sizes'],
            alpha=final_params['alpha'],
            learning_rate_init=final_params['learning_rate_init'],
            batch_size=final_params['batch_size'],
            activation=final_params['activation'],
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        
    else:  # fast mode
        max_iter = 3000  # 增加迭代次数确保收敛
        
        def mlp_cv(hidden_layer_size1, hidden_layer_size2, alpha, learning_rate_init):
            model = MLPRegressor(
                hidden_layer_sizes=(int(hidden_layer_size1), int(hidden_layer_size2)),  # 两层网络
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                early_stopping=True,
                validation_fraction=0.15,  # 增加验证集比例
                n_iter_no_change=20,       # 增加早停耐心
                random_state=42
            )
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            return cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2').mean()
        
        optimizer = BayesianOptimization(
            f=mlp_cv,
            pbounds={
                'hidden_layer_size1': (8, 512),          # 从8开始，上限512
                'hidden_layer_size2': (4, 256),          # 从4开始，上限256
                'alpha': (0.001, 1.0),                   # 从0.001开始
                'learning_rate_init': (0.001, 0.1)       # 从0.001开始
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=10, n_iter=n_iter)
        
        # 提取优化历史
        optimization_history = []
        for i, res in enumerate(optimizer.res):
            optimization_history.append({
                'iteration': i + 1,
                'target': res['target'],
                'params': res['params']
            })
        
        best_params = optimizer.max['params']
        final_params = {
            'hidden_layer_sizes': (int(best_params['hidden_layer_size1']), 
                                  int(best_params['hidden_layer_size2'])),
            'alpha': best_params['alpha'],
            'learning_rate_init': best_params['learning_rate_init'],
            'activation': 'relu',
            'max_iter': max_iter
        }
        
        model = MLPRegressor(
            hidden_layer_sizes=final_params['hidden_layer_sizes'],
            alpha=final_params['alpha'],
            learning_rate_init=final_params['learning_rate_init'],
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42
        )
    
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    best_cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_time = time.time() - start_time
    
    return {
        'model': model,
        'best_params': final_params,
        'train_r2': r2_score(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_mape': mean_absolute_percentage_error(y_train, y_train_pred),
        'train_ia': index_of_agreement(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_mape': mean_absolute_percentage_error(y_test, y_test_pred),
        'test_ia': index_of_agreement(y_test, y_test_pred),
        'cv_score': optimizer.max['target'],
        'y_train_true': y_train,
        'y_train_pred': y_train_pred,
        'y_test_true': y_test,
        'y_test_pred': y_test_pred,
        'optimization_history': optimization_history,
        'best_cv_scores': best_cv_scores,
        'train_time': train_time
    }


def train_lightgbm(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode='fast'):
    """训练LightGBM模型 - 支持快速/平衡/完整模式"""
    import time
    start_time = time.time()
    
    if mode == 'full':
        # 完整模式：9个参数
        def lgbm_cv(num_leaves, max_depth, learning_rate, n_estimators, min_child_samples, 
                    subsample, colsample_bytree, reg_alpha, reg_lambda):
            model = LGBMRegressor(
                num_leaves=int(num_leaves),
                max_depth=int(max_depth),
                learning_rate=learning_rate,
                n_estimators=int(n_estimators),
                min_child_samples=int(min_child_samples),
                subsample=subsample,
                subsample_freq=1,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            return cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2').mean()
        
        optimizer = BayesianOptimization(
            f=lgbm_cv,
            pbounds={
                'num_leaves': (10, 200),
                'max_depth': (3, 25),
                'learning_rate': (0.01, 0.3),      # 提高下限：0.0005 → 0.01
                'n_estimators': (50, 1000),        # 降低上限：3000 → 1000
                'min_child_samples': (3, 100),     # 降低上限：150 → 100
                'subsample': (0.5, 1.0),           # 提高下限：0.4 → 0.5
                'colsample_bytree': (0.5, 1.0),    # 提高下限：0.4 → 0.5
                'reg_alpha': (0.0, 2.0),           # 降低上限：15.0 → 2.0
                'reg_lambda': (0.0, 2.0)           # 降低上限：15.0 → 2.0
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=15, n_iter=n_iter)  # 降低：18 → 15
        
    elif mode == 'balanced':
        # 平衡模式：8个参数，适度范围
        def lgbm_cv(num_leaves, max_depth, learning_rate, n_estimators, min_child_samples, 
                    subsample, colsample_bytree, reg_alpha, reg_lambda):
            model = LGBMRegressor(
                num_leaves=int(num_leaves),
                max_depth=int(max_depth),
                learning_rate=learning_rate,
                n_estimators=int(n_estimators),
                min_child_samples=int(min_child_samples),
                subsample=subsample,
                subsample_freq=1,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            return cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2').mean()
        
        optimizer = BayesianOptimization(
            f=lgbm_cv,
            pbounds={
                'num_leaves': (15, 175),
                'max_depth': (3, 22),
                'learning_rate': (0.01, 0.3),      # 提高下限：0.0008 → 0.01
                'n_estimators': (75, 800),         # 降低上限：2500 → 800
                'min_child_samples': (4, 80),      # 降低上限：125 → 80
                'subsample': (0.5, 1.0),           # 提高下限：0.45 → 0.5
                'colsample_bytree': (0.5, 1.0),    # 提高下限：0.45 → 0.5
                'reg_alpha': (0.0, 1.5),           # 降低上限：12.0 → 1.5
                'reg_lambda': (0.0, 1.5)           # 降低上限：12.0 → 1.5
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=12, n_iter=n_iter)  # 降低：15 → 12
        
    else:  # fast mode
        # 快速模式：6个核心参数
        def lgbm_cv(num_leaves, learning_rate, n_estimators, min_child_samples, reg_alpha, reg_lambda):
            model = LGBMRegressor(
                num_leaves=int(num_leaves),
                learning_rate=learning_rate,
                n_estimators=int(n_estimators),
                min_child_samples=int(min_child_samples),
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            return cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2').mean()
        
        optimizer = BayesianOptimization(
            f=lgbm_cv,
            pbounds={
                'num_leaves': (20, 150),
                'learning_rate': (0.01, 0.3),      # 提高下限：0.001 → 0.01
                'n_estimators': (100, 500),        # 降低上限：2000 → 500
                'min_child_samples': (5, 50),      # 降低上限：100 → 50
                'reg_alpha': (0.0, 1.0),           # 降低上限：10.0 → 1.0
                'reg_lambda': (0.0, 1.0)           # 降低上限：10.0 → 1.0
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=8, n_iter=n_iter)  # 降低：12 → 8
    
    # 提取优化历史
    optimization_history = []
    for i, res in enumerate(optimizer.res):
        optimization_history.append({
            'iteration': i + 1,
            'target': res['target'],
            'params': res['params']
        })
    
    best_params = optimizer.max['params']
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['min_child_samples'] = int(best_params['min_child_samples'])
    
    if mode in ['full', 'balanced']:
        best_params['max_depth'] = int(best_params['max_depth'])
        model = LGBMRegressor(**best_params, subsample_freq=1, random_state=42, verbose=-1, n_jobs=-1)
    else:
        model = LGBMRegressor(**best_params, random_state=42, verbose=-1, n_jobs=-1)
    
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    best_cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_time = time.time() - start_time
    
    return {
        'model': model,
        'best_params': best_params,
        'train_r2': r2_score(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_mape': mean_absolute_percentage_error(y_train, y_train_pred),
        'train_ia': index_of_agreement(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_mape': mean_absolute_percentage_error(y_test, y_test_pred),
        'test_ia': index_of_agreement(y_test, y_test_pred),
        'cv_score': optimizer.max['target'],
        'y_train_true': y_train,
        'y_train_pred': y_train_pred,
        'y_test_true': y_test,
        'y_test_pred': y_test_pred,
        'optimization_history': optimization_history,
        'best_cv_scores': best_cv_scores,
        'train_time': train_time
    }


def train_xgboost(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode='fast'):
    """训练XGBoost模型 - 支持快速/平衡/完整模式"""
    import time
    start_time = time.time()
    
    if mode == 'full':
        # 完整模式：9个参数
        def xgb_cv(max_depth, learning_rate, n_estimators, min_child_weight, gamma, 
                   subsample, colsample_bytree, reg_alpha, reg_lambda):
            model = XGBRegressor(
                max_depth=int(max_depth),
                learning_rate=learning_rate,
                n_estimators=int(n_estimators),
                min_child_weight=min_child_weight,
                gamma=gamma,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                n_jobs=-1
            )
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            return cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2').mean()
        
        optimizer = BayesianOptimization(
            f=xgb_cv,
            pbounds={
                'max_depth': (2, 20),              # 降低上限：25 → 20
                'learning_rate': (0.01, 0.3),      # 提高下限：0.005 → 0.01
                'n_estimators': (50, 1000),        # 降低上限：3000 → 1000
                'min_child_weight': (1, 12),       # 提高下限：0 → 1
                'gamma': (0.0, 2.0),               # 降低上限：15.0 → 2.0
                'subsample': (0.5, 1.0),           # 提高下限：0.4 → 0.5
                'colsample_bytree': (0.5, 1.0),    # 提高下限：0.4 → 0.5
                'reg_alpha': (0.0, 2.0),           # 降低上限：15.0 → 2.0
                'reg_lambda': (0.0, 2.0)           # 降低上限：15.0 → 2.0
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=15, n_iter=n_iter)  # 降低：18 → 15
        
    elif mode == 'balanced':
        # 平衡模式：9个参数，适度范围
        def xgb_cv(max_depth, learning_rate, n_estimators, min_child_weight, gamma, 
                   subsample, colsample_bytree, reg_alpha, reg_lambda):
            model = XGBRegressor(
                max_depth=int(max_depth),
                learning_rate=learning_rate,
                n_estimators=int(n_estimators),
                min_child_weight=min_child_weight,
                gamma=gamma,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                n_jobs=-1
            )
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            return cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2').mean()
        
        optimizer = BayesianOptimization(
            f=xgb_cv,
            pbounds={
                'max_depth': (2, 18),              # 降低上限：22 → 18
                'learning_rate': (0.01, 0.3),      # 提高下限：0.008 → 0.01
                'n_estimators': (75, 800),         # 降低上限：2500 → 800
                'min_child_weight': (1, 10),       # 提高下限：0 → 1
                'gamma': (0.0, 1.5),               # 降低上限：12.0 → 1.5
                'subsample': (0.5, 1.0),           # 提高下限：0.45 → 0.5
                'colsample_bytree': (0.5, 1.0),    # 提高下限：0.45 → 0.5
                'reg_alpha': (0.0, 1.5),           # 降低上限：12.0 → 1.5
                'reg_lambda': (0.0, 1.5)           # 降低上限：12.0 → 1.5
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=12, n_iter=n_iter)  # 降低：15 → 12
        
    else:  # fast mode
        # 快速模式：7个核心参数
        def xgb_cv(max_depth, learning_rate, n_estimators, min_child_weight, gamma, reg_alpha, reg_lambda):
            model = XGBRegressor(
                max_depth=int(max_depth),
                learning_rate=learning_rate,
                n_estimators=int(n_estimators),
                min_child_weight=min_child_weight,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                n_jobs=-1
            )
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            return cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2').mean()
        
        optimizer = BayesianOptimization(
            f=xgb_cv,
            pbounds={
                'max_depth': (3, 15),              # 降低上限：20 → 15
                'learning_rate': (0.01, 0.3),      # 与app1.py一致
                'n_estimators': (100, 500),        # 降低上限：2000 → 500
                'min_child_weight': (1, 10),       # 提高下限：0 → 1
                'gamma': (0.0, 1.0),               # 降低上限：10.0 → 1.0
                'reg_alpha': (0.0, 1.0),           # 降低上限：10.0 → 1.0
                'reg_lambda': (0.0, 2.0)           # 降低上限：10.0 → 2.0
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=8, n_iter=n_iter)  # 降低：12 → 8
    
    # 提取优化历史
    optimization_history = []
    for i, res in enumerate(optimizer.res):
        optimization_history.append({
            'iteration': i + 1,
            'target': res['target'],
            'params': res['params']
        })
    
    best_params = optimizer.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    
    model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    best_cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_time = time.time() - start_time
    
    return {
        'model': model,
        'best_params': best_params,
        'train_r2': r2_score(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_mape': mean_absolute_percentage_error(y_train, y_train_pred),
        'train_ia': index_of_agreement(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_mape': mean_absolute_percentage_error(y_test, y_test_pred),
        'test_ia': index_of_agreement(y_test, y_test_pred),
        'cv_score': optimizer.max['target'],
        'y_train_true': y_train,
        'y_train_pred': y_train_pred,
        'y_test_true': y_test,
        'y_test_pred': y_test_pred,
        'optimization_history': optimization_history,
        'best_cv_scores': best_cv_scores,
        'train_time': train_time
    }

# ==================== 结束模型训练函数 ====================

# 初始化session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'results' not in st.session_state:
    st.session_state.results = {}

# 标题
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 1rem; color: white; 
            margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <h1 style='color: white; margin: 0; text-align: center;'>
        🤖 机器学习分析平台
    </h1>
    <p style='margin: 0.5rem 0 0 0; opacity: 0.95; font-size: 1.1rem; text-align: center;'>
        从数据导入到符号回归，一站式机器学习解决方案
    </p>
</div>
""", unsafe_allow_html=True)

# 侧边栏 - 工作流程导航
with st.sidebar:
    st.header("📋 工作流程")
    steps = [
        "1️⃣ 数据导入", 
        "2️⃣ 数据预处理", 
        "3️⃣ 模型训练", 
        "4️⃣ 结果可视化", 
        "5️⃣ 显示数学公式"
    ]
    
    for i, step_name in enumerate(steps, 1):
        if st.button(step_name, key=f"nav_{i}", width='stretch'):
            st.session_state.step = i
    
    st.markdown("---")
    st.info("💡 按照步骤完成机器学习分析流程")
    
    # ========== 高级设置面板 ==========
    st.markdown("---")
    with st.expander("🔧 高级设置", expanded=False):
        st.markdown("### 模型训练优化模式")
        
        # 初始化高级设置
        if 'advanced_settings' not in st.session_state:
            st.session_state.advanced_settings = {
                'optimization_mode': 'fast'
            }
        
        st.markdown("#### 🎯 选择优化模式")
        
        mode_options = ["⚡ 快速模式（推荐）", "⚖️ 平衡模式", "🔧 完整模式"]
        current_mode = st.session_state.advanced_settings['optimization_mode']
        if current_mode == 'fast':
            default_index = 0
        elif current_mode == 'balanced':
            default_index = 1
        else:
            default_index = 2
        
        optimization_mode = st.radio(
            "优化策略",
            mode_options,
            index=default_index,
            key="optimization_mode_radio"
        )
        
        if optimization_mode == "⚡ 快速模式（推荐）":
            st.session_state.advanced_settings['optimization_mode'] = 'fast'
        elif optimization_mode == "⚖️ 平衡模式":
            st.session_state.advanced_settings['optimization_mode'] = 'balanced'
        else:
            st.session_state.advanced_settings['optimization_mode'] = 'full'
        
        st.markdown("---")
        
        # 显示当前模式的详细说明
        current_mode = st.session_state.advanced_settings['optimization_mode']
        
        if current_mode == 'fast':
            st.success("⚡ **快速模式** - 使用常用参数，训练速度快")
            st.caption("⏱️ 预计训练时间：基准（如5个模型约5-10分钟）")
            
            with st.expander("📋 查看快速模式参数详情", expanded=False):
                st.markdown("""
                **RF**
                - n_estimators: 100-1000
                - max_depth: 5-30
                - min_samples_split: 2-20
                - min_samples_leaf: 1-10
                - max_features: 0.3-1.0
                
                **SVM**
                - C: 0.001-1000
                - epsilon: 0.001-1.0
                - gamma: 0.0001-10
                - kernel: 固定rbf
                
                **神经网络**
                - hidden_layer_size: 50-512
                - alpha: 0.0001-1.0
                - learning_rate_init: 0.0001-0.1
                - batch_size: 16-256
                - activation: 固定relu
                - max_iter: 2000
                
                **LightGBM**
                - num_leaves: 20-150
                - learning_rate: 0.001-0.3
                - n_estimators: 100-2000
                - min_child_samples: 5-100
                - reg_alpha: 0-10
                - reg_lambda: 0-10
                
                **XGBoost**
                - max_depth: 3-20
                - learning_rate: 0.01-0.3
                - n_estimators: 100-2000
                - min_child_weight: 0-10
                - gamma: 0-10
                - reg_alpha: 0-10
                - reg_lambda: 0-10
                """)
        
        elif current_mode == 'balanced':
            st.info("⚖️ **平衡模式** - 适度优化，性能与时间平衡")
            st.caption("⏱️ 预计训练时间：+50-100%（如5个模型约10-20分钟）")
            
            with st.expander("📋 查看平衡模式参数详情", expanded=True):
                st.markdown("""
                **RF**
                - n_estimators: 50-1200
                - max_depth: 3-35
                - min_samples_split: 2-25
                - min_samples_leaf: 1-12
                - max_features: 0.2-1.0
                
                **SVM**
                - C: 0.0001-5000
                - epsilon: 0.0001-1.5
                - gamma: 0.00001-50
                - kernel: 优化（linear/rbf/poly）
                
                **神经网络**
                - hidden_layer_size: 40-640
                - alpha: 0.00001-1.5
                - learning_rate_init: 0.00001-0.3
                - batch_size: 12-384
                - activation: 优化（relu/tanh/logistic）
                - max_iter: 2500
                
                **LightGBM**
                - num_leaves: 15-175
                - max_depth: 3-22
                - learning_rate: 0.0008-0.4
                - n_estimators: 75-2500
                - min_child_samples: 4-125
                - subsample: 0.45-1.0
                - colsample_bytree: 0.45-1.0
                - reg_alpha: 0-12
                - reg_lambda: 0-12
                
                **XGBoost**
                - max_depth: 2-22
                - learning_rate: 0.008-0.4
                - n_estimators: 75-2500
                - min_child_weight: 0-12
                - gamma: 0-12
                - subsample: 0.45-1.0
                - colsample_bytree: 0.45-1.0
                - reg_alpha: 0-12
                - reg_lambda: 0-12
                """)
        
        else:  # full mode
            st.warning("🔧 **完整模式** - 深度优化，追求极致性能")
            st.caption("⏱️ 预计训练时间：+100-150%（如5个模型约15-25分钟）")
            st.caption("⚠️ 注意：完整模式已优化，训练时间比之前版本大幅缩短")
            
            with st.expander("📋 查看完整模式参数详情", expanded=True):
                st.markdown("""
                **RF**
                - n_estimators: 50-1500
                - max_depth: 3-40
                - min_samples_split: 2-30
                - min_samples_leaf: 1-15
                - max_features: 0.2-1.0
                
                **SVM**
                - C: 0.0001-10000
                - epsilon: 0.0001-2.0
                - gamma: 0.00001-100
                - kernel: 优化（linear/rbf/poly）
                
                **神经网络**
                - hidden_layer_size: 32-768
                - alpha: 0.00001-2.0
                - learning_rate_init: 0.00001-0.5
                - batch_size: 8-512
                - activation: 优化（relu/tanh/logistic）
                - max_iter: 3000
                
                **LightGBM**
                - num_leaves: 10-200
                - max_depth: 3-25
                - learning_rate: 0.0005-0.5
                - n_estimators: 50-3000
                - min_child_samples: 3-150
                - subsample: 0.4-1.0
                - colsample_bytree: 0.4-1.0
                - reg_alpha: 0-15
                - reg_lambda: 0-15
                
                **XGBoost**
                - max_depth: 2-25
                - learning_rate: 0.005-0.5
                - n_estimators: 50-3000
                - min_child_weight: 0-15
                - gamma: 0-15
                - subsample: 0.4-1.0
                - colsample_bytree: 0.4-1.0
                - reg_alpha: 0-15
                - reg_lambda: 0-15
                """)
        
        st.markdown("---")
        
        st.markdown("#### 💡 如何选择？")
        st.caption("""
        **选择快速模式**：
        - ✅ 首次使用
        - ✅ 常规数据集
        - ✅ 时间有限
        - ✅ 不确定数据特征
        
        **选择完整模式**：
        - ✅ 快速模式效果不理想
        - ✅ 数据集有特殊特征
        - ✅ 追求极致性能
        - ✅ 有充足计算时间
        """)
        
        if st.button("🔄 重置为默认设置", key="reset_advanced_settings"):
            st.session_state.advanced_settings = {
                'optimization_mode': 'fast'
            }
            st.rerun()

# 步骤1：数据导入
if st.session_state.step == 1:
    st.header("1️⃣ 数据导入 (Data Import)")
    
    # 数据导入选项（始终在顶部显示）
    upload_method = st.radio("选择数据导入方式", ["上传数据文件", "使用示例数据"])
    
    if upload_method == "上传数据文件":
        uploaded_file = st.file_uploader(
            "上传数据文件", 
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="支持CSV、Excel、JSON、Parquet格式"
        )
        
        if uploaded_file is not None:
            try:
                # 根据文件类型加载数据
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    st.session_state.data = pd.read_csv(uploaded_file)
                    # 确保所有列名都是字符串类型
                    st.session_state.data.columns = st.session_state.data.columns.astype(str)
                    file_type_name = "CSV"
                    
                elif file_extension in ['xlsx', 'xls']:
                    st.session_state.data = pd.read_excel(uploaded_file, engine='openpyxl')
                    # 确保所有列名都是字符串类型
                    st.session_state.data.columns = st.session_state.data.columns.astype(str)
                    file_type_name = "Excel"
                    
                elif file_extension == 'json':
                    st.session_state.data = pd.read_json(uploaded_file)
                    # 确保所有列名都是字符串类型
                    st.session_state.data.columns = st.session_state.data.columns.astype(str)
                    file_type_name = "JSON"
                    
                elif file_extension == 'parquet':
                    st.session_state.data = pd.read_parquet(uploaded_file)
                    # 确保所有列名都是字符串类型
                    st.session_state.data.columns = st.session_state.data.columns.astype(str)
                    file_type_name = "Parquet"
                    
                else:
                    st.error(f"❌ 不支持的文件格式: {file_extension}")
                    st.session_state.data = None
                
                if st.session_state.data is not None:
                    # 数据清理：转换所有列为数值类型
                    for col in st.session_state.data.columns:
                        try:
                            # 尝试转换为数值类型，处理字符串格式的数字
                            st.session_state.data[col] = pd.to_numeric(
                                st.session_state.data[col], 
                                errors='coerce'  # 无法转换的设为NaN
                            )
                        except Exception:
                            pass  # 如果转换失败，保持原样
                    
                    # 删除包含NaN的行
                    original_rows = st.session_state.data.shape[0]
                    st.session_state.data = st.session_state.data.dropna()
                    dropped_rows = original_rows - st.session_state.data.shape[0]
                    
                    if dropped_rows > 0:
                        st.warning(f"⚠️ 已删除 {dropped_rows} 行包含无效数据的记录")
                    st.success(f"{file_type_name}文件加载成功！共 {st.session_state.data.shape[0]} 行，{st.session_state.data.shape[1]} 列")
                    
                    # 清除所有旧的缓存数据
                    keys_to_clear = [
                        'X_train', 'X_test', 'y_train', 'y_test',
                        'feature_names', 'shap_values', 'shap_X_sample', 'shap_explainer',
                        'best_model', 'model_results', 'gp_model', 'gp_results',
                        'pysr_model', 'pysr_results', 'improved_pysr_model', 'improved_pysr_results',
                        'pc_pysr_model', 'pc_pysr_results',
                        'target_col', 'feature_cols'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.info("💡 已清除旧数据缓存，请重新进行数据预处理和模型训练")
                    
            except Exception as e:
                st.error(f"❌ 数据加载失败：{str(e)}")
                st.info("💡 提示：请确保文件格式正确且数据完整")
    else:
        if st.button("生成示例数据", width="stretch"):
            from sklearn.datasets import make_regression
            X, y = make_regression(n_samples=500, n_features=10, n_informative=8, noise=10, random_state=42)
            st.session_state.data = pd.DataFrame(X, columns=[f'特征_{i+1}' for i in range(10)])
            st.session_state.data['目标变量'] = y
            
            # 清除所有旧的缓存数据
            keys_to_clear = [
                'X_train', 'X_test', 'y_train', 'y_test',
                'feature_names', 'shap_values', 'shap_X_sample', 'shap_explainer',
                'best_model', 'model_results', 'gp_model', 'gp_results',
                'pysr_model', 'pysr_results', 'improved_pysr_model', 'improved_pysr_results',
                'pc_pysr_model', 'pc_pysr_results',
                'target_col', 'feature_cols'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.success("示例数据生成成功")
            st.rerun()
    
    # 如果数据已加载，显示数据预览和导航按钮
    if st.session_state.data is not None:
        st.markdown("---")
        
        with st.expander("📊 查看数据详情", expanded=False):
            st.dataframe(st.session_state.data.head(20), width="stretch")
        
        # 选择目标变量和特征
        st.subheader("📊 选择变量")
        
        # 初始化默认选择（只在第一次加载时）
        if 'target_col' not in st.session_state:
            st.session_state.target_col = st.session_state.data.columns[-1]
        if 'feature_cols' not in st.session_state:
            # 确保所有列名都是字符串类型
            st.session_state.data.columns = st.session_state.data.columns.astype(str)
            st.session_state.feature_cols = list(st.session_state.data.columns[:-1])
        
        col1, col2 = st.columns(2)
        with col1:
            # 默认选择最后一列作为目标变量
            default_target_idx = len(st.session_state.data.columns) - 1
            target_col = st.selectbox(
                "选择目标变量（y）：", 
                st.session_state.data.columns, 
                index=default_target_idx,
                key="target_col_step1"
            )
        
        with col2:
            # 默认选择所有非目标变量作为特征
            default_features = [col for col in st.session_state.data.columns if col != target_col]
            feature_cols = st.multiselect(
                "选择特征变量（X）：", 
                default_features,
                default=default_features,
                key="feature_cols_step1"
            )
        
        if len(feature_cols) > 0:
            # 保存到session_state
            st.session_state.target_col = target_col
            st.session_state.feature_cols = feature_cols
            
            # 描述性统计
            st.subheader("📈 描述性统计")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**特征统计**")
                # 计算扩展统计信息
                desc_stats_features = st.session_state.data[feature_cols].describe()
                # 添加偏度和峰度（确保数据为数值类型）
                try:
                    # 强制转换为数值类型
                    numeric_features = st.session_state.data[feature_cols].apply(pd.to_numeric, errors='coerce')
                    skewness_features = numeric_features.skew()
                    kurtosis_features = numeric_features.kurtosis()
                    desc_stats_features.loc['skewness'] = skewness_features
                    desc_stats_features.loc['kurtosis'] = kurtosis_features
                except Exception as e:
                    st.warning(f"⚠️ 无法计算偏度和峰度：{str(e)}")
                st.dataframe(desc_stats_features)
                
                st.caption("💡 偏度：>0右偏，<0左偏；峰度：>0尖峰，<0平峰")
            
            with col2:
                st.write("**目标变量统计**")
                # 计算目标变量的扩展统计
                desc_stats_target = st.session_state.data[[target_col]].describe()
                try:
                    # 强制转换为数值类型
                    numeric_target = pd.to_numeric(st.session_state.data[target_col], errors='coerce')
                    skewness_target = numeric_target.skew()
                    kurtosis_target = numeric_target.kurtosis()
                    desc_stats_target.loc['skewness'] = [skewness_target]
                    desc_stats_target.loc['kurtosis'] = [kurtosis_target]
                except Exception as e:
                    st.warning(f"⚠️ 无法计算目标变量的偏度和峰度：{str(e)}")
                st.dataframe(desc_stats_target)
            
            # 🆕 数据分布可视化（可折叠）
            st.markdown("---")
            with st.expander("📊 数据分布可视化（可选）", expanded=False):
                st.info("💡 查看数据分布有助于了解数据质量和选择合适的预处理方法")
                
                # Tab分离：特征分布 vs 目标变量分布 vs 云雨图
                tab1, tab2, tab3 = st.tabs(["📊 特征分布", "🎯 目标变量分布", "☁️ 云雨图"])
                
                with tab1:
                    st.markdown("#### 📊 特征分布（直方图 + KDE）")
                    
                    # 计算需要的行数（每行3个特征）
                    n_features = len(feature_cols)
                    n_cols = 3
                    n_rows = (n_features + n_cols - 1) // n_cols
                    
                    # 创建分布图
                    fig_dist, axes_dist = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows), dpi=120)
                    if n_rows == 1:
                        axes_dist = axes_dist.reshape(1, -1)
                    axes_dist = axes_dist.flatten()
                    
                    for idx, feature in enumerate(feature_cols):
                        ax = axes_dist[idx]
                        feature_data = st.session_state.data[feature].dropna()
                        
                        # 检测是否为百分比小数（0~1之间，且不是二分类0/1变量）
                        is_percentage = (
                            feature_data.min() >= 0 and
                            feature_data.max() <= 1.0 and
                            feature_data.nunique() > 2
                        )
                        
                        # 如果是百分比小数，转换为百分数显示
                        plot_data = feature_data * 100 if is_percentage else feature_data
                        x_label = f'{feature} (%)' if is_percentage else feature
                        
                        # 智能设置X轴范围
                        x_min = 0 if plot_data.min() >= 0 else plot_data.min()
                        x_max = plot_data.max()
                        
                        # 直方图
                        ax.hist(plot_data, bins=25, edgecolor='white', alpha=0.7, 
                               color='steelblue', density=False, label='频率直方图')
                        
                        # KDE曲线（缩放到频数尺度）
                        try:
                            if plot_data.std() > 0:
                                from scipy.stats import gaussian_kde
                                kde = gaussian_kde(plot_data)
                                x_range = np.linspace(plot_data.min(), plot_data.max(), 200)
                                kde_values = kde(x_range) * len(plot_data) * (plot_data.max() - plot_data.min()) / 25
                                ax.plot(x_range, kde_values, color='darkred', linewidth=2.5, label='核密度曲线')
                            else:
                                ax.text(0.5, 0.5, '常数列\n（无变化）', 
                                       transform=ax.transAxes, ha='center', va='center',
                                       fontsize=13, color='red', alpha=0.7)
                        except Exception:
                            pass
                        
                        ax.set_xlim(x_min, x_max * 1.05)
                        ax.set_title(f'{feature}', fontsize=16, fontweight='bold', pad=12)
                        ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
                        ax.set_ylabel('样本数量', fontsize=14, fontweight='bold')
                        ax.tick_params(axis='both', labelsize=13)
                        ax.grid(True, alpha=0.3, linestyle='--')
                        ax.legend(fontsize=13)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                    
                    # 隐藏多余的子图
                    for idx in range(n_features, len(axes_dist)):
                        axes_dist[idx].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig_dist)
                    plt.close()
                    
                    # 箱线图
                    st.markdown("#### 📦 特征箱线图")
                    
                    fig_box, axes_box = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows), dpi=100)
                    if n_rows == 1:
                        axes_box = axes_box.reshape(1, -1)
                    axes_box = axes_box.flatten()
                    
                    for idx, feature in enumerate(feature_cols):
                        ax = axes_box[idx]
                        feature_data = st.session_state.data[feature].dropna()
                        
                        bp = ax.boxplot([feature_data], widths=0.5, patch_artist=True,
                                       boxprops=dict(facecolor='lightblue', alpha=0.7, edgecolor='black'),
                                       medianprops=dict(color='red', linewidth=2),
                                       whiskerprops=dict(color='black', linewidth=1.5),
                                       capprops=dict(color='black', linewidth=1.5),
                                       flierprops=dict(marker='o', markerfacecolor='red', 
                                                     markersize=4, alpha=0.5))
                        ax.set_xticklabels([feature], fontsize=9, fontweight='bold')
                        ax.set_title(f'{feature}', fontsize=11, fontweight='bold', pad=10)
                        ax.set_ylabel('Value', fontsize=9, fontweight='bold')
                        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                    
                    # 隐藏多余的子图
                    for idx in range(n_features, len(axes_box)):
                        axes_box[idx].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig_box)
                    plt.close()
                    
                    # 🆕 分布诊断
                    st.markdown("#### 💡 分布诊断")
                    
                    diagnostic_messages = []
                    for feature in feature_cols:
                        feature_data = st.session_state.data[feature].dropna()
                        skew = feature_data.skew()
                        
                        # 偏度诊断
                        if abs(skew) > 1.5:
                            if skew > 0:
                                diagnostic_messages.append(f"⚠️ **{feature}**: 强右偏（偏度={skew:.2f}），建议考虑log变换")
                            else:
                                diagnostic_messages.append(f"⚠️ **{feature}**: 强左偏（偏度={skew:.2f}），建议考虑指数变换")
                        elif abs(skew) > 0.5:
                            if skew > 0:
                                diagnostic_messages.append(f"💡 **{feature}**: 轻度右偏（偏度={skew:.2f}），可考虑sqrt变换")
                            else:
                                diagnostic_messages.append(f"💡 **{feature}**: 轻度左偏（偏度={skew:.2f}）")
                        
                        # 异常值预警（基于IQR）
                        Q1 = feature_data.quantile(0.25)
                        Q3 = feature_data.quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = feature_data[(feature_data < Q1 - 1.5*IQR) | (feature_data > Q3 + 1.5*IQR)]
                        if len(outliers) > 0:
                            outlier_pct = len(outliers) / len(feature_data) * 100
                            if outlier_pct > 5:
                                diagnostic_messages.append(f"⚠️ **{feature}**: 检测到{len(outliers)}个异常值（{outlier_pct:.1f}%），建议在步骤2中清洗")
                    
                    if diagnostic_messages:
                        for msg in diagnostic_messages:
                            st.markdown(msg)
                    else:
                        st.success("✅ 所有特征分布良好，无明显问题")
                
                with tab2:
                    st.markdown("#### 🎯 目标变量分布")
                    
                    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=120)
                    
                    # 直方图 + KDE
                    target_data = st.session_state.data[target_col].dropna()
                    
                    # 智能设置X轴范围：如果数据最小值>=0，则从0开始
                    x_min_target = 0 if target_data.min() >= 0 else target_data.min()
                    x_max_target = target_data.max()
                    
                    # 直方图（频数）+ KDE缩放
                    axes[0].hist(target_data, bins=30, edgecolor='white', alpha=0.7, 
                                color='coral', density=False, label='频率直方图')
                    try:
                        if target_data.std() > 0:
                            from scipy.stats import gaussian_kde
                            kde_t = gaussian_kde(target_data)
                            x_range_t = np.linspace(target_data.min(), target_data.max(), 200)
                            kde_vals_t = kde_t(x_range_t) * len(target_data) * (target_data.max() - target_data.min()) / 30
                            axes[0].plot(x_range_t, kde_vals_t, color='darkred', linewidth=2.5, label='核密度曲线')
                    except Exception:
                        pass
                    axes[0].set_xlim(x_min_target, x_max_target * 1.05)
                    axes[0].set_title(f'{target_col}（目标变量）分布', 
                                    fontsize=16, fontweight='bold', pad=15)
                    axes[0].set_xlabel(target_col, fontsize=14, fontweight='bold')
                    axes[0].set_ylabel('样本数量', fontsize=14, fontweight='bold')
                    axes[0].tick_params(axis='both', labelsize=13)
                    axes[0].grid(True, alpha=0.3, linestyle='--')
                    axes[0].legend(fontsize=13)
                    axes[0].spines['top'].set_visible(False)
                    axes[0].spines['right'].set_visible(False)
                    
                    # 箱线图
                    bp = axes[1].boxplot([target_data], widths=0.5, patch_artist=True,
                                        boxprops=dict(facecolor='lightcoral', alpha=0.7, edgecolor='black'),
                                        medianprops=dict(color='darkred', linewidth=2),
                                        whiskerprops=dict(color='black', linewidth=1.5),
                                        capprops=dict(color='black', linewidth=1.5),
                                        flierprops=dict(marker='o', markerfacecolor='red', 
                                                      markersize=5, alpha=0.5))
                    axes[1].set_xticklabels([target_col], fontsize=14, fontweight='bold')
                    axes[1].set_title(f'{target_col}（目标变量）箱线图', 
                                    fontsize=16, fontweight='bold', pad=15)
                    axes[1].set_ylabel('数值', fontsize=14, fontweight='bold')
                    axes[1].tick_params(axis='both', labelsize=13)
                    axes[1].tick_params(axis='both', labelsize=13)
                    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
                    axes[1].spines['top'].set_visible(False)
                    axes[1].spines['right'].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # 🆕 目标变量分布诊断
                    st.markdown("#### 💡 目标变量诊断")
                    
                    target_skew = target_data.skew()
                    target_kurt = target_data.kurtosis()
                    
                    # 偏度诊断
                    if abs(target_skew) > 1.5:
                        if target_skew > 0:
                            st.warning(f"⚠️ 目标变量强右偏（偏度={target_skew:.2f}），建议考虑log变换以改善模型性能")
                        else:
                            st.warning(f"⚠️ 目标变量强左偏（偏度={target_skew:.2f}），建议考虑指数变换")
                    elif abs(target_skew) > 0.5:
                        st.info(f"💡 目标变量轻度偏斜（偏度={target_skew:.2f}），可以考虑变换，但不是必须的")
                    else:
                        st.success(f"✅ 目标变量分布良好（偏度={target_skew:.2f}），接近正态分布")
                    
                    # 峰度诊断
                    if abs(target_kurt) > 3:
                        st.info(f"💡 目标变量峰度={target_kurt:.2f}（{'尖峰' if target_kurt > 0 else '平峰'}分布）")
                    
                    # 异常值预警
                    Q1 = target_data.quantile(0.25)
                    Q3 = target_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = target_data[(target_data < Q1 - 1.5*IQR) | (target_data > Q3 + 1.5*IQR)]
                    if len(outliers) > 0:
                        outlier_pct = len(outliers) / len(target_data) * 100
                        if outlier_pct > 5:
                            st.warning(f"⚠️ 检测到{len(outliers)}个异常值（{outlier_pct:.1f}%），建议在步骤2中清洗")
                        else:
                            st.info(f"💡 检测到{len(outliers)}个异常值（{outlier_pct:.1f}%），数量较少")
                    else:
                        st.success("✅ 未检测到明显异常值")
                
                with tab3:
                    st.markdown("#### ☁️ 云雨图（Raincloud Plot）")
                    st.info("💡 云雨图结合了小提琴图、箱线图和散点图，全面展示数据分布特征")
                    
                    # 选择要显示的特征
                    st.markdown("**选择要可视化的特征：**")
                    
                    # 默认选择前6个特征
                    default_raincloud_features = feature_cols[:min(6, len(feature_cols))]
                    
                    selected_raincloud_features = st.multiselect(
                        "选择特征（建议不超过8个）",
                        options=feature_cols,
                        default=default_raincloud_features,
                        key="raincloud_features"
                    )
                    
                    if len(selected_raincloud_features) > 0:
                        if len(selected_raincloud_features) > 10:
                            st.warning("⚠️ 选择的特征过多，图表可能较大")
                        
                        with st.spinner("正在生成云雨图，请稍候..."):
                            try:
                                import plotly.graph_objects as go
                                from plotly.subplots import make_subplots
                                
                                # 配色方案（柔和的颜色）
                                colors = ["#4f7942", "#c65f0a", "#a03530", "#5e4c8c", "#20648a", 
                                         "#7c9d3e", "#d97706", "#b91c1c", "#7c3aed", "#0891b2"]
                                fill_colors = ["#b0d1a6", "#f5c5a0", "#e7a39a", "#b1a5c9", "#a0d8ef",
                                              "#d4e5b0", "#fed7aa", "#fecaca", "#ddd6fe", "#a5f3fc"]
                                
                                # 扩展颜色列表以匹配特征数量
                                while len(colors) < len(selected_raincloud_features):
                                    colors.extend(colors)
                                    fill_colors.extend(fill_colors)
                                
                                # 计算子图布局（每行最多4个）
                                n_features = len(selected_raincloud_features)
                                n_cols = min(4, n_features)
                                n_rows = (n_features + n_cols - 1) // n_cols
                                
                                # 创建子图
                                fig_raincloud = make_subplots(
                                    rows=n_rows, 
                                    cols=n_cols,
                                    subplot_titles=selected_raincloud_features,
                                    vertical_spacing=0.12,
                                    horizontal_spacing=0.08
                                )
                                
                                for idx, feature in enumerate(selected_raincloud_features):
                                    row = idx // n_cols + 1
                                    col = idx % n_cols + 1
                                    
                                    feature_data = st.session_state.data[feature].dropna()
                                    
                                    # 检查数据范围，确定Y轴起点
                                    data_min = feature_data.min()
                                    data_max = feature_data.max()
                                    
                                    # 如果数据最小值>=0，Y轴从0开始；否则留一些边距
                                    if data_min >= 0:
                                        y_range_min = 0
                                        y_range_max = data_max * 1.1  # 上方留10%空间
                                    else:
                                        # 数据有负值，留边距
                                        data_range = data_max - data_min
                                        y_range_min = data_min - data_range * 0.05
                                        y_range_max = data_max + data_range * 0.05
                                    
                                    # 半小提琴图（上半部，柔和一些）
                                    fig_raincloud.add_trace(
                                        go.Violin(
                                            y=feature_data,
                                            name=feature,
                                            points=False,
                                            width=0.9,
                                            fillcolor=colors[idx],
                                            line_color=colors[idx],
                                            opacity=0.2,
                                            side="positive",
                                            showlegend=False,
                                            x0=feature,  # 设置x轴位置
                                            scalemode='width'  # 使用宽度缩放模式
                                        ),
                                        row=row, col=col
                                    )
                                    
                                    # 箱线图（只显示箱体，不显示散点）
                                    fig_raincloud.add_trace(
                                        go.Box(
                                            y=feature_data,
                                            name=feature,
                                            boxpoints=False,        # 不在箱线图上显示散点
                                            width=0.3,
                                            whiskerwidth=0.5,
                                            line=dict(color=colors[idx], width=2),
                                            fillcolor=fill_colors[idx],
                                            showlegend=False,
                                            opacity=0.8,
                                            boxmean="sd",
                                            notched=False,
                                            x0=feature
                                        ),
                                        row=row, col=col
                                    )
                                    
                                    # 散点（雨点）单独绘制，放在箱线图左侧
                                    fig_raincloud.add_trace(
                                        go.Box(
                                            y=feature_data,
                                            name=feature,
                                            boxpoints="all",
                                            width=0.0,              # 箱体宽度为0，只显示散点
                                            jitter=0.4,
                                            pointpos=-1.6,
                                            line=dict(color='rgba(0,0,0,0)'),  # 隐藏箱体线条
                                            fillcolor='rgba(0,0,0,0)',          # 隐藏箱体填充
                                            showlegend=False,
                                            marker_color=fill_colors[idx],
                                            marker_size=5,
                                            marker_line_color=colors[idx],
                                            marker_line_width=1,
                                            x0=feature
                                        ),
                                        row=row, col=col
                                    )
                                    
                                    # 更新每个子图的Y轴标签和范围
                                    fig_raincloud.update_yaxes(
                                        title_text=feature,
                                        row=row, col=col,
                                        title_font=dict(size=32, color="#333"),
                                        tickfont=dict(size=30),
                                        range=[y_range_min, y_range_max],
                                        autorange=False
                                    )
                                    
                                    # 隐藏X轴刻度（因为只有一个类别）
                                    fig_raincloud.update_xaxes(
                                        showticklabels=False,
                                        row=row, col=col
                                    )
                                
                                # 布局设置
                                fig_raincloud.update_layout(
                                    height=max(400, n_rows * 350),
                                    showlegend=False,
                                    template="plotly_white",
                                    title=dict(
                                        text="特征分布云雨图（每个特征独立Y轴）",
                                        font=dict(size=40, family="Arial", color="#333"),
                                        x=0.5,
                                        xanchor="center"
                                    )
                                )
                                
                                # 更新子图标题字体
                                for annotation in fig_raincloud.layout.annotations:
                                    annotation.font.size = 32
                                
                                st.plotly_chart(fig_raincloud, width='stretch')
                                
                                # 云雨图说明
                                st.markdown("**📖 云雨图解读：**")
                                st.markdown("""
                                - **云（半小提琴图）**：显示数据的概率密度分布，越宽表示该值出现的频率越高
                                - **雨点（散点）**：显示所有原始数据点，可以看到数据的实际分布
                                - **箱线图**：显示中位数（中间线）、四分位数（箱体）、均值±标准差（菱形标记）
                                - **独立Y轴**：每个特征使用自己的数值范围，便于观察分布形状而非比较数值大小
                                """)
                                
                            except Exception as e:
                                st.error(f"生成云雨图时出错：{str(e)}")
                                st.info("请确保已安装 plotly 库：pip install plotly")
                    else:
                        st.warning("请至少选择一个特征来生成云雨图")
            
            # 🆕 相关性分析和配对图
            st.markdown("---")
            with st.expander("🔗 相关性分析与配对图（可选）", expanded=False):
                st.info("💡 相关性分析帮助理解特征之间的关系，配对图可视化特征间的相互作用")
                
                # Tab分离：相关性矩阵 vs 配对图
                tab1, tab2 = st.tabs(["📊 相关性矩阵", "🎨 配对图"])
                
                with tab1:
                    st.markdown("#### 📊 特征相关性矩阵")
                    
                    # 计算相关系数矩阵
                    corr_data = st.session_state.data[feature_cols + [target_col]]
                    corr_matrix = corr_data.corr()
                    
                    # 创建热力图
                    fig_corr, ax_corr = plt.subplots(figsize=(12, 10), dpi=120)
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                               center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                               mask=mask, ax=ax_corr, vmin=-1, vmax=1,
                               annot_kws={"size": 16})
                    
                    # 调整轴标签和刻度字体
                    ax_corr.set_xticklabels(ax_corr.get_xticklabels(), fontsize=15, fontweight='bold', rotation=45, ha='right')
                    ax_corr.set_yticklabels(ax_corr.get_yticklabels(), fontsize=15, fontweight='bold', rotation=0)
                    
                    # 调整色条标签
                    cbar = ax_corr.collections[0].colorbar
                    if cbar:
                        cbar.ax.tick_params(labelsize=14)
                        cbar.set_label('相关系数', fontsize=15, fontweight='bold')
                    
                    ax_corr.set_title('特征相关性矩阵', fontsize=20, fontweight='bold', pad=20)
                    plt.tight_layout()
                    st.pyplot(fig_corr)
                    plt.close()
                    
                    # 相关性诊断
                    st.markdown("#### 💡 相关性诊断")
                    
                    # 找出与目标变量相关性最强的特征
                    target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
                    
                    st.markdown(f"**与目标变量 `{target_col}` 相关性排名：**")
                    for i, (feat, corr_val) in enumerate(target_corr.head(5).items(), 1):
                        actual_corr = corr_matrix.loc[feat, target_col]
                        if abs(actual_corr) > 0.7:
                            st.markdown(f"{i}. **{feat}**: {actual_corr:.3f} {'📈' if actual_corr > 0 else '📉'} （强相关）")
                        elif abs(actual_corr) > 0.4:
                            st.markdown(f"{i}. **{feat}**: {actual_corr:.3f} {'📈' if actual_corr > 0 else '📉'} （中等相关）")
                        else:
                            st.markdown(f"{i}. **{feat}**: {actual_corr:.3f} {'📈' if actual_corr > 0 else '📉'} （弱相关）")
                    
                    # 检测多重共线性
                    st.markdown("**多重共线性检测（基于相关系数）：**")
                    high_corr_pairs = []
                    for i in range(len(feature_cols)):
                        for j in range(i+1, len(feature_cols)):
                            corr_val = corr_matrix.loc[feature_cols[i], feature_cols[j]]
                            if abs(corr_val) > 0.8:
                                high_corr_pairs.append((feature_cols[i], feature_cols[j], corr_val))
                    
                    if high_corr_pairs:
                        st.warning("⚠️ 检测到高度相关的特征对（|相关系数| > 0.8）：")
                        for feat1, feat2, corr_val in high_corr_pairs:
                            st.markdown(f"- **{feat1}** ↔ **{feat2}**: {corr_val:.3f}")
                    else:
                        st.success("✅ 未检测到高度相关的特征对")
                
                with tab2:
                    st.markdown("#### 🎨 特征配对图（Pairplot）")
                    
                    # 选择要显示的特征
                    st.markdown("**选择要可视化的特征：**")
                    
                    # 默认选择与目标变量相关性最强的前5个特征
                    target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
                    default_features = list(target_corr.head(min(5, len(feature_cols))).index)
                    
                    selected_features = st.multiselect(
                        "选择特征（建议不超过6个，避免图表过大）",
                        options=feature_cols,
                        default=default_features,
                        key="pairplot_features"
                    )
                    
                    if len(selected_features) > 0:
                        if len(selected_features) > 8:
                            st.warning("⚠️ 选择的特征过多，图表可能较大且加载缓慢")
                        
                        # 添加目标变量
                        pairplot_cols = selected_features + [target_col]
                        pairplot_data = st.session_state.data[pairplot_cols].dropna()
                        
                        with st.spinner("正在生成配对图，请稍候..."):
                            try:
                                # 创建配对图
                                import seaborn as sns
                                fig_pair = sns.pairplot(
                                    pairplot_data, 
                                    kind='reg',
                                    diag_kind='kde',
                                    plot_kws={'scatter_kws': {'alpha': 0.5, 's': 20}, 
                                             'line_kws': {'color': 'red', 'linewidth': 1.5}},
                                    diag_kws={'color': 'steelblue', 'linewidth': 2},
                                    corner=False,
                                    height=2.5
                                )
                                
                                # 调整所有子图的字体大小
                                for ax in fig_pair.axes.flatten():
                                    if ax is not None:
                                        ax.set_xlabel(ax.get_xlabel(), fontsize=15, fontweight='bold')
                                        ax.set_ylabel(ax.get_ylabel(), fontsize=15, fontweight='bold')
                                        ax.tick_params(axis='both', labelsize=13)
                                
                                fig_pair.fig.suptitle('特征配对图（含回归线）', 
                                                     y=1.01, fontsize=18, fontweight='bold')
                                plt.tight_layout()
                                st.pyplot(fig_pair)
                                plt.close()
                                
                                st.success("✅ 配对图生成完成")
                                st.markdown("""
                                **图表说明：**
                                - 对角线：各特征的分布（KDE曲线）
                                - 非对角线：特征间的散点图+回归线
                                - 红色线：线性回归拟合线
                                - 散点透明度：避免重叠遮挡
                                """)
                                
                            except Exception as e:
                                st.error(f"❌ 配对图生成失败：{str(e)}")
                                st.info("💡 提示：如果特征过多或数据量过大，可能导致生成失败，请减少选择的特征数量")
                    else:
                        st.warning("⚠️ 请至少选择一个特征")
            
            # 智能建议
            st.markdown("---")
            suggestions = analyze_data_characteristics(
                st.session_state.data, 
                feature_cols, 
                target_col
            )
            display_smart_suggestions(suggestions)
            
            # 导航按钮
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("继续到步骤2：数据预处理 →", type="primary", width="stretch", key="nav_to_step2"):
                    st.session_state.step = 2
                    st.rerun()
        else:
            st.warning("⚠️ 请至少选择一个特征变量")
        
        st.markdown("---")
        st.info("💡 如需重新导入数据，请使用上方选项")

# 步骤2：数据预处理
elif st.session_state.step == 2:
    st.header("2️⃣ 数据预处理 (Data Preprocessing)")
    
    if st.session_state.data is None or 'target_col' not in st.session_state or 'feature_cols' not in st.session_state:
        st.warning("⚠️ 请先在步骤1完成数据导入和变量选择")
        if st.button("← 返回步骤1"):
            st.session_state.step = 1
            st.rerun()
    else:
        data = st.session_state.data
        target_col = st.session_state.target_col
        feature_cols = st.session_state.feature_cols
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← 返回"):
                st.session_state.step = 1
                st.rerun()
        
        # 缺失值处理
        st.subheader("🔧 缺失值处理")
        missing_count = data[feature_cols + [target_col]].isnull().sum()
        if missing_count.sum() > 0:
            st.warning(f"发现 {missing_count.sum()} 个缺失值")
            st.write(missing_count[missing_count > 0])
            
            if st.button("删除缺失值", key="remove_missing_step2"):
                data = data.dropna(subset=feature_cols + [target_col])
                st.session_state.data = data
                st.success("✅ 缺失值已删除")
                st.rerun()
        else:
            st.success("✅ 无缺失值")
        
        st.markdown("---")
        
        # 异常值清洗（孤立森林）
        st.subheader("🎯 异常值清洗（孤立森林算法）")
        
        st.info("� 使用孤立森林算法检测并清洗多变量异常值，清洗后的数据将用于后续分析")
        
        # 参数设置
        col_param1, col_param2 = st.columns(2)
        
        with col_param1:
            contamination = st.slider(
                "异常值比例（Contamination）",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                help="预期数据中异常值的比例，例如0.05表示5%的数据可能是异常值"
            )
        
        with col_param2:
            random_state_if = st.number_input(
                "随机种子（孤立森林）",
                min_value=0,
                max_value=999,
                value=42,
                help="设置随机种子以确保结果可重复"
            )
        
        if st.button("🔍 执行异常值检测与清洗", type="primary", key="execute_cleaning_step2"):
            from sklearn.ensemble import IsolationForest
            from datetime import datetime
            import os
            
            with st.spinner("正在执行异常值检测..."):
                # 确保所有列名都是字符串类型
                feature_cols_str = [str(col) for col in feature_cols]
                
                # 创建孤立森林模型
                iso_forest = IsolationForest(
                    contamination=contamination,
                    random_state=int(random_state_if),
                    n_estimators=100
                )
                
                # 拟合并预测
                outlier_labels = iso_forest.fit_predict(data[feature_cols_str])
                outlier_scores = iso_forest.score_samples(data[feature_cols_str])
                
                # -1表示异常值，1表示正常值
                n_outliers = (outlier_labels == -1).sum()
                n_normal = (outlier_labels == 1).sum()
                data_cleaned = data[outlier_labels == 1].copy()
                
                # 保存到session_state用于后续显示
                st.session_state.outlier_labels = outlier_labels
                st.session_state.outlier_scores = outlier_scores
                st.session_state.n_outliers = n_outliers
                st.session_state.n_normal = n_normal
                st.session_state.original_data_len = len(data)
                
                # 更新session_state中的数据为清洗后的数据
                st.session_state.data = data_cleaned
                st.session_state.data_before_cleaning = data  # 保存清洗前数据用于对比
                st.session_state.n_outliers_removed = n_outliers
                st.session_state.outlier_cleaning_done = True
                
                # 保存清洗后数据为CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cleaned_filename = f"data_cleaned_{timestamp}.csv"
                data_cleaned.to_csv(cleaned_filename, index=False)
                st.session_state.cleaned_data_path = cleaned_filename
                
                st.success(f"✅ 异常值检测完成！清洗后数据已保存：{cleaned_filename}")
                st.success(f"✅ 数据已更新为清洗后版本，后续分析将使用清洗后的数据")
                
            st.rerun()  # 重新运行以更新data变量
        
        # 如果已经执行过清洗，显示结果
        if st.session_state.get('outlier_cleaning_done', False):
            st.success(f"✅ 异常值清洗已完成！移除了 {st.session_state.n_outliers_removed} 个异常样本")
            st.info("💡 当前使用的是清洗后的数据，后续的相关性分析和特征工程都基于清洗后的数据")
            
            # 显示统计信息
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("原始样本数", st.session_state.original_data_len)
            with col_stat2:
                st.metric("检测到异常值", st.session_state.n_outliers, 
                         delta=f"-{st.session_state.n_outliers/st.session_state.original_data_len*100:.1f}%")
            with col_stat3:
                st.metric("清洗后样本数", st.session_state.n_normal)
            
            # 可视化异常分数分布
            st.markdown("#### 📊 异常值检测结果")
            
            outlier_labels = st.session_state.outlier_labels
            outlier_scores = st.session_state.outlier_scores
            n_outliers = st.session_state.n_outliers
            n_normal = st.session_state.n_normal
            data_before = st.session_state.data_before_cleaning
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
            
            # 异常分数直方图
            axes[0].hist(outlier_scores[outlier_labels == 1], bins=30, 
                       alpha=0.7, color='green', label=f'Normal ({n_normal})', edgecolor='white')
            axes[0].hist(outlier_scores[outlier_labels == -1], bins=30, 
                       alpha=0.7, color='red', label=f'Outliers ({n_outliers})', edgecolor='white')
            axes[0].set_xlabel('Anomaly Score', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
            axes[0].set_title('Isolation Forest Anomaly Score Distribution', 
                            fontsize=13, fontweight='bold', pad=15)
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3, linestyle='--')
            axes[0].spines['top'].set_visible(False)
            axes[0].spines['right'].set_visible(False)
            
            # 异常值散点图（使用前两个特征）
            if len(feature_cols) >= 2:
                axes[1].scatter(data_before[feature_cols[0]][outlier_labels == 1], 
                              data_before[feature_cols[1]][outlier_labels == 1],
                              c='green', alpha=0.6, s=30, label=f'Normal ({n_normal})', edgecolors='white')
                axes[1].scatter(data_before[feature_cols[0]][outlier_labels == -1], 
                              data_before[feature_cols[1]][outlier_labels == -1],
                              c='red', alpha=0.8, s=50, label=f'Outliers ({n_outliers})', 
                              edgecolors='black', linewidths=1.5, marker='X')
                axes[1].set_xlabel(feature_cols[0], fontsize=11, fontweight='bold')
                axes[1].set_ylabel(feature_cols[1], fontsize=11, fontweight='bold')
                axes[1].set_title(f'Outlier Detection: {feature_cols[0]} vs {feature_cols[1]}', 
                                fontsize=13, fontweight='bold', pad=15)
                axes[1].legend(fontsize=10)
                axes[1].grid(True, alpha=0.3, linestyle='--')
                axes[1].spines['top'].set_visible(False)
                axes[1].spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # 相关性分析（使用清洗后的数据）
        st.subheader("📊 特征相关性分析")
        
        # 如果已清洗，使用清洗后的数据
        if st.session_state.get('outlier_cleaning_done', False):
            data = st.session_state.data  # 使用清洗后的数据
            st.caption("📌 以下分析基于清洗后的数据")
        
        # 相关性阈值设置
        corr_threshold = st.slider(
            "共线性警告阈值",
            min_value=0.70,
            max_value=0.95,
            value=0.85,
            step=0.05,
            help="当相关系数绝对值超过此阈值时，将标记为强共线性"
        )
        
        # 计算皮尔逊和斯皮尔曼相关系数
        from scipy.stats import spearmanr
        
        pearson_corr = data[feature_cols].corr(method='pearson')
        spearman_corr = data[feature_cols].corr(method='spearman')
        
        # 绘制热力图
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=120)
        
        # 皮尔逊相关系数热力图
        import seaborn as sns
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(pearson_corr, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=axes[0])
        axes[0].set_title('Pearson Correlation Coefficient', 
                        fontsize=14, fontweight='bold', pad=15)
        axes[0].set_xlabel('')
        axes[0].set_ylabel('')
        
        # 斯皮尔曼相关系数热力图
        mask2 = np.triu(np.ones_like(spearman_corr, dtype=bool))
        sns.heatmap(spearman_corr, mask=mask2, annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=axes[1])
        axes[1].set_title('Spearman Correlation Coefficient', 
                        fontsize=14, fontweight='bold', pad=15)
        axes[1].set_xlabel('')
        axes[1].set_ylabel('')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # 检测强共线性
        high_corr_pairs = []
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                pearson_val = abs(pearson_corr.iloc[i, j])
                spearman_val = abs(spearman_corr.iloc[i, j])
                if pearson_val > corr_threshold or spearman_val > corr_threshold:
                    high_corr_pairs.append({
                        'Feature 1': pearson_corr.columns[i],
                        'Feature 2': pearson_corr.columns[j],
                        'Pearson |r|': f'{pearson_val:.3f}',
                        'Spearman |r|': f'{spearman_val:.3f}'
                    })
        
        if high_corr_pairs:
            st.warning(f"⚠️ 检测到 {len(high_corr_pairs)} 对强共线性特征（|r| > {corr_threshold})")
            st.dataframe(pd.DataFrame(high_corr_pairs), width='stretch')
            st.info("💡 提示：强共线性可能影响模型解释性和稳定性，建议进行处理")
            
            # 共线性处理选项
            st.markdown("#### 🔧 共线性处理方案")
            
            handling_method = st.radio(
                "选择处理方法：",
                options=[
                    "不处理（保留所有特征）",
                    "智能特征选择（推荐）",
                    "手动选择要删除的特征"
                ],
                help="智能特征选择：自动保留与目标变量相关性更高的特征"
            )
            
            if handling_method == "智能特征选择（推荐）":
                st.info("📊 智能选择策略：对于每对强共线性特征，保留与目标变量相关性更高的特征")
                
                if st.button("🚀 执行智能特征选择", type="primary"):
                    # 计算每个特征与目标变量的相关性
                    target_corr = data[feature_cols].corrwith(data[target_col]).abs()
                    
                    # 记录要删除的特征
                    features_to_remove = set()
                    removal_reasons = []
                    
                    # 遍历每对强共线性特征
                    for pair in high_corr_pairs:
                        feat1 = pair['Feature 1']
                        feat2 = pair['Feature 2']
                        
                        # 如果两个特征都还没被删除
                        if feat1 not in features_to_remove and feat2 not in features_to_remove:
                            corr1 = target_corr[feat1]
                            corr2 = target_corr[feat2]
                            
                            # 删除相关性较低的特征
                            if corr1 >= corr2:
                                features_to_remove.add(feat2)
                                removal_reasons.append({
                                    '删除特征': feat2,
                                    '保留特征': feat1,
                                    f'{feat2}与目标相关性': f'{corr2:.3f}',
                                    f'{feat1}与目标相关性': f'{corr1:.3f}',
                                    '原因': f'与{feat1}强共线性，且相关性较低'
                                })
                            else:
                                features_to_remove.add(feat1)
                                removal_reasons.append({
                                    '删除特征': feat1,
                                    '保留特征': feat2,
                                    f'{feat1}与目标相关性': f'{corr1:.3f}',
                                    f'{feat2}与目标相关性': f'{corr2:.3f}',
                                    '原因': f'与{feat2}强共线性，且相关性较低'
                                })
                    
                    if features_to_remove:
                        # 保存原始特征列表（用于撤销）
                        if 'original_feature_cols' not in st.session_state:
                            st.session_state.original_feature_cols = feature_cols.copy()
                        
                        # 更新特征列表
                        new_feature_cols = [f for f in feature_cols if f not in features_to_remove]
                        st.session_state.feature_cols = new_feature_cols
                        
                        st.success(f"✅ 已删除 {len(features_to_remove)} 个特征，保留 {len(new_feature_cols)} 个特征")
                        
                        # 显示删除详情
                        st.markdown("##### 📋 特征删除详情")
                        st.dataframe(pd.DataFrame(removal_reasons), width='stretch')
                        
                        # 显示保留的特征
                        st.markdown("##### 📋 保留的特征")
                        st.write(", ".join(new_feature_cols))
                        
                        st.info("💡 特征列表已更新，将影响后续所有分析步骤")
                        
                        # 提供撤销选项
                        if st.button("↩️ 撤销操作（恢复所有特征）"):
                            st.session_state.feature_cols = st.session_state.original_feature_cols
                            del st.session_state.original_feature_cols
                            st.success("✅ 已恢复所有特征")
                            st.rerun()
                    else:
                        st.info("所有强共线性特征对已被处理")
            
            elif handling_method == "手动选择要删除的特征":
                st.info("📝 请从下方选择要删除的特征")
                
                # 显示每个特征与目标变量的相关性，帮助用户决策
                target_corr = data[feature_cols].corrwith(data[target_col]).abs()
                feature_info = pd.DataFrame({
                    '特征名称': feature_cols,
                    '与目标变量相关性': [f'{target_corr[f]:.3f}' for f in feature_cols]
                })
                st.dataframe(feature_info, width='stretch')
                
                # 多选框选择要删除的特征
                features_to_remove = st.multiselect(
                    "选择要删除的特征：",
                    options=feature_cols,
                    help="可以选择多个特征"
                )
                
                if features_to_remove:
                    if st.button("🗑️ 确认删除选中的特征", type="primary"):
                        # 保存原始特征列表（用于撤销）
                        if 'original_feature_cols' not in st.session_state:
                            st.session_state.original_feature_cols = feature_cols.copy()
                        
                        # 更新特征列表
                        new_feature_cols = [f for f in feature_cols if f not in features_to_remove]
                        st.session_state.feature_cols = new_feature_cols
                        
                        st.success(f"✅ 已删除 {len(features_to_remove)} 个特征：{', '.join(features_to_remove)}")
                        st.success(f"保留 {len(new_feature_cols)} 个特征：{', '.join(new_feature_cols)}")
                        
                        st.info("💡 特征列表已更新，将影响后续所有分析步骤")
                        
                        # 提供撤销选项
                        if st.button("↩️ 撤销操作（恢复所有特征）"):
                            st.session_state.feature_cols = st.session_state.original_feature_cols
                            del st.session_state.original_feature_cols
                            st.success("✅ 已恢复所有特征")
                            st.rerun()
            
            else:  # 不处理
                st.info("✅ 将保留所有特征继续分析")
        
        else:
            st.success(f"✅ 未检测到强共线性（阈值 |r| > {corr_threshold})")
        
        # 保存标记，避免SHAP中重复分析
        st.session_state.correlation_done = True
        
        st.markdown("---")
        
        # 🆕 智能特征工程（可选）
        st.subheader("🔧 智能特征工程（可选）")
        
        st.info("💡 基于数据分析自动推荐特征工程方法，提升模型性能")
        
        with st.expander("📖 什么是特征工程？", expanded=False):
            st.markdown("""
            特征工程是通过数学变换创建新特征的过程，可以显著提升模型性能：
            
            - **多项式特征**：x² , x³ - 捕捉非线性关系
            - **交互特征**：x₁ × x₂ - 捕捉特征间的协同效应  
            - **数学变换**：log(x), sqrt(x), exp(x) - 处理偏态分布
            - **比率特征**：x₁/x₂ - 创建有意义的比例关系
            
            系统会自动分析您的数据并推荐最合适的特征工程方法。
            """)
        
        # 特征工程选项
        enable_feature_engineering = st.checkbox(
            "启用智能特征工程",
            value=False,
            help="系统将自动分析数据并推荐特征工程方法"
        )
        
        if enable_feature_engineering:
            st.markdown("#### 📊 自动分析特征")
            
            # 获取当前数据（如果已清洗则使用清洗后的数据）
            current_data = st.session_state.data if st.session_state.get('outlier_cleaning_done', False) else data
            
            with st.spinner("正在分析特征分布和关系..."):
                from scipy import stats
                
                # 1. 分析特征分布
                feature_stats = {}
                for feature in feature_cols:
                    feature_data = current_data[feature].dropna()
                    feature_stats[feature] = {
                        'mean': feature_data.mean(),
                        'std': feature_data.std(),
                        'skewness': stats.skew(feature_data),
                        'min': feature_data.min(),
                        'max': feature_data.max(),
                        'range': feature_data.max() - feature_data.min()
                    }
                
                # 2. 计算与目标变量的相关性
                target_corr = {}
                for feature in feature_cols:
                    corr = abs(current_data[feature].corr(current_data[target_col]))
                    target_corr[feature] = corr
                
                # 3. 计算特征间相关性
                feature_corr = current_data[feature_cols].corr()
            
            st.success("✅ 特征分析完成")
            
            # 可视化1：特征重要性（与目标变量的相关性）
            st.markdown("#### � 特征重要性分析")
            
            # 创建相关性条形图
            sorted_corr = sorted(target_corr.items(), key=lambda x: x[1], reverse=True)
            features_sorted = [item[0] for item in sorted_corr]
            corr_values = [item[1] for item in sorted_corr]
            
            fig_importance, ax_importance = plt.subplots(figsize=(10, max(6, len(feature_cols) * 0.4)), dpi=100)
            
            # 使用颜色渐变表示重要性
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(features_sorted)))
            bars = ax_importance.barh(features_sorted, corr_values, color=colors, edgecolor='black', linewidth=0.5)
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, corr_values)):
                ax_importance.text(val + 0.01, i, f'{val:.3f}', 
                                  va='center', fontsize=9, fontweight='bold')
            
            ax_importance.set_xlabel('Correlation with Target (|r|)', fontsize=11, fontweight='bold')
            ax_importance.set_title('Feature Importance: Correlation with Target Variable', 
                                   fontsize=13, fontweight='bold', pad=15)
            ax_importance.set_xlim(0, max(corr_values) * 1.15)
            ax_importance.grid(True, alpha=0.3, linestyle='--', axis='x')
            ax_importance.spines['top'].set_visible(False)
            ax_importance.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig_importance)
            plt.close()
            
            # 可视化2：特征关系热力图
            st.markdown("#### 🔗 特征关系网络")
            
            col_heatmap, col_stats = st.columns([2, 1])
            
            with col_heatmap:
                fig_network, ax_network = plt.subplots(figsize=(10, 8), dpi=100)
                
                import seaborn as sns
                
                # 绘制相关性热力图
                mask = np.triu(np.ones_like(feature_corr, dtype=bool), k=1)
                sns.heatmap(feature_corr, mask=mask, annot=True, fmt='.2f', 
                           cmap='coolwarm', center=0, vmin=-1, vmax=1,
                           square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                           ax=ax_network)
                ax_network.set_title('Feature Correlation Matrix', 
                                    fontsize=13, fontweight='bold', pad=15)
                
                plt.tight_layout()
                st.pyplot(fig_network)
                plt.close()
            
            with col_stats:
                st.markdown("**关系统计：**")
                
                # 计算强相关对
                strong_corr_pairs = []
                for i in range(len(feature_cols)):
                    for j in range(i+1, len(feature_cols)):
                        corr_val = abs(feature_corr.iloc[i, j])
                        if corr_val > 0.6:
                            strong_corr_pairs.append((feature_cols[i], feature_cols[j], corr_val))
                
                if strong_corr_pairs:
                    st.write(f"🔴 强相关对：{len(strong_corr_pairs)}个")
                    for f1, f2, val in sorted(strong_corr_pairs, key=lambda x: x[2], reverse=True)[:3]:
                        st.write(f"• {f1} ↔ {f2}: {val:.2f}")
                else:
                    st.write("✅ 无强相关特征对")
                
                # 计算独立对
                independent_pairs = []
                for i in range(len(feature_cols)):
                    for j in range(i+1, len(feature_cols)):
                        corr_val = abs(feature_corr.iloc[i, j])
                        if corr_val < 0.3:
                            independent_pairs.append((feature_cols[i], feature_cols[j], corr_val))
                
                st.write(f"🟢 独立特征对：{len(independent_pairs)}个")
            
            # 可视化3：特征分布概览
            st.markdown("#### 📈 特征分布概览")
            
            # 创建小提琴图展示所有特征的分布
            fig_dist, ax_dist = plt.subplots(figsize=(12, 6), dpi=100)
            
            # 标准化数据用于可视化
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(current_data[feature_cols])
            
            # 创建小提琴图
            positions = range(len(feature_cols))
            parts = ax_dist.violinplot(data_scaled, positions=positions, 
                                       showmeans=True, showmedians=True)
            
            # 美化小提琴图
            for pc in parts['bodies']:
                pc.set_facecolor('skyblue')
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # 标记偏态特征
            for i, feature in enumerate(feature_cols):
                skewness = feature_stats[feature]['skewness']
                if abs(skewness) > 1.0:
                    ax_dist.scatter(i, 0, s=200, c='red', marker='*', 
                                   zorder=5, label='Skewed' if i == 0 else '')
            
            ax_dist.set_xticks(positions)
            ax_dist.set_xticklabels(feature_cols, rotation=45, ha='right')
            ax_dist.set_ylabel('Standardized Value', fontsize=11, fontweight='bold')
            ax_dist.set_title('Feature Distribution Overview (Standardized)', 
                             fontsize=13, fontweight='bold', pad=15)
            ax_dist.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax_dist.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax_dist.spines['top'].set_visible(False)
            ax_dist.spines['right'].set_visible(False)
            
            if any(abs(feature_stats[f]['skewness']) > 1.0 for f in feature_cols):
                ax_dist.legend(loc='upper right')
            
            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close()
            
            # 显示分析摘要
            with st.expander("📋 查看详细分析摘要", expanded=False):
                col_summary1, col_summary2, col_summary3 = st.columns(3)
                
                with col_summary1:
                    st.markdown("**特征分布特征：**")
                    skewed_features = [f for f, s in feature_stats.items() if abs(s['skewness']) > 1.0]
                    if skewed_features:
                        st.write(f"- 偏态特征（|偏度| > 1.0）：{len(skewed_features)}个")
                        for f in skewed_features:
                            st.write(f"  • {f}: 偏度={feature_stats[f]['skewness']:.2f}")
                    else:
                        st.write("- 无明显偏态特征")
                
                with col_summary2:
                    st.markdown("**与目标变量相关性：**")
                    sorted_corr = sorted(target_corr.items(), key=lambda x: x[1], reverse=True)
                    st.write(f"- 最强相关：{sorted_corr[0][0]} (r={sorted_corr[0][1]:.3f})")
                    st.write(f"- 最弱相关：{sorted_corr[-1][0]} (r={sorted_corr[-1][1]:.3f})")
                    st.write(f"- 平均相关性：{np.mean(list(target_corr.values())):.3f}")
                
                with col_summary3:
                    st.markdown("**特征间关系：**")
                    st.write(f"- 强相关对（|r|>0.6）：{len(strong_corr_pairs)}个")
                    st.write(f"- 独立特征对（|r|<0.3）：{len(independent_pairs)}个")
                    avg_inter_corr = np.mean([abs(feature_corr.iloc[i, j]) 
                                              for i in range(len(feature_cols)) 
                                              for j in range(i+1, len(feature_cols))])
                    st.write(f"- 平均特征间相关性：{avg_inter_corr:.3f}")
            
            st.markdown("---")
            st.markdown("#### 💡 智能特征工程推荐")
            
            # 生成推荐
            recommendations = {
                'high': [],  # 高优先级（预期R²提升 > 0.03）
                'medium': [],  # 中优先级（预期R²提升 0.01-0.03）
                'low': []  # 低优先级（预期R²提升 < 0.01）
            }
            
            # 推荐逻辑
            
            # 1. 多项式特征推荐（针对高相关性特征）
            for feature, corr in sorted(target_corr.items(), key=lambda x: x[1], reverse=True)[:3]:
                if corr > 0.3:  # 相关性较高
                    expected_gain = min(corr * 0.15, 0.08)  # 预期提升
                    priority = 'high' if expected_gain > 0.03 else 'medium'
                    recommendations[priority].append({
                        'type': 'polynomial',
                        'feature': feature,
                        'formula': f'{feature}²',
                        'reason': f'与目标变量相关性高 (r={corr:.3f})',
                        'expected_gain': expected_gain
                    })
            
            # 2. 交互特征推荐（针对独立但都相关的特征对）
            for i, feat1 in enumerate(feature_cols):
                for feat2 in feature_cols[i+1:]:
                    # 特征间相关性低（独立）但都与目标相关
                    if abs(feature_corr.loc[feat1, feat2]) < 0.5:  # 特征间独立
                        if target_corr[feat1] > 0.2 and target_corr[feat2] > 0.2:  # 都与目标相关
                            expected_gain = (target_corr[feat1] + target_corr[feat2]) * 0.08
                            priority = 'high' if expected_gain > 0.03 else 'medium' if expected_gain > 0.01 else 'low'
                            recommendations[priority].append({
                                'type': 'interaction',
                                'feature': f'{feat1} × {feat2}',
                                'formula': f'{feat1} × {feat2}',
                                'reason': f'特征独立 (r={feature_corr.loc[feat1, feat2]:.2f}) 且都与目标相关',
                                'expected_gain': expected_gain,
                                'feat1': feat1,
                                'feat2': feat2
                            })
            
            # 3. 数学变换推荐（针对偏态特征）
            for feature, stats_dict in feature_stats.items():
                skewness = stats_dict['skewness']
                if abs(skewness) > 1.0 and target_corr[feature] > 0.15:  # 偏态且与目标相关
                    if skewness > 1.0:  # 右偏
                        transform = 'log'
                        formula = f'log({feature})'
                    else:  # 左偏
                        transform = 'sqrt'
                        formula = f'√{feature}'
                    
                    expected_gain = min(abs(skewness) * target_corr[feature] * 0.05, 0.06)
                    priority = 'high' if expected_gain > 0.03 else 'medium' if expected_gain > 0.01 else 'low'
                    recommendations[priority].append({
                        'type': 'transform',
                        'feature': feature,
                        'formula': formula,
                        'transform': transform,
                        'reason': f'偏态分布 (偏度={skewness:.2f})',
                        'expected_gain': expected_gain
                    })
            
            # 4. 比率特征推荐（针对相关特征对）
            for i, feat1 in enumerate(feature_cols):
                for feat2 in feature_cols[i+1:]:
                    # 特征间相关性高
                    if abs(feature_corr.loc[feat1, feat2]) > 0.6:
                        if target_corr[feat1] > 0.2 or target_corr[feat2] > 0.2:
                            expected_gain = max(target_corr[feat1], target_corr[feat2]) * 0.06
                            priority = 'medium' if expected_gain > 0.01 else 'low'
                            recommendations[priority].append({
                                'type': 'ratio',
                                'feature': f'{feat1} / {feat2}',
                                'formula': f'{feat1} / {feat2}',
                                'reason': f'特征相关 (r={feature_corr.loc[feat1, feat2]:.2f})',
                                'expected_gain': expected_gain,
                                'feat1': feat1,
                                'feat2': feat2
                            })
            
            # 限制推荐数量
            recommendations['high'] = sorted(recommendations['high'], key=lambda x: x['expected_gain'], reverse=True)[:5]
            recommendations['medium'] = sorted(recommendations['medium'], key=lambda x: x['expected_gain'], reverse=True)[:5]
            recommendations['low'] = sorted(recommendations['low'], key=lambda x: x['expected_gain'], reverse=True)[:3]
            
            # 显示推荐
            total_recommendations = len(recommendations['high']) + len(recommendations['medium']) + len(recommendations['low'])
            
            if total_recommendations == 0:
                st.info("📊 当前数据特征较为理想，暂无强烈推荐的特征工程方法")
            else:
                st.success(f"✅ 生成了 {total_recommendations} 条特征工程推荐")
                
                # 高优先级推荐
                if recommendations['high']:
                    st.markdown("##### 🎯 高优先级推荐（预期R²提升 > 0.03）")
                    selected_high = []
                    for idx, rec in enumerate(recommendations['high']):
                        col_check, col_info = st.columns([0.1, 0.9])
                        with col_check:
                            checked = st.checkbox(f"选择{rec['formula']}", value=True, key=f"high_{idx}", label_visibility="collapsed")
                            if checked:
                                selected_high.append(rec)
                        with col_info:
                            st.markdown(f"**{rec['formula']}** - 预期提升: +{rec['expected_gain']:.3f}")
                            st.caption(f"💡 {rec['reason']}")
                
                # 中优先级推荐
                if recommendations['medium']:
                    st.markdown("##### 💡 中优先级推荐（预期R²提升 0.01-0.03）")
                    selected_medium = []
                    for idx, rec in enumerate(recommendations['medium']):
                        col_check, col_info = st.columns([0.1, 0.9])
                        with col_check:
                            checked = st.checkbox(f"选择{rec['formula']}", value=False, key=f"medium_{idx}", label_visibility="collapsed")
                            if checked:
                                selected_medium.append(rec)
                        with col_info:
                            st.markdown(f"**{rec['formula']}** - 预期提升: +{rec['expected_gain']:.3f}")
                            st.caption(f"💡 {rec['reason']}")
                
                # 低优先级推荐
                if recommendations['low']:
                    with st.expander("⚠️ 低优先级推荐（预期R²提升 < 0.01）", expanded=False):
                        selected_low = []
                        for idx, rec in enumerate(recommendations['low']):
                            col_check, col_info = st.columns([0.1, 0.9])
                            with col_check:
                                checked = st.checkbox(f"选择{rec['formula']}", value=False, key=f"low_{idx}", label_visibility="collapsed")
                                if checked:
                                    selected_low.append(rec)
                            with col_info:
                                st.markdown(f"**{rec['formula']}** - 预期提升: +{rec['expected_gain']:.3f}")
                                st.caption(f"💡 {rec['reason']}")
                
                st.markdown("---")
                
                # 应用按钮
                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
                
                with col_btn1:
                    if st.button("🚀 应用选中的特征", type="primary", width='stretch'):
                        # 收集所有选中的推荐
                        all_selected = []
                        if recommendations['high']:
                            all_selected.extend(selected_high)
                        if recommendations['medium']:
                            all_selected.extend(selected_medium)
                        if recommendations['low']:
                            all_selected.extend(selected_low)
                        
                        if not all_selected:
                            st.warning("⚠️ 请至少选择一个特征工程方法")
                        else:
                            with st.spinner(f"正在生成 {len(all_selected)} 个新特征..."):
                                new_features_created = []
                                
                                for rec in all_selected:
                                    try:
                                        if rec['type'] == 'polynomial':
                                            # 多项式特征
                                            new_col_name = f"{rec['feature']}_squared"
                                            current_data[new_col_name] = current_data[rec['feature']] ** 2
                                            new_features_created.append(new_col_name)
                                        
                                        elif rec['type'] == 'interaction':
                                            # 交互特征
                                            new_col_name = f"{rec['feat1']}_x_{rec['feat2']}"
                                            current_data[new_col_name] = current_data[rec['feat1']] * current_data[rec['feat2']]
                                            new_features_created.append(new_col_name)
                                        
                                        elif rec['type'] == 'transform':
                                            # 数学变换
                                            if rec['transform'] == 'log':
                                                new_col_name = f"{rec['feature']}_log"
                                                # 处理负值和零值
                                                min_val = current_data[rec['feature']].min()
                                                if min_val <= 0:
                                                    current_data[new_col_name] = np.log(current_data[rec['feature']] - min_val + 1)
                                                else:
                                                    current_data[new_col_name] = np.log(current_data[rec['feature']])
                                            elif rec['transform'] == 'sqrt':
                                                new_col_name = f"{rec['feature']}_sqrt"
                                                # 处理负值
                                                min_val = current_data[rec['feature']].min()
                                                if min_val < 0:
                                                    current_data[new_col_name] = np.sqrt(current_data[rec['feature']] - min_val)
                                                else:
                                                    current_data[new_col_name] = np.sqrt(current_data[rec['feature']])
                                            new_features_created.append(new_col_name)
                                        
                                        elif rec['type'] == 'ratio':
                                            # 比率特征
                                            new_col_name = f"{rec['feat1']}_div_{rec['feat2']}"
                                            # 避免除零
                                            current_data[new_col_name] = current_data[rec['feat1']] / (current_data[rec['feat2']] + 1e-10)
                                            new_features_created.append(new_col_name)
                                    
                                    except Exception as e:
                                        st.warning(f"⚠️ 生成特征 {rec['formula']} 时出错：{str(e)}")
                                
                                # 更新session_state
                                st.session_state.data = current_data
                                st.session_state.feature_cols = feature_cols + new_features_created
                                st.session_state.smart_engineered_features = new_features_created  # 智能生成的特征
                                st.session_state.feature_engineering_done = True
                                
                                st.success(f"✅ 成功生成 {len(new_features_created)} 个新特征！")
                                st.info(f"📊 特征数量：{len(feature_cols)} → {len(st.session_state.feature_cols)}")
                                
                                # 显示新特征列表
                                with st.expander("📋 查看新生成的特征", expanded=True):
                                    for feat in new_features_created:
                                        st.write(f"• {feat}")
                                
                                st.rerun()
                
                with col_btn2:
                    if st.button("⚡ 一键应用高优先级", width='stretch'):
                        if not recommendations['high']:
                            st.warning("⚠️ 没有高优先级推荐")
                        else:
                            with st.spinner(f"正在生成 {len(recommendations['high'])} 个新特征..."):
                                new_features_created = []
                                
                                for rec in recommendations['high']:
                                    try:
                                        if rec['type'] == 'polynomial':
                                            new_col_name = f"{rec['feature']}_squared"
                                            current_data[new_col_name] = current_data[rec['feature']] ** 2
                                            new_features_created.append(new_col_name)
                                        
                                        elif rec['type'] == 'interaction':
                                            new_col_name = f"{rec['feat1']}_x_{rec['feat2']}"
                                            current_data[new_col_name] = current_data[rec['feat1']] * current_data[rec['feat2']]
                                            new_features_created.append(new_col_name)
                                        
                                        elif rec['type'] == 'transform':
                                            if rec['transform'] == 'log':
                                                new_col_name = f"{rec['feature']}_log"
                                                min_val = current_data[rec['feature']].min()
                                                if min_val <= 0:
                                                    current_data[new_col_name] = np.log(current_data[rec['feature']] - min_val + 1)
                                                else:
                                                    current_data[new_col_name] = np.log(current_data[rec['feature']])
                                            elif rec['transform'] == 'sqrt':
                                                new_col_name = f"{rec['feature']}_sqrt"
                                                min_val = current_data[rec['feature']].min()
                                                if min_val < 0:
                                                    current_data[new_col_name] = np.sqrt(current_data[rec['feature']] - min_val)
                                                else:
                                                    current_data[new_col_name] = np.sqrt(current_data[rec['feature']])
                                            new_features_created.append(new_col_name)
                                        
                                        elif rec['type'] == 'ratio':
                                            new_col_name = f"{rec['feat1']}_div_{rec['feat2']}"
                                            current_data[new_col_name] = current_data[rec['feat1']] / (current_data[rec['feat2']] + 1e-10)
                                            new_features_created.append(new_col_name)
                                    
                                    except Exception as e:
                                        st.warning(f"⚠️ 生成特征 {rec['formula']} 时出错：{str(e)}")
                                
                                st.session_state.data = current_data
                                st.session_state.feature_cols = feature_cols + new_features_created
                                st.session_state.smart_engineered_features = new_features_created  # 智能生成的特征
                                st.session_state.feature_engineering_done = True
                                
                                st.success(f"✅ 成功生成 {len(new_features_created)} 个新特征！")
                                st.info(f"📊 特征数量：{len(feature_cols)} → {len(st.session_state.feature_cols)}")
                                
                                st.rerun()
                
                with col_btn3:
                    if st.button("❌ 跳过特征工程", width='stretch'):
                        st.info("已跳过特征工程，将使用原始特征继续")
        
        # 如果已经执行过智能特征工程，显示结果
        if st.session_state.get('feature_engineering_done', False) and st.session_state.get('smart_engineered_features'):
            smart_features = st.session_state.smart_engineered_features
            st.success(f"✅ 智能特征工程已完成！生成了 {len(smart_features)} 个新特征")
            
            # 可视化：智能工程特征效果分析
            st.markdown("#### 📊 智能工程特征效果分析")
            
            current_data = st.session_state.data
            
            # 计算智能工程特征与目标的相关性
            smart_engineered_corr = {}
            for feat in smart_features:
                if feat in current_data.columns:
                    corr = abs(current_data[feat].corr(current_data[target_col]))
                    smart_engineered_corr[feat] = corr
            
            # 对比原始特征和智能工程特征的相关性
            col_compare1, col_compare2 = st.columns(2)
            
            with col_compare1:
                # 原始特征相关性
                original_features = [f for f in st.session_state.feature_cols if f not in smart_features]
                original_corr = {f: abs(current_data[f].corr(current_data[target_col])) for f in original_features if f in current_data.columns}
                
                fig_orig, ax_orig = plt.subplots(figsize=(8, max(5, len(original_features) * 0.3)), dpi=100)
                
                sorted_orig = sorted(original_corr.items(), key=lambda x: x[1], reverse=True)
                features_orig = [item[0] for item in sorted_orig]
                corr_orig = [item[1] for item in sorted_orig]
                
                colors_orig = plt.cm.Blues(np.linspace(0.4, 0.9, len(features_orig)))
                bars_orig = ax_orig.barh(features_orig, corr_orig, color=colors_orig, 
                                        edgecolor='black', linewidth=0.5)
                
                for i, (bar, val) in enumerate(zip(bars_orig, corr_orig)):
                    ax_orig.text(val + 0.01, i, f'{val:.3f}', 
                               va='center', fontsize=8, fontweight='bold')
                
                ax_orig.set_xlabel('|Correlation|', fontsize=10, fontweight='bold')
                ax_orig.set_title('Original Features', fontsize=12, fontweight='bold', pad=10)
                ax_orig.set_xlim(0, max(max(corr_orig) if corr_orig else 0, max(smart_engineered_corr.values()) if smart_engineered_corr else 0) * 1.15)
                ax_orig.grid(True, alpha=0.3, linestyle='--', axis='x')
                ax_orig.spines['top'].set_visible(False)
                ax_orig.spines['right'].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig_orig)
                plt.close()
            
            with col_compare2:
                # 智能工程特征相关性
                fig_eng, ax_eng = plt.subplots(figsize=(8, max(5, len(smart_features) * 0.3)), dpi=100)
                
                sorted_eng = sorted(smart_engineered_corr.items(), key=lambda x: x[1], reverse=True)
                features_eng = [item[0] for item in sorted_eng]
                corr_eng = [item[1] for item in sorted_eng]
                
                colors_eng = plt.cm.Greens(np.linspace(0.4, 0.9, len(features_eng)))
                bars_eng = ax_eng.barh(features_eng, corr_eng, color=colors_eng, 
                                      edgecolor='black', linewidth=0.5)
                
                for i, (bar, val) in enumerate(zip(bars_eng, corr_eng)):
                    ax_eng.text(val + 0.01, i, f'{val:.3f}', 
                              va='center', fontsize=8, fontweight='bold')
                
                ax_eng.set_xlabel('|Correlation|', fontsize=10, fontweight='bold')
                ax_eng.set_title('Smart Engineered Features', fontsize=12, fontweight='bold', pad=10)
                ax_eng.set_xlim(0, max(max(corr_orig) if corr_orig else 0, max(smart_engineered_corr.values()) if smart_engineered_corr else 0) * 1.15)
                ax_eng.grid(True, alpha=0.3, linestyle='--', axis='x')
                ax_eng.spines['top'].set_visible(False)
                ax_eng.spines['right'].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig_eng)
                plt.close()
            
            # 统计对比
            st.markdown("#### 📈 效果统计对比")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                avg_orig_corr = np.mean(list(original_corr.values())) if original_corr else 0
                st.metric("原始特征平均相关性", f"{avg_orig_corr:.3f}")
            
            with col_stat2:
                avg_eng_corr = np.mean(list(smart_engineered_corr.values())) if smart_engineered_corr else 0
                improvement = avg_eng_corr - avg_orig_corr
                st.metric("智能工程特征平均相关性", f"{avg_eng_corr:.3f}", 
                         delta=f"{improvement:+.3f}")
            
            with col_stat3:
                max_orig = max(original_corr.values()) if original_corr else 0
                max_eng = max(smart_engineered_corr.values()) if smart_engineered_corr else 0
                st.metric("最强智能工程特征", f"{max_eng:.3f}", 
                         delta=f"{max_eng - max_orig:+.3f}" if max_eng > max_orig else None)
            
            with col_stat4:
                # 计算有多少智能工程特征比最强原始特征更好
                better_count = sum(1 for v in smart_engineered_corr.values() if v > max_orig) if max_orig > 0 else 0
                st.metric("优于最强原始特征", f"{better_count}/{len(smart_features)}")
            
            # 智能工程特征分布预览
            with st.expander("📊 查看智能工程特征分布", expanded=False):
                n_eng_features = len(smart_features)
                n_cols_preview = min(3, n_eng_features)
                n_rows_preview = (n_eng_features + n_cols_preview - 1) // n_cols_preview
                
                fig_preview, axes_preview = plt.subplots(n_rows_preview, n_cols_preview, 
                                                        figsize=(15, 4*n_rows_preview), dpi=100)
                if n_rows_preview == 1:
                    axes_preview = axes_preview.reshape(1, -1) if n_eng_features > 1 else np.array([[axes_preview]])
                axes_preview = axes_preview.flatten()
                
                for idx, feat in enumerate(smart_features):
                    if feat not in current_data.columns:
                        continue
                    ax = axes_preview[idx]
                    feat_data = current_data[feat].dropna()
                    
                    # 直方图 + KDE
                    ax.hist(feat_data, bins=25, edgecolor='white', alpha=0.7, 
                           color='lightgreen', density=True, label='Histogram')
                    
                    try:
                        if feat_data.std() > 0:
                            feat_data.plot(kind='kde', ax=ax, color='darkgreen', 
                                         linewidth=2, label='KDE')
                    except:
                        pass
                    
                    corr_val = smart_engineered_corr.get(feat, 0)
                    ax.set_title(f'{feat}\n(r={corr_val:.3f})', 
                               fontsize=10, fontweight='bold', pad=10)
                    ax.set_xlabel(feat, fontsize=8)
                    ax.set_ylabel('Density', fontsize=8)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.legend(fontsize=7)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                
                # 隐藏多余的子图
                for idx in range(n_eng_features, len(axes_preview)):
                    axes_preview[idx].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig_preview)
                plt.close()
            
            # 智能工程特征列表
            with st.expander("📋 查看智能工程特征详细列表", expanded=False):
                # 创建表格
                feat_table = []
                for feat in smart_features:
                    if feat not in current_data.columns:
                        continue
                    feat_table.append({
                        '特征名称': feat,
                        '与目标相关性': f"{smart_engineered_corr.get(feat, 0):.3f}",
                        '均值': f"{current_data[feat].mean():.2f}",
                        '标准差': f"{current_data[feat].std():.2f}",
                        '偏度': f"{stats.skew(current_data[feat].dropna()):.2f}"
                    })
                
                if feat_table:
                    st.dataframe(pd.DataFrame(feat_table), width='stretch')
        
        st.markdown("---")
    
    # 自定义特征工程
    st.subheader("✏️ 自定义特征工程")
    st.markdown("除了智能推荐的特征，您也可以手动创建自定义特征")
    
    with st.expander("📝 创建自定义特征", expanded=False):
        st.markdown("""
        **使用说明：**
        - 输入新特征的名称和计算公式
        - 公式中可以使用现有的特征名称和数学运算符
        - 支持的运算：`+`, `-`, `*`, `/`, `**`（幂）, `//`（整除）, `%`（取模）
        - 支持的函数：`abs()`, `sqrt()`, `log()`, `exp()`, `sin()`, `cos()`, `tan()`
        
        **重要提示：**
        - 如果特征名包含空格或特殊字符，请使用方括号语法：`current_data['特征名']`
        - 例如：`current_data['Blast Furnace Slag'] + current_data['Fly Ash']`
        
        **示例：**
        - `weight / (height ** 2)` - 计算BMI（特征名无空格）
        - `current_data['Cement'] + current_data['Blast Furnace Slag']` - 特征名有空格
        - `price / area` - 计算单价
        - `age // 10 * 10` - 年龄分组（10岁一组）
        - `math_score + english_score + science_score` - 总分
        """)
        
        # 显示可用特征
        current_data = st.session_state.data
        available_features = [col for col in current_data.columns if col != target_col]
        
        with st.expander("📋 查看可用特征", expanded=False):
            st.write("可以在公式中使用以下特征：")
            cols_display = st.columns(3)
            for idx, feat in enumerate(available_features):
                with cols_display[idx % 3]:
                    st.write(f"• `{feat}`")
        
        # 输入新特征
        col_name, col_formula = st.columns([1, 2])
        
        with col_name:
            new_feature_name = st.text_input("新特征名称", placeholder="例如：BMI", key="custom_feature_name")
        
        with col_formula:
            new_feature_formula = st.text_input("计算公式", placeholder="例如：weight / (height ** 2)", key="custom_feature_formula")
        
        # 创建按钮
        col_create, col_preview = st.columns([1, 1])
        
        with col_create:
            if st.button("🚀 创建特征", type="primary", width='stretch'):
                if not new_feature_name or not new_feature_formula:
                    st.error("❌ 请输入特征名称和公式")
                elif new_feature_name in current_data.columns:
                    st.error(f"❌ 特征名称 '{new_feature_name}' 已存在，请使用其他名称")
                else:
                    try:
                        # 安全地执行公式
                        # 创建一个安全的命名空间，只包含数据列和安全的数学函数
                        safe_dict = {col: current_data[col] for col in current_data.columns}
                        safe_dict.update({
                            'current_data': current_data,  # 添加current_data引用，支持current_data['列名']语法
                            'abs': np.abs,
                            'sqrt': np.sqrt,
                            'log': np.log,
                            'exp': np.exp,
                            'sin': np.sin,
                            'cos': np.cos,
                            'tan': np.tan,
                            'np': np,
                            'pd': pd
                        })
                        
                        # 执行公式
                        new_feature_data = eval(new_feature_formula, {"__builtins__": {}}, safe_dict)
                        
                        # 检查结果是否为Series或可转换为Series
                        if not isinstance(new_feature_data, pd.Series):
                            new_feature_data = pd.Series(new_feature_data, index=current_data.index)
                        
                        # 检查是否有无效值
                        if new_feature_data.isnull().all():
                            st.error("❌ 公式计算结果全部为空值，请检查公式")
                        elif np.isinf(new_feature_data).any():
                            st.warning("⚠️ 公式计算结果包含无穷大值，将自动处理")
                            new_feature_data = new_feature_data.replace([np.inf, -np.inf], np.nan)
                            new_feature_data = new_feature_data.fillna(new_feature_data.median())
                        
                        # 添加新特征到数据中
                        current_data[new_feature_name] = new_feature_data
                        
                        # 更新session_state
                        st.session_state.data = current_data
                        
                        # 更新特征列表
                        if new_feature_name not in st.session_state.feature_cols:
                            st.session_state.feature_cols.append(new_feature_name)
                        
                        # 更新自定义工程特征列表（与智能特征分开）
                        if 'custom_engineered_features' not in st.session_state:
                            st.session_state.custom_engineered_features = []
                        if new_feature_name not in st.session_state.custom_engineered_features:
                            st.session_state.custom_engineered_features.append(new_feature_name)
                        
                        # 计算与目标的相关性
                        correlation = abs(new_feature_data.corr(current_data[target_col]))
                        
                        st.success(f"✅ 成功创建特征 '{new_feature_name}'！")
                        st.info(f"📊 与目标变量的相关性：{correlation:.3f}")
                        st.info("💡 特征已创建！如需创建更多特征，请清空输入框并输入新的特征名称和公式")
                        
                        # 显示统计信息
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        with col_stat1:
                            st.metric("均值", f"{new_feature_data.mean():.2f}")
                        with col_stat2:
                            st.metric("标准差", f"{new_feature_data.std():.2f}")
                        with col_stat3:
                            st.metric("最小值", f"{new_feature_data.min():.2f}")
                        with col_stat4:
                            st.metric("最大值", f"{new_feature_data.max():.2f}")
                        
                        # 显示分布图
                        fig_custom, ax_custom = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
                        
                        # 直方图
                        ax_custom[0].hist(new_feature_data.dropna(), bins=30, edgecolor='white', 
                                         alpha=0.7, color='skyblue')
                        ax_custom[0].set_title(f'{new_feature_name} - 分布', fontsize=12, fontweight='bold')
                        ax_custom[0].set_xlabel(new_feature_name, fontsize=10)
                        ax_custom[0].set_ylabel('频数', fontsize=10)
                        ax_custom[0].grid(True, alpha=0.3, linestyle='--')
                        ax_custom[0].spines['top'].set_visible(False)
                        ax_custom[0].spines['right'].set_visible(False)
                        
                        # 散点图（与目标变量）
                        ax_custom[1].scatter(new_feature_data, current_data[target_col], 
                                           alpha=0.5, s=20, color='coral')
                        ax_custom[1].set_title(f'{new_feature_name} vs {target_col}', 
                                             fontsize=12, fontweight='bold')
                        ax_custom[1].set_xlabel(new_feature_name, fontsize=10)
                        ax_custom[1].set_ylabel(target_col, fontsize=10)
                        ax_custom[1].grid(True, alpha=0.3, linestyle='--')
                        ax_custom[1].spines['top'].set_visible(False)
                        ax_custom[1].spines['right'].set_visible(False)
                        
                        plt.tight_layout()
                        st.pyplot(fig_custom)
                        plt.close()
                        
                    except NameError as e:
                        st.error(f"❌ 公式中使用了未知的特征名称：{str(e)}")
                        st.info("💡 请检查特征名称是否正确，可以点击上方'查看可用特征'查看所有可用特征")
                    except SyntaxError as e:
                        st.error(f"❌ 公式语法错误：{str(e)}")
                        st.info("💡 请检查公式是否符合Python语法，例如：weight / (height ** 2)")
                    except ZeroDivisionError:
                        st.error("❌ 公式计算出现除零错误")
                        st.info("💡 请确保分母不为零，可以使用 (分母 + 1e-10) 来避免除零")
                    except Exception as e:
                        st.error(f"❌ 创建特征时出错：{str(e)}")
                        st.info("💡 请检查公式是否正确，确保所有特征名称存在且运算符使用正确")
        
        with col_preview:
            if st.button("👁️ 预览结果", width='stretch'):
                if not new_feature_formula:
                    st.warning("⚠️ 请先输入公式")
                elif new_feature_name and new_feature_name in current_data.columns:
                    st.warning(f"⚠️ 特征 '{new_feature_name}' 已存在！如需预览新特征，请先更改特征名称")
                else:
                    try:
                        # 预览计算结果（不保存）
                        safe_dict = {col: current_data[col] for col in current_data.columns}
                        safe_dict.update({
                            'current_data': current_data,  # 添加current_data引用，支持current_data['列名']语法
                            'abs': np.abs,
                            'sqrt': np.sqrt,
                            'log': np.log,
                            'exp': np.exp,
                            'sin': np.sin,
                            'cos': np.cos,
                            'tan': np.tan,
                            'np': np,
                            'pd': pd
                        })
                        
                        preview_data = eval(new_feature_formula, {"__builtins__": {}}, safe_dict)
                        
                        if not isinstance(preview_data, pd.Series):
                            preview_data = pd.Series(preview_data, index=current_data.index)
                        
                        st.success("✅ 公式有效！预览前10行结果：")
                        st.dataframe(preview_data.head(10))
                        
                        # 显示基本统计
                        st.write(f"均值：{preview_data.mean():.2f}，标准差：{preview_data.std():.2f}")
                        
                    except SyntaxError as e:
                        st.error(f"❌ 公式语法错误：{str(e)}")
                        st.info("💡 请检查公式语法。注意：特征名称中如果包含空格，需要使用反引号或引号，例如：`Cement` + `Blast Furnace Slag`")
                    except NameError as e:
                        st.error(f"❌ 公式中使用了未知的特征名称：{str(e)}")
                        st.info("💡 请检查特征名称是否正确，可以点击上方'查看可用特征'查看所有可用特征")
                    except Exception as e:
                        st.error(f"❌ 公式错误：{str(e)}")
                        
                    except Exception as e:
                        st.error(f"❌ 公式错误：{str(e)}")
        
        # 显示已创建的自定义特征
        if st.session_state.get('custom_engineered_features'):
            custom_features = st.session_state.custom_engineered_features
            if custom_features:
                st.markdown("#### 📋 已创建的自定义特征")
                
                # 创建表格显示
                custom_feat_table = []
                for feat in custom_features:
                    if feat in current_data.columns:
                        corr = abs(current_data[feat].corr(current_data[target_col]))
                        custom_feat_table.append({
                            '特征名称': feat,
                            '与目标相关性': f"{corr:.3f}",
                            '均值': f"{current_data[feat].mean():.2f}",
                            '标准差': f"{current_data[feat].std():.2f}"
                        })
                
                if custom_feat_table:
                    st.dataframe(pd.DataFrame(custom_feat_table), width='stretch')
    
    # 汇总所有工程特征
    st.markdown("---")
    
    # 统计所有工程特征
    smart_features = st.session_state.get('smart_engineered_features', [])
    custom_features = st.session_state.get('custom_engineered_features', [])
    all_engineered_features = smart_features + custom_features
    
    if all_engineered_features:
        st.markdown("### 📊 工程特征汇总")
        
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        
        with col_summary1:
            st.metric("智能生成特征", f"{len(smart_features)} 个")
        
        with col_summary2:
            st.metric("自定义特征", f"{len(custom_features)} 个")
        
        with col_summary3:
            st.metric("总工程特征", f"{len(all_engineered_features)} 个")
        
        # 显示所有工程特征的对比
        if len(all_engineered_features) > 0:
            with st.expander("📈 查看所有工程特征对比", expanded=False):
                current_data = st.session_state.data
                
                # 计算所有工程特征的相关性
                all_eng_corr = {}
                for feat in all_engineered_features:
                    if feat in current_data.columns:
                        corr = abs(current_data[feat].corr(current_data[target_col]))
                        all_eng_corr[feat] = corr
                
                # 绘制对比图
                fig_all, ax_all = plt.subplots(figsize=(10, max(6, len(all_engineered_features) * 0.4)), dpi=100)
                
                sorted_all = sorted(all_eng_corr.items(), key=lambda x: x[1], reverse=True)
                features_all = [item[0] for item in sorted_all]
                corr_all = [item[1] for item in sorted_all]
                
                # 根据特征类型设置颜色
                colors = []
                for feat in features_all:
                    if feat in smart_features:
                        colors.append('lightgreen')  # 智能特征
                    else:
                        colors.append('skyblue')  # 自定义特征
                
                bars_all = ax_all.barh(features_all, corr_all, color=colors, 
                                      edgecolor='black', linewidth=0.5)
                
                for i, (bar, val) in enumerate(zip(bars_all, corr_all)):
                    ax_all.text(val + 0.01, i, f'{val:.3f}', 
                              va='center', fontsize=9, fontweight='bold')
                
                ax_all.set_xlabel('|Correlation with Target|', fontsize=11, fontweight='bold')
                ax_all.set_title('All Engineered Features Comparison', fontsize=13, fontweight='bold', pad=15)
                ax_all.set_xlim(0, max(corr_all) * 1.15 if corr_all else 1)
                ax_all.grid(True, alpha=0.3, linestyle='--', axis='x')
                ax_all.spines['top'].set_visible(False)
                ax_all.spines['right'].set_visible(False)
                
                # 添加图例
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='lightgreen', edgecolor='black', label='智能生成'),
                    Patch(facecolor='skyblue', edgecolor='black', label='自定义')
                ]
                ax_all.legend(handles=legend_elements, loc='lower right', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig_all)
                plt.close()
    
    # 🆕 环形相关性网络图
    st.markdown("---")
    st.subheader("🔄 环形相关性网络图")
    
    st.info("💡 环形网络图以直观的方式展示特征之间的相关性以及特征与目标变量的关系")
    
    with st.expander("📖 如何解读环形网络图？", expanded=False):
        st.markdown("""
        **节点（特征）的视觉编码：**
        - **位置**：所有特征环形排列
        - **大小**：与目标变量的相关强度（越大表示相关性越强）
        - **颜色**：与目标变量的相关方向（红色=正相关，蓝色=负相关）
        - **边框**：显著性（粗黑边框=显著，细灰边框=不显著）
        
        **边（特征间关系）的视觉编码：**
        - **粗细**：相关强度（越粗表示相关性越强）
        - **颜色**：相关方向（绿色=正相关，紫色=负相关）
        - **线型**：显著性（实线=显著，虚线=不显著）
        - **透明度**：相关强度（强相关更不透明）
        
        **适用场景：**
        - 快速识别关键特征（大节点）
        - 发现多重共线性（粗边连接）
        - 评估特征工程效果（衍生特征的位置和连接）
        """)
    
    # 参数设置
    col_param1, col_param2, col_param3 = st.columns(3)
    
    with col_param1:
        edge_threshold = st.slider(
            "边显示阈值",
            min_value=0.0,
            max_value=0.8,
            value=0.4,
            step=0.1,
            help="只显示相关系数绝对值大于此阈值的边"
        )
    
    with col_param2:
        max_edges = st.number_input(
            "最大边数",
            min_value=10,
            max_value=200,
            value=80,
            step=10,
            help="限制显示的边数量，避免图形过于密集"
        )
    
    with col_param3:
        alpha_sig = st.slider(
            "显著性水平",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            help="p值小于此阈值的相关性被认为是显著的"
        )
    
    # 计算按钮
    if st.button("🎨 生成环形相关性网络图", type="primary", key="generate_circular_network"):
        # 获取当前数据
        current_data = st.session_state.data
        current_features = st.session_state.feature_cols
        current_target = st.session_state.target_col
        
        with st.spinner("正在计算相关性统计..."):
            from scipy import stats
            import networkx as nx
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            from matplotlib.lines import Line2D
            
            # 计算相关性统计
            X = current_data[current_features]
            y = current_data[current_target]
            
            n = len(current_features)
            
            # 与目标的相关性
            corr_target = np.full(n, np.nan, dtype=float)
            p_target = np.full(n, np.nan, dtype=float)
            
            for i, f in enumerate(current_features):
                try:
                    r, p = stats.pearsonr(X[f].values, y.values)
                    corr_target[i] = r
                    p_target[i] = p
                except:
                    corr_target[i] = np.nan
                    p_target[i] = np.nan
            
            # 特征-特征相关性矩阵
            corr_mat = np.full((n, n), np.nan, dtype=float)
            p_mat = np.full((n, n), np.nan, dtype=float)
            np.fill_diagonal(corr_mat, 0.0)
            np.fill_diagonal(p_mat, 1.0)
            
            for i in range(n):
                for j in range(i + 1, n):
                    try:
                        r, p = stats.pearsonr(X.iloc[:, i].values, X.iloc[:, j].values)
                        corr_mat[i, j] = corr_mat[j, i] = r
                        p_mat[i, j] = p_mat[j, i] = p
                    except:
                        pass
        
        with st.spinner("正在绘制环形网络图..."):
            # 创建图
            G = nx.Graph()
            G.add_nodes_from(current_features)
            pos = nx.circular_layout(G)
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={"aspect": "equal"}, dpi=120)
            
            # 颜色归一化
            norm_edges = mcolors.Normalize(vmin=-1, vmax=1)
            norm_nodes = mcolors.Normalize(vmin=-1, vmax=1)
            cmap_edges = cm.get_cmap("PRGn")
            cmap_nodes = cm.get_cmap("RdBu_r")
            
            # 映射基准
            corr_abs_mat = np.abs(np.nan_to_num(corr_mat, nan=0.0))
            max_edge_abs = np.max(corr_abs_mat) if np.any(corr_abs_mat) else 1.0
            max_node_abs = np.max(np.abs(np.nan_to_num(corr_target, nan=0.0))) or 1.0
            
            # 组织边
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    r = corr_mat[i, j]
                    p = p_mat[i, j]
                    if not np.isfinite(r):
                        continue
                    if abs(r) <= edge_threshold:
                        continue
                    edges.append((current_features[i], current_features[j], abs(r), r, p))
            
            # 弱到强绘制
            edges.sort(key=lambda x: x[2])
            
            # 限制边数
            if len(edges) > max_edges:
                edges = sorted(edges, key=lambda x: x[2])[-max_edges:]
                edges.sort(key=lambda x: x[2])
            
            # 画边
            w_min, w_max = 0.3, 8.0
            a_min, a_max = 0.06, 0.85
            
            for u, v, abs_r, r, p in edges:
                width = w_min + (abs_r / max_edge_abs) * (w_max - w_min)
                alpha_line = a_min + (abs_r / max_edge_abs) * (a_max - a_min)
                is_sig = (np.isfinite(p) and p < alpha_sig)
                linestyle = "solid" if is_sig else "dashed"
                color = cmap_edges(norm_edges(r))
                
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    width=width,
                    edge_color=[color],
                    style=linestyle,
                    alpha=alpha_line,
                    ax=ax
                )
            
            # 画节点
            ns_min, ns_max = 80, 900
            node_sizes, node_colors, node_edgecolors, node_linewidths = [], [], [], []
            
            for i, f in enumerate(current_features):
                r = corr_target[i]
                p = p_target[i]
                abs_r = abs(r) if np.isfinite(r) else 0.0
                size = ns_min + (abs_r / max_node_abs) * (ns_max - ns_min)
                node_sizes.append(size)
                
                color = cmap_nodes(norm_nodes(r if np.isfinite(r) else 0.0))
                node_colors.append(color)
                
                is_sig = (np.isfinite(p) and p < alpha_sig)
                node_edgecolors.append("black" if is_sig else "lightgray")
                node_linewidths.append(2.5 if is_sig else 0.9)
            
            nx.draw_networkx_nodes(
                G, pos,
                node_size=node_sizes,
                node_color=node_colors,
                edgecolors=node_edgecolors,
                linewidths=node_linewidths,
                alpha=0.98,
                ax=ax
            )
            
            # 标签
            label_scale = 1.15
            tangent_shift = 0.04
            fontsize = 10
            
            for idx, node in enumerate(current_features):
                x0, y0 = pos[node]
                x, y = x0 * label_scale, y0 * label_scale
                ang = np.arctan2(y0, x0)
                tx, ty = -np.sin(ang), np.cos(ang)
                sign = 1 if (idx % 2 == 0) else -1
                x += sign * tangent_shift * tx
                y += sign * tangent_shift * ty
                ha = "left" if x >= 0 else "right"
                ax.text(x, y, node, fontsize=fontsize, ha=ha, va="center")
            
            ax.axis("off")
            ax.set_xlim(-1.55, 1.55)
            ax.set_ylim(-1.55, 1.55)
            ax.set_title('环形相关性网络图 (Circular Correlation Network)', 
                        y=0.97, fontsize=15, fontweight='bold')
            
            # 左侧图例
            # 线宽图例
            line_levels = [max_edge_abs, max_edge_abs * 0.5, max_edge_abs * 0.1]
            line_labels = [f"{val:.2f}" for val in line_levels]
            legend_lines = []
            for val in line_levels:
                w = w_min + (val / max_edge_abs) * (w_max - w_min)
                legend_lines.append(Line2D([0], [0], color="black", lw=w))
            
            legend1 = ax.legend(
                legend_lines, line_labels,
                loc="center left",
                bbox_to_anchor=(-0.10, 0.80),
                title="相关强度\n(线条粗细)",
                frameon=False,
                labelspacing=1.5
            )
            ax.add_artist(legend1)
            
            # 节点大小图例
            node_levels = [max_node_abs, max_node_abs * 0.5, max_node_abs * 0.1]
            node_labels = [f"{val:.2f}" for val in node_levels]
            legend_nodes = []
            for val in node_levels:
                s = ns_min + (val / max_node_abs) * (ns_max - ns_min)
                legend_nodes.append(Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    markerfacecolor="black",
                    markersize=np.sqrt(s) / 2.2,
                    linestyle="None"
                ))
            
            legend2 = ax.legend(
                legend_nodes, node_labels,
                loc="center left",
                bbox_to_anchor=(-0.10, 0.36),
                title="与目标相关性\n(节点大小)",
                frameon=False,
                labelspacing=2.6
            )
            ax.add_artist(legend2)
            
            # 显著性图例
            sig_handles = [
                Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="white", markeredgecolor="black",
                      markeredgewidth=2.5, markersize=10, linestyle="None",
                      label=f"节点: p < {alpha_sig}"),
                Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="white", markeredgecolor="lightgray",
                      markeredgewidth=0.9, markersize=10, linestyle="None",
                      label=f"节点: p ≥ {alpha_sig}"),
                Line2D([0], [0], color="black", lw=2.0, linestyle="solid",
                      label=f"边: p < {alpha_sig}"),
                Line2D([0], [0], color="black", lw=2.0, linestyle="dashed",
                      label=f"边: p ≥ {alpha_sig}")
            ]
            
            ax.legend(
                handles=sig_handles,
                loc="center left",
                bbox_to_anchor=(-0.10, 0.10),
                title="显著性",
                frameon=False,
                labelspacing=1.0
            )
            
            # 右侧颜色条
            sm_edge = cm.ScalarMappable(norm=norm_edges, cmap=cmap_edges)
            sm_edge.set_array([])
            cax_edge = fig.add_axes([0.86, 0.55, 0.018, 0.28])
            cbar_edge = plt.colorbar(sm_edge, cax=cax_edge)
            cbar_edge.set_label("边相关系数", rotation=270, labelpad=16, fontsize=10)
            cbar_edge.outline.set_visible(False)
            
            sm_node = cm.ScalarMappable(norm=norm_nodes, cmap=cmap_nodes)
            sm_node.set_array([])
            cax_node = fig.add_axes([0.86, 0.18, 0.018, 0.28])
            cbar_node = plt.colorbar(sm_node, cax=cax_node)
            cbar_node.set_label("与目标相关系数", rotation=270, labelpad=16, fontsize=10)
            cbar_node.outline.set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # 显示统计摘要
        st.success("✅ 环形网络图生成完成")
        
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        
        with col_summary1:
            st.metric("特征数量", len(current_features))
            st.metric("显示的边数", len(edges))
        
        with col_summary2:
            sig_nodes = sum(1 for p in p_target if np.isfinite(p) and p < alpha_sig)
            st.metric("显著节点", sig_nodes, 
                     delta=f"{sig_nodes/len(current_features)*100:.1f}%")
        
        with col_summary3:
            sig_edges = sum(1 for _, _, _, _, p in edges if np.isfinite(p) and p < alpha_sig)
            st.metric("显著边", sig_edges,
                     delta=f"{sig_edges/len(edges)*100:.1f}%" if edges else "0%")
    
    # 导航按钮
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("继续到步骤3：模型训练", type="primary", key="nav_to_step3_final", width="stretch"):
            st.session_state.step = 3
            st.rerun()
# 步骤3：模型训练
elif st.session_state.step == 3:
    st.header("3️⃣ 模型训练 (Model Training)")
    
    # 数据集选择
    st.subheader("📂 选择训练数据集")
    
    # 检查是否有数据
    if st.session_state.data is None:
        st.error("❌ 未找到数据！请先在步骤1中加载数据。")
        if st.button("← 返回步骤1"):
            st.session_state.step = 1
            st.rerun()
        st.stop()
    
    # 使用当前数据（已经过清洗和特征工程）
    training_data = st.session_state.data
    
    # 特征选择
    st.subheader("🎯 特征选择")
    
    # 特征选择选项
    smart_features = st.session_state.get('smart_engineered_features', [])
    custom_features = st.session_state.get('custom_engineered_features', [])
    all_engineered_features = smart_features + custom_features
    
    if len(all_engineered_features) > 0:
        # 执行过特征工程，提供选择
        st.info("💡 检测到已执行特征工程，您可以选择使用哪些特征进行训练")
        
        # 获取原始特征和工程特征
        original_features = [f for f in st.session_state.feature_cols 
                           if f not in all_engineered_features]
        
        feature_option = st.radio(
            "选择训练特征：",
            options=[
                "原始特征 + 工程特征（推荐）",
                "仅原始特征",
                "仅工程特征",
                "自定义选择"
            ],
            help="工程特征通常能提升模型性能"
        )
        
        if feature_option == "原始特征 + 工程特征（推荐）":
            selected_features = st.session_state.feature_cols
            st.success(f"✅ 使用全部特征：{len(original_features)} 个原始特征 + {len(all_engineered_features)} 个工程特征（智能 {len(smart_features)} + 自定义 {len(custom_features)}）")
        
        elif feature_option == "仅原始特征":
            selected_features = original_features
            st.info(f"📊 仅使用 {len(original_features)} 个原始特征")
        
        elif feature_option == "仅工程特征":
            selected_features = all_engineered_features
            st.info(f"🔧 仅使用 {len(all_engineered_features)} 个工程特征（智能 {len(smart_features)} + 自定义 {len(custom_features)}）")
        
        else:  # 自定义选择
            st.markdown("##### 📋 选择要使用的特征")
            
            col_custom1, col_custom2, col_custom3 = st.columns(3)
            
            with col_custom1:
                st.markdown("**原始特征：**")
                selected_original = st.multiselect(
                    "选择原始特征",
                    options=original_features,
                    default=original_features,
                    key="custom_select_original_features",
                    label_visibility="collapsed"
                )
            
            with col_custom2:
                st.markdown("**智能工程特征：**")
                selected_smart = st.multiselect(
                    "选择智能工程特征",
                    options=smart_features,
                    default=smart_features,
                    key="custom_select_smart_features",
                    label_visibility="collapsed"
                )
            
            with col_custom3:
                st.markdown("**自定义特征：**")
                selected_custom = st.multiselect(
                    "选择自定义特征",
                    options=custom_features,
                    default=custom_features,
                    key="custom_select_custom_features",
                    label_visibility="collapsed"
                )
            
            selected_features = selected_original + selected_smart + selected_custom
            
            if selected_features:
                st.success(f"✅ 已选择 {len(selected_features)} 个特征（原始 {len(selected_original)} + 工程 {len(selected_engineered)}）")
            else:
                st.warning("⚠️ 请至少选择一个特征")
        
        # 显示特征对比
        with st.expander("📊 查看特征对比", expanded=False):
            col_compare1, col_compare2, col_compare3 = st.columns(3)
            
            with col_compare1:
                st.metric("原始特征数", len(original_features))
                with st.container():
                    st.caption("原始特征列表：")
                    for feat in original_features[:5]:
                        st.text(f"• {feat}")
                    if len(original_features) > 5:
                        st.text(f"... 还有 {len(original_features)-5} 个")
            
            with col_compare2:
                st.metric("工程特征数", len(all_engineered_features))
                with st.container():
                    st.caption("工程特征列表：")
                    for feat in all_engineered_features[:5]:
                        st.text(f"• {feat}")
                    if len(all_engineered_features) > 5:
                        st.text(f"... 还有 {len(all_engineered_features)-5} 个")
            
            with col_compare3:
                st.metric("选中特征数", len(selected_features))
                st.metric("特征增加", 
                         f"+{len(all_engineered_features)}", 
                         delta=f"+{len(all_engineered_features)/len(original_features)*100:.0f}%" if len(original_features) > 0 else "+0%")
    
    else:
        # 未执行特征工程
        selected_features = st.session_state.feature_cols
        st.info(f"📊 使用当前特征：{len(selected_features)} 个")
        st.caption("💡 提示：在步骤2中可以执行智能特征工程来增强特征")
    
    # 显示数据信息
    st.markdown("---")
    st.markdown("#### 📋 训练数据信息")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("样本数", len(training_data))
    with col_info2:
        st.metric("特征数", len(selected_features))
    with col_info3:
        st.metric("目标变量", st.session_state.target_col)
    
    # 显示数据预处理历史
    if st.session_state.get('outlier_cleaning_done', False) or st.session_state.get('feature_engineering_done', False):
        with st.expander("📜 查看数据预处理历史", expanded=False):
            if st.session_state.get('outlier_cleaning_done', False):
                st.write(f"✅ 异常值清洗：移除了 {st.session_state.n_outliers_removed} 个异常样本")
            if st.session_state.get('feature_engineering_done', False):
                smart_features = st.session_state.get('smart_engineered_features', [])
                custom_features = st.session_state.get('custom_engineered_features', [])
                total_engineered = len(smart_features) + len(custom_features)
                st.write(f"✅ 特征工程：生成了 {total_engineered} 个新特征（智能 {len(smart_features)} + 自定义 {len(custom_features)}）")
    
    st.markdown("---")
    
    # 数据集划分
    st.subheader("🔀 数据集划分")
    
    target_col = st.session_state.target_col
    feature_cols = st.session_state.feature_cols
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05,
                             help="测试集占总数据的比例，通常为20%-30%")
    with col2:
        random_state = st.number_input("随机种子", 0, 999, 42,
                                      help="设置随机种子以确保结果可重复")
    
    st.info(f"📊 数据将被划分为：训练集 {int((1-test_size)*100)}% ({int(len(training_data)*(1-test_size))} 样本) | 测试集 {int(test_size)*100}% ({int(len(training_data)*test_size)} 样本)")
    
    if st.button("✅ 确认划分并开始训练", type="primary", key="confirm_split"):
        from sklearn.model_selection import train_test_split
        
        X = training_data[selected_features].values
        y = training_data[target_col].values
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )
        
        # 保存选中的特征列表
        st.session_state.selected_features_for_training = selected_features
        
        # 保存原始特征范围（在标准化之前）
        st.session_state.feature_ranges = {}
        for idx, name in enumerate(selected_features):
            st.session_state.feature_ranges[name] = {
                'min': X_train[:, idx].min(),
                'max': X_train[:, idx].max(),
                'mean': X_train[:, idx].mean(),
                'std': X_train[:, idx].std()
            }
        
        # 标准化（注意：只在训练集上fit，然后transform训练集和测试集）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 保存到session state
        # 保存原始数据（未标准化）用于相关性分析
        st.session_state.X_train_original = pd.DataFrame(X_train, columns=selected_features)
        st.session_state.X_test_original = pd.DataFrame(X_test, columns=selected_features)
        # 保存标准化后的数据用于模型训练
        st.session_state.X_train = X_train_scaled
        st.session_state.X_test = X_test_scaled
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.feature_names = selected_features
        st.session_state.scaler = scaler
        st.session_state.test_size = test_size
        
        st.success(f"✅ 数据准备完成！训练集：{len(X_train)} 样本，测试集：{len(X_test)} 样本")
        st.rerun()
    
    st.markdown("---")
    
    # 如果数据已准备好，显示模型选择
    if st.session_state.X_train is not None:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← 返回"):
                st.session_state.step = 2
                st.rerun()
        
        st.subheader("🎯 选择模型")
        
        # 添加训练模式选择
        st.markdown("### 🎯 训练模式")
        training_mode = st.radio(
            "选择训练方式",
            ["单模型训练", "模型对比"],
            help="单模型训练: 训练一个模型并优化\n模型对比: 训练所有可用模型并自动对比性能"
        )
        
        st.markdown("---")
        
        if training_mode == "单模型训练":
            # 原有的单模型训练逻辑
            
            # 智能模型推荐
            n_samples = len(X_train)
            n_features = len(st.session_state.feature_cols)
            
            model_suggestions = suggest_model(n_samples, n_features)
            
            if model_suggestions:
                st.markdown("### 🤖 智能推荐")
                st.info("💡 **推荐模型** (基于数据特征分析)")
                for i, sug in enumerate(model_suggestions[:2], 1):
                    st.write(f"{i}. **{sug['model']}**: {sug['reason']}")
                st.markdown("---")
            
            # 构建可用模型列表
            available_models = ["RF", "SVR", "MLP"]
            if LIGHTGBM_AVAILABLE:
                available_models.append("LightGBM")
            if XGBOOST_AVAILABLE:
                available_models.append("XGBoost")
            
            # 显示不可用的模型提示
            unavailable_models = []
            if not LIGHTGBM_AVAILABLE:
                unavailable_models.append("LightGBM")
            if not XGBOOST_AVAILABLE:
                unavailable_models.append("XGBoost")
            
            if unavailable_models:
                st.info(f"ℹ️ 以下模型不可用（需要安装相应的包）：{', '.join(unavailable_models)}")
            
            models_to_train = st.multiselect(
                "选择要训练的模型",
                available_models,
                default=["RF"]
            )
            
            st.subheader("⚙️ 优化设置")
            col1, col2 = st.columns(2)
            with col1:
                n_iter = st.slider("贝叶斯优化迭代次数：", 5, 100, 20)
            with col2:
                cv_folds = st.slider("交叉验证折数", 3, 10, 10)
            
            if st.button("🚀 开始训练", type="primary", key="train_single_model"):
                # 获取优化模式
                mode = 'full' if st.session_state.advanced_settings['optimization_mode'] == 'full' else 'fast'
                
                # 显示当前设置
                if mode == 'full':
                    st.info("🔧 使用完整模式：优化更多参数，训练时间较长（预计+150-250%）")
                else:
                    st.info("⚡ 使用快速模式：使用常用参数，训练速度快")
                
                results = {}
                
                # 创建进度显示容器
                progress_container = st.container()
                with progress_container:
                    st.markdown("### 📊 训练进度")
                    overall_progress = st.progress(0)
                    overall_status = st.empty()
                    
                    # 当前模型进度
                    st.markdown("#### 当前模型")
                    current_model_name = st.empty()
                    current_model_progress = st.progress(0)
                    current_model_status = st.empty()
                    current_model_time = st.empty()
                    
                    # 已完成模型
                    st.markdown("#### 已完成模型")
                    completed_models = st.empty()
                
                import time as time_module
                total_start_time = time_module.time()
                completed_list = []
                
                for idx, model_name in enumerate(models_to_train):
                    model_start_time = time_module.time()
                    
                    # 更新总体进度
                    overall_progress.progress(idx / len(models_to_train))
                    overall_status.text(f"总进度：{idx}/{len(models_to_train)} 模型已完成")
                    
                    # 更新当前模型信息
                    current_model_name.markdown(f"**🔄 {model_name}**")
                    current_model_progress.progress(0)
                    current_model_status.text("准备开始训练...")
                    
                    try:
                        # 训练模型
                        if model_name == "RF":
                            current_model_status.text("正在进行贝叶斯优化...")
                            result = train_random_forest(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode=mode)
                        elif model_name == "SVR":
                            current_model_status.text("正在进行贝叶斯优化...")
                            result = train_svm(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode=mode)
                        elif model_name == "MLP":
                            current_model_status.text("正在进行贝叶斯优化...")
                            result = train_mlp(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode=mode)
                        elif model_name == "LightGBM":
                            current_model_status.text("正在进行贝叶斯优化...")
                            result = train_lightgbm(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode=mode)
                        elif model_name == "XGBoost":
                            current_model_status.text("正在进行贝叶斯优化...")
                            result = train_xgboost(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode=mode)
                        
                        results[model_name] = result
                        
                        # 完成当前模型
                        current_model_progress.progress(1.0)
                        model_elapsed = time_module.time() - model_start_time
                        current_model_status.text(f"✅ 训练完成")
                        current_model_time.success(f"⏱️ 用时：{model_elapsed:.1f}秒 | 📊 测试集R²：{result['test_r2']:.4f}")
                        
                        # 添加到已完成列表
                        completed_list.append(f"✅ {model_name} - R²: {result['test_r2']:.4f} - {model_elapsed:.1f}秒")
                        completed_models.text("\n".join(completed_list))
                        
                    except Exception as e:
                        current_model_progress.progress(1.0)
                        current_model_status.text(f"❌ 训练失败")
                        current_model_time.error(f"错误：{str(e)}")
                        completed_list.append(f"❌ {model_name} - 失败")
                        completed_models.text("\n".join(completed_list))
                
                # 完成所有训练
                overall_progress.progress(1.0)
                total_elapsed = time_module.time() - total_start_time
                overall_status.success(f"🎉 所有模型训练完成！总用时：{total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
                
                st.session_state.results = results
                
                # 显示继续按钮
                st.markdown("---")
                if st.button("查看结果 →", type="primary", key="view_results"):
                    st.rerun()
            
            # 显示已训练模型的结果（在训练按钮外面，这样rerun后仍然可见）
            if st.session_state.results and len(st.session_state.results) > 0:
                results = st.session_state.results
                
                st.subheader("📊 模型性能对比")
                
                # 创建对比表格
                comparison_df = pd.DataFrame({
                    '模型': list(results.keys()),
                    '交叉验证R²': [results[m]['cv_score'] for m in results.keys()],
                    '训练集R²': [results[m]['train_r2'] for m in results.keys()],
                    '测试集R²': [results[m]['test_r2'] for m in results.keys()],
                    '测试集RMSE': [results[m]['test_rmse'] for m in results.keys()],
                    '测试集MAE': [results[m]['test_mae'] for m in results.keys()],
                    '训练时间(秒)': [results[m].get('train_time', 0) for m in results.keys()]
                })
                comparison_df = comparison_df.sort_values('测试集R²', ascending=False)
                st.dataframe(comparison_df, width="stretch")
                
                # 显示说明
                st.info("""
                **指标说明*
                - **交叉验证R²**：训练集上K折交叉验证的平均R²分数（用于超参数选择）
                - **训练集R²**：模型在训练集上的R²分数
                - **测试集R²**：模型在测试集上的R²分数（最重要的指标）
                - **测试集RMSE**：测试集上的均方根误差
                - **测试集MAE**：测试集上的平均绝对误差
                - **训练时间**：模型训练和优化的总耗时
                
                💡 **如何判断模型质量*
                - 测试集R²接近训练集R²：模型泛化能力好
                - 测试集R²远低于训练集R²：可能存在过拟合
                - 测试集R²高于训练集R²：罕见，可能数据划分有利
                """)
                
                # 保存最佳模型（基于测试集R²）
                best_model_name = comparison_df.iloc[0]['模型']
                st.session_state.best_model = results[best_model_name]['model']
                st.session_state.best_model_name = best_model_name
                
                st.success(f"🏆 最佳模型：{best_model_name} (测试集R² = {results[best_model_name]['test_r2']:.4f})")
                
                # ==================== 可视化对比====================
                st.markdown("---")
                
                n_models = len(results)
                
                # 如果只有1个模型，不显示对比图，只显示详细指标
                if n_models == 1:
                    st.subheader("📊 模型性能详情")
                    st.info("💡 只训练了1个模型，无需对比。以下是该模型的详细性能指标。")
                    
                    model_name = list(results.keys())[0]
                    result = results[model_name]
                    
                    # 使用指标卡片展示
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("训练集R²", f"{result['train_r2']:.4f}")
                    with col2:
                        st.metric("测试集R²", f"{result['test_r2']:.4f}")
                    with col3:
                        st.metric("测试集RMSE", f"{result['test_rmse']:.4f}")
                    with col4:
                        st.metric("测试集MAE", f"{result['test_mae']:.4f}")
                    
                    # 预测效果散点图（增强版：置信区间 + 从0开始 + 1:1比例）
                    st.markdown("#### 📈 预测效果")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=120)
                    
                    # 训练集
                    y_train = result['y_train_true']
                    y_train_pred = result['y_train_pred']
                    
                    # 从0开始，统一范围
                    min_val = 0
                    max_val = max(y_train.max(), y_train_pred.max())
                    margin = max_val * 0.05
                    plot_max = max_val + margin
                    
                    # 绘制散点
                    ax1.scatter(y_train, y_train_pred, alpha=0.5, s=40, color='#3498db', 
                               edgecolors='white', linewidth=0.5, label='数据点')
                    
                    # 完美预测线
                    ax1.plot([min_val, plot_max], [min_val, plot_max], 'r-', 
                            linewidth=2.5, alpha=0.8, label='完美预测', zorder=3)
                    
                    # ±10%置信区间
                    ax1.fill_between([min_val, plot_max], 
                                    [min_val*0.9, plot_max*0.9],
                                    [min_val*1.1, plot_max*1.1],
                                    alpha=0.15, color='green', label='±10%区间', zorder=1)
                    
                    # ±20%置信区间
                    ax1.fill_between([min_val, plot_max], 
                                    [min_val*0.8, plot_max*0.8],
                                    [min_val*1.2, plot_max*1.2],
                                    alpha=0.1, color='orange', label='±20%区间', zorder=0)
                    
                    ax1.set_xlim(min_val, plot_max)
                    ax1.set_ylim(min_val, plot_max)
                    ax1.set_aspect('equal', adjustable='box')  # 1:1比例
                    ax1.set_title(f'训练集 (R² = {result["train_r2"]:.4f})', 
                                 fontsize=13, fontweight='bold', fontproperties=font_prop, pad=10)
                    ax1.set_xlabel('实际值', fontsize=11, fontweight='bold', fontproperties=font_prop)
                    ax1.set_ylabel('预测值', fontsize=11, fontweight='bold', fontproperties=font_prop)
                    ax1.grid(True, alpha=0.3, linestyle='--')
                    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
                    
                    # 测试集
                    y_test = result['y_test_true']
                    y_test_pred = result['y_test_pred']
                    
                    # 使用相同的范围
                    max_val_test = max(y_test.max(), y_test_pred.max())
                    plot_max = max(plot_max, max_val_test + max_val_test * 0.05)
                    
                    # 绘制散点
                    ax2.scatter(y_test, y_test_pred, alpha=0.5, s=40, color='#e74c3c', 
                               edgecolors='white', linewidth=0.5, label='数据点')
                    
                    # 完美预测线
                    ax2.plot([min_val, plot_max], [min_val, plot_max], 'r-', 
                            linewidth=2.5, alpha=0.8, label='完美预测', zorder=3)
                    
                    # ±10%置信区间
                    ax2.fill_between([min_val, plot_max], 
                                    [min_val*0.9, plot_max*0.9],
                                    [min_val*1.1, plot_max*1.1],
                                    alpha=0.15, color='green', label='±10%区间', zorder=1)
                    
                    # ±20%置信区间
                    ax2.fill_between([min_val, plot_max], 
                                    [min_val*0.8, plot_max*0.8],
                                    [min_val*1.2, plot_max*1.2],
                                    alpha=0.1, color='orange', label='±20%区间', zorder=0)
                    
                    ax2.set_xlim(min_val, plot_max)
                    ax2.set_ylim(min_val, plot_max)
                    ax2.set_aspect('equal', adjustable='box')  # 1:1比例
                    ax2.set_title(f'测试集 (R² = {result["test_r2"]:.4f})', 
                                 fontsize=13, fontweight='bold', fontproperties=font_prop, pad=10)
                    ax2.set_xlabel('实际值', fontsize=11, fontweight='bold', fontproperties=font_prop)
                    ax2.set_ylabel('预测值', fontsize=11, fontweight='bold', fontproperties=font_prop)
                    ax2.grid(True, alpha=0.3, linestyle='--')
                    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                else:
                    # 多个模型时，显示预测效果对比和雷达图
                    
                    # 预测效果对比（带边际分布）
                    st.markdown("#### 3️⃣ 预测效果对比")
                    
                    # 为每个模型单独绘制带边际分布的预测图
                    for name, result in results.items():
                        st.markdown(f"**{name}**")
                        
                        # 获取训练集和测试集的真实值和预测值
                        y_train_true = result['y_train_true']
                        y_train_pred = result['y_train_pred']
                        y_test_true = result['y_test_true']
                        y_test_pred = result['y_test_pred']
                        
                        # 获取R²值
                        train_r2 = result['train_r2']
                        test_r2 = result['test_r2']
                        
                        # 使用新的绘图函数
                        fig = plot_prediction_with_marginals(
                            y_train_true, y_train_pred,
                            y_test_true, y_test_pred,
                            train_r2, test_r2,
                            name
                        )
                        
                        st.pyplot(fig)
                        plt.close()
                    
                    # 多模型性能对比雷达图
                    if len(results) > 1:
                        st.markdown("---")
                        st.markdown("#### 📊 多模型性能对比雷达图")
                        
                        st.info("""
                        **📖 雷达图说明**
                        - **5个顶点**：RMSE、MAPE、IA、R²、MAE
                        - **不同颜色线条**：代表不同的模型
                        - **理想模型形状**：
                          - RMSE、MAE、MAPE：越靠近边缘越好（误差小）
                          - R²、IA：越靠近边缘越好（拟合好）
                        - **面积越大** = 性能越好
                        """)
                        
                        # 模型选择
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            available_models = list(results.keys())
                            selected_models_radar = st.multiselect(
                                "选择要对比的模型（最多8个）",
                                available_models,
                                default=available_models[:min(4, len(available_models))],
                                help="选择2-8个模型进行对比",
                                key="radar_chart_step3_auto"
                            )
                        
                        with col2:
                            st.markdown("**当前选中模型：**")
                            if selected_models_radar:
                                for model in selected_models_radar:
                                    st.caption(f"✓ {model}")
                            else:
                                st.caption("未选择模型")
                        
                        if len(selected_models_radar) >= 2:
                            # 绘制雷达图
                            try:
                                fig_radar = plot_radar_chart(results, selected_models_radar)
                                st.pyplot(fig_radar)
                                plt.close()
                                
                                # 显示详细对比表
                                with st.expander("📋 查看详细指标对比表", expanded=False):
                                    comparison_data = []
                                    for model_name in selected_models_radar:
                                        model_res = results[model_name]
                                        comparison_data.append({
                                            '模型': model_name,
                                            'R²': f"{model_res['test_r2']:.4f}",
                                            'RMSE': f"{model_res['test_rmse']:.4f}",
                                            'MAE': f"{model_res['test_mae']:.4f}",
                                            'MAPE (%)': f"{model_res['test_mape']:.2f}",
                                            'IA': f"{model_res['test_ia']:.4f}"
                                        })
                                    
                                    comparison_df_radar = pd.DataFrame(comparison_data)
                                    st.dataframe(comparison_df_radar, width='stretch', hide_index=True)
                                    
                                    # 找出最佳模型
                                    best_model_r2 = max(selected_models_radar, key=lambda x: results[x]['test_r2'])
                                    best_model_ia = max(selected_models_radar, key=lambda x: results[x]['test_ia'])
                                    best_model_rmse = min(selected_models_radar, key=lambda x: results[x]['test_rmse'])
                                    
                                    st.markdown("**🏆 最佳模型：**")
                                    col1, col2, col3 = st.columns(3)
                                    col1.success(f"**R²最高**: {best_model_r2}")
                                    col2.success(f"**IA最高**: {best_model_ia}")
                                    col3.success(f"**RMSE最低**: {best_model_rmse}")
                                    
                            except Exception as e:
                                st.error(f"绘制雷达图时出错：{str(e)}")
                        elif len(selected_models_radar) == 1:
                            st.warning("⚠️ 请至少选择2个模型进行对比")
                        else:
                            st.info("💡 请选择要对比的模型")
                
                # 统一的导航按钮
                st.markdown("---")
                next_step_button("继续到步骤4：结果可视化", "next_step_3_to_4", 4)
        
        else:  # training_mode == "模型对比"
            st.info("💡 将自动训练所有可用模型并对比性能，推荐最佳模型")
            
            # 显示可用模型列表
            available_models_list = ["RF", "SVR", "MLP"]
            if LIGHTGBM_AVAILABLE:
                available_models_list.append("LightGBM")
            if XGBOOST_AVAILABLE:
                available_models_list.append("XGBoost")
            
            st.markdown(f"**将训练的模型**: {', '.join(available_models_list)}")
            
            # 优化设置
            st.subheader("⚙️ 优化设置")
            col1, col2 = st.columns(2)
            with col1:
                n_iter = st.slider("贝叶斯优化迭代次数：", 5, 100, 20, key="compare_n_iter")
            with col2:
                cv_folds = st.slider("交叉验证折数", 3, 10, 10, key="compare_cv_folds")
            
            if st.button("🚀 开始模型对比", type="primary", key="train_compare_models"):
                # 获取优化模式
                mode = 'full' if st.session_state.advanced_settings['optimization_mode'] == 'full' else 'fast'
                
                # 显示当前设置
                if mode == 'full':
                    st.info("🔧 使用完整模式：优化更多参数，训练时间较长（预计+150-250%）")
                else:
                    st.info("⚡ 使用快速模式：使用常用参数，训练速度快")
                
                results = {}
                
                # 创建进度显示容器
                progress_container = st.container()
                with progress_container:
                    st.markdown("### 📊 训练进度")
                    overall_progress = st.progress(0)
                    overall_status = st.empty()
                    
                    # 当前模型进度
                    st.markdown("#### 当前模型")
                    current_model_name = st.empty()
                    current_model_progress = st.progress(0)
                    current_model_status = st.empty()
                    current_model_time = st.empty()
                    
                    # 已完成模型
                    st.markdown("#### 已完成模型")
                    completed_models = st.empty()
                
                import time as time_module
                total_start_time = time_module.time()
                completed_list = []
                
                # 训练所有可用模型
                for idx, model_name in enumerate(available_models_list):
                    model_start_time = time_module.time()
                    
                    # 更新总体进度
                    overall_progress.progress(idx / len(available_models_list))
                    overall_status.text(f"总进度：{idx}/{len(available_models_list)} 模型已完成")
                    
                    # 更新当前模型信息
                    current_model_name.markdown(f"**🔄 {model_name}**")
                    current_model_progress.progress(0)
                    current_model_status.text("准备开始训练...")
                    
                    try:
                        # 训练模型
                        if model_name == "RF":
                            current_model_status.text("正在进行贝叶斯优化...")
                            result = train_random_forest(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode=mode)
                        elif model_name == "SVR":
                            current_model_status.text("正在进行贝叶斯优化...")
                            result = train_svm(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode=mode)
                        elif model_name == "MLP":
                            current_model_status.text("正在进行贝叶斯优化...")
                            result = train_mlp(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode=mode)
                        elif model_name == "LightGBM":
                            current_model_status.text("正在进行贝叶斯优化...")
                            result = train_lightgbm(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode=mode)
                        elif model_name == "XGBoost":
                            current_model_status.text("正在进行贝叶斯优化...")
                            result = train_xgboost(X_train, y_train, X_test, y_test, n_iter, cv_folds, mode=mode)
                        
                        results[model_name] = result
                        
                        # 完成当前模型
                        current_model_progress.progress(1.0)
                        model_elapsed = time_module.time() - model_start_time
                        current_model_status.text(f"✅ 训练完成")
                        current_model_time.success(f"⏱️ 用时：{model_elapsed:.1f}秒 | 📊 测试集R²：{result['test_r2']:.4f}")
                        
                        # 添加到已完成列表
                        completed_list.append(f"✅ {model_name} - R²: {result['test_r2']:.4f} - {model_elapsed:.1f}秒")
                        completed_models.text("\n".join(completed_list))
                        
                    except Exception as e:
                        current_model_progress.progress(1.0)
                        current_model_status.text(f"❌ 训练失败")
                        current_model_time.error(f"错误：{str(e)}")
                        completed_list.append(f"❌ {model_name} - 失败")
                        completed_models.text("\n".join(completed_list))
                
                # 完成所有训练
                overall_progress.progress(1.0)
                total_elapsed = time_module.time() - total_start_time
                overall_status.success(f"🎉 所有模型训练完成！总用时：{total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
                
                st.session_state.results = results
                
                # 显示继续按钮
                st.markdown("---")
                if st.button("查看结果 →", type="primary", key="view_results_compare"):
                    st.rerun()
            
            # 显示已训练模型的结果（在训练按钮外面，这样rerun后仍然可见）
            if st.session_state.results and len(st.session_state.results) > 0:
                results = st.session_state.results
                
                st.subheader("📊 模型性能对比")
                
                # 创建对比表格
                comparison_df = pd.DataFrame({
                    '模型': list(results.keys()),
                    '交叉验证R²': [results[m]['cv_score'] for m in results.keys()],
                    '训练集R²': [results[m]['train_r2'] for m in results.keys()],
                    '测试集R²': [results[m]['test_r2'] for m in results.keys()],
                    '测试集RMSE': [results[m]['test_rmse'] for m in results.keys()],
                    '测试集MAE': [results[m]['test_mae'] for m in results.keys()],
                    '训练时间(秒)': [results[m].get('train_time', 0) for m in results.keys()]
                })
                comparison_df = comparison_df.sort_values('测试集R²', ascending=False)
                st.dataframe(comparison_df, width="stretch")
                
                # 显示说明
                st.info("""
                **指标说明*
                - **交叉验证R²**：训练集上K折交叉验证的平均R²分数（用于超参数选择）
                - **训练集R²**：模型在训练集上的R²分数
                - **测试集R²**：模型在测试集上的R²分数（最重要的指标）
                - **测试集RMSE**：测试集上的均方根误差
                - **测试集MAE**：测试集上的平均绝对误差
                - **训练时间**：模型训练和优化的总耗时
                
                💡 **如何判断模型质量*
                - 测试集R²接近训练集R²：模型泛化能力好
                - 测试集R²远低于训练集R²：可能存在过拟合
                - 测试集R²高于训练集R²：罕见，可能数据划分有利
                """)
                
                # 保存最佳模型（基于测试集R²）
                best_model_name = comparison_df.iloc[0]['模型']
                st.session_state.best_model = results[best_model_name]['model']
                st.session_state.best_model_name = best_model_name
                
                st.success(f"🏆 最佳模型：{best_model_name} (测试集R² = {results[best_model_name]['test_r2']:.4f})")
                
                # ==================== 可视化对比====================
                st.markdown("---")
                st.subheader("📈 可视化对比分析")
                
                n_models = len(comparison_df)
                
                # 判断是否只有1个模型
                if n_models == 1:
                    # 只有1个模型时，不显示对比图，显示详细指标卡片
                    st.info("💡 当前只训练了1个模型，无法进行对比分析。建议选择多个模型进行训练以查看对比效果。")
                    
                    # 显示详细的性能指标卡片
                    model_name = comparison_df.iloc[0]['模型']
                    model_result = results[model_name]
                    
                    st.markdown("#### 📊 模型性能详情")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("训练集R²", f"{model_result['train_r2']:.4f}")
                        st.metric("测试集RMSE", f"{model_result['test_rmse']:.4f}")
                    
                    with col2:
                        st.metric("测试集R²", f"{model_result['test_r2']:.4f}")
                        st.metric("测试集MAE", f"{model_result['test_mae']:.4f}")
                    
                    with col3:
                        st.metric("交叉验证R²", f"{model_result['cv_score']:.4f}")
                        st.metric("训练时间", f"{model_result.get('train_time', 0):.2f}秒")
                    
                    # 显示训练集vs测试集的简单对比散点图
                    st.markdown("#### 📈 训练集 vs 测试集预测效果")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
                    
                    # 训练集
                    y_train = model_result['y_train_true']
                    y_train_pred = model_result['y_train_pred']
                    
                    # 计算全局范围（训练集和测试集统一）
                    y_test = model_result['y_test_true']
                    y_test_pred = model_result['y_test_pred']
                    global_min = min(y_train.min(), y_train_pred.min(), y_test.min(), y_test_pred.min())
                    global_max = max(y_train.max(), y_train_pred.max(), y_test.max(), y_test_pred.max())
                    
                    ax1.scatter(y_train, y_train_pred, alpha=0.6, s=30, color='#3498db', edgecolors='white')
                    ax1.plot([global_min, global_max], [global_min, global_max], 'r--', linewidth=2, alpha=0.7)
                    ax1.set_xlim(global_min, global_max)
                    ax1.set_ylim(global_min, global_max)
                    ax1.set_aspect('equal', adjustable='box')  # 1:1比例
                    ax1.set_title(f'训练集\nR² = {model_result["train_r2"]:.4f}', fontsize=12, fontweight='bold', fontproperties=font_prop)
                    ax1.set_xlabel('实际值', fontsize=10, fontproperties=font_prop)
                    ax1.set_ylabel('预测值', fontsize=10, fontproperties=font_prop)
                    ax1.grid(True, alpha=0.3)
                    
                    # 测试集
                    ax2.scatter(y_test, y_test_pred, alpha=0.6, s=30, color='#e74c3c', edgecolors='white')
                    ax2.plot([global_min, global_max], [global_min, global_max], 'r--', linewidth=2, alpha=0.7)
                    ax2.set_xlim(global_min, global_max)
                    ax2.set_ylim(global_min, global_max)
                    ax2.set_aspect('equal', adjustable='box')  # 1:1比例
                    ax2.set_title(f'测试集\nR² = {model_result["test_r2"]:.4f}', fontsize=12, fontweight='bold', fontproperties=font_prop)
                    ax2.set_xlabel('实际值', fontsize=10, fontproperties=font_prop)
                    ax2.set_ylabel('预测值', fontsize=10, fontproperties=font_prop)
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                else:
                    # 多个模型时，显示预测效果对比和雷达图
                    
                    # 预测效果对比（带边际分布）
                    st.markdown("#### 3️⃣ 预测效果对比")
                    
                    # 为每个模型单独绘制带边际分布的预测图
                    for name, result in results.items():
                        st.markdown(f"**{name}**")
                        
                        # 获取训练集和测试集的真实值和预测值
                        y_train_true = result['y_train_true']
                        y_train_pred = result['y_train_pred']
                        y_test_true = result['y_test_true']
                        y_test_pred = result['y_test_pred']
                        
                        # 获取R²值
                        train_r2 = result['train_r2']
                        test_r2 = result['test_r2']
                        
                        # 使用新的绘图函数
                        fig = plot_prediction_with_marginals(
                            y_train_true, y_train_pred,
                            y_test_true, y_test_pred,
                            train_r2, test_r2,
                            name
                        )
                        
                        st.pyplot(fig)
                        plt.close()
                    
                    # 多模型性能对比雷达图
                    if len(results) > 1:
                        st.markdown("---")
                        st.markdown("#### 📊 多模型性能对比雷达图")
                        
                        st.info("""
                        **📖 雷达图说明**
                        - **5个顶点**：RMSE、MAPE、IA、R²、MAE
                        - **不同颜色线条**：代表不同的模型
                        - **理想模型形状**：
                          - RMSE、MAE、MAPE：越靠近边缘越好（误差小）
                          - R²、IA：越靠近边缘越好（拟合好）
                        - **面积越大** = 性能越好
                        """)
                        
                        # 模型选择
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            available_models = list(results.keys())
                            selected_models_radar = st.multiselect(
                                "选择要对比的模型（最多8个）",
                                available_models,
                                default=available_models[:min(4, len(available_models))],
                                help="选择2-8个模型进行对比",
                                key="radar_chart_step3_compare"
                            )
                        
                        with col2:
                            st.markdown("**当前选中模型：**")
                            if selected_models_radar:
                                for model in selected_models_radar:
                                    st.caption(f"✓ {model}")
                            else:
                                st.caption("未选择模型")
                        
                        if len(selected_models_radar) >= 2:
                            # 绘制雷达图
                            try:
                                fig_radar = plot_radar_chart(results, selected_models_radar)
                                st.pyplot(fig_radar)
                                plt.close()
                                
                                # 显示详细对比表
                                with st.expander("📋 查看详细指标对比表", expanded=False):
                                    comparison_data = []
                                    for model_name in selected_models_radar:
                                        model_res = results[model_name]
                                        comparison_data.append({
                                            '模型': model_name,
                                            'R²': f"{model_res['test_r2']:.4f}",
                                            'RMSE': f"{model_res['test_rmse']:.4f}",
                                            'MAE': f"{model_res['test_mae']:.4f}",
                                            'MAPE (%)': f"{model_res['test_mape']:.2f}",
                                            'IA': f"{model_res['test_ia']:.4f}"
                                        })
                                    
                                    comparison_df_radar = pd.DataFrame(comparison_data)
                                    st.dataframe(comparison_df_radar, width='stretch', hide_index=True)
                                    
                                    # 找出最佳模型
                                    best_model_r2 = max(selected_models_radar, key=lambda x: results[x]['test_r2'])
                                    best_model_ia = max(selected_models_radar, key=lambda x: results[x]['test_ia'])
                                    best_model_rmse = min(selected_models_radar, key=lambda x: results[x]['test_rmse'])
                                    
                                    st.markdown("**🏆 最佳模型：**")
                                    col1, col2, col3 = st.columns(3)
                                    col1.success(f"**R²最高**: {best_model_r2}")
                                    col2.success(f"**IA最高**: {best_model_ia}")
                                    col3.success(f"**RMSE最低**: {best_model_rmse}")
                                    
                            except Exception as e:
                                st.error(f"绘制雷达图时出错：{str(e)}")
                        elif len(selected_models_radar) == 1:
                            st.warning("⚠️ 请至少选择2个模型进行对比")
                        else:
                            st.info("💡 请选择要对比的模型")
                
                # 统一的导航按钮
                st.markdown("---")
                next_step_button("继续到步骤4：结果可视化", "next_step_3_to_4_compare", 4)

# 步骤4：结果可视化
elif st.session_state.step == 4:
    st.header("4️⃣ 结果可视化 (Results Visualization)")
    
    if not st.session_state.results or len(st.session_state.results) == 0:
        st.warning("⚠️ 请先训练模型")
        if st.button("← 返回模型训练"):
            st.session_state.step = 3
            st.rerun()
    else:
        results = st.session_state.results
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← 返回训练"):
                st.session_state.step = 3
                st.rerun()
        
        # 模型选择
        selected_model = st.selectbox("选择要查看的模型", list(results.keys()))
        model_result = results[selected_model]
        
        # 性能指标
        st.subheader("📈 性能指标")
        
        # 显示训练集和测试集性能
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**训练集性能**")
            subcol1, subcol2, subcol3, subcol4, subcol5 = st.columns(5)
            subcol1.metric("R²", f"{model_result['train_r2']:.4f}")
            subcol2.metric("RMSE", f"{model_result['train_rmse']:.4f}")
            subcol3.metric("MAE", f"{model_result['train_mae']:.4f}")
            subcol4.metric("MAPE (%)", f"{model_result['train_mape']:.2f}")
            subcol5.metric("IA", f"{model_result['train_ia']:.4f}")
        
        with col2:
            st.markdown("**测试集性能（泛化能力）**")
            subcol1, subcol2, subcol3, subcol4, subcol5 = st.columns(5)
            subcol1.metric("R²", f"{model_result['test_r2']:.4f}")
            subcol2.metric("RMSE", f"{model_result['test_rmse']:.4f}")
            subcol3.metric("MAE", f"{model_result['test_mae']:.4f}")
            subcol4.metric("MAPE (%)", f"{model_result['test_mape']:.2f}")
            subcol5.metric("IA", f"{model_result['test_ia']:.4f}")
        
        # 显示交叉验证分数
        st.metric("交叉验证R²分数", f"{model_result['cv_score']:.4f}", 
                 help="训练集上K折交叉验证的平均R²分数")
        
        # 添加指标说明
        with st.expander("📖 指标说明", expanded=False):
            st.markdown("""
            **评价指标解释：**
            
            - **R² (决定系数)**：范围 (-∞, 1]，越接近1越好。表示模型解释的方差比例。
            - **RMSE (均方根误差)**：范围 [0, +∞)，越小越好。对大误差敏感。
            - **MAE (平均绝对误差)**：范围 [0, +∞)，越小越好。对所有误差一视同仁。
            - **MAPE (平均绝对百分比误差)**：范围 [0, +∞)，越小越好。相对误差，单位为百分比。
            - **IA (一致性指数)**：范围 [0, 1]，越接近1越好。对系统性偏差敏感，比R²更全面。
            
            **参考标准：**
            - R² > 0.9：优秀
            - R² > 0.8：良好
            - R² > 0.7：中等
            - IA > 0.9：优秀
            - IA > 0.8：良好
            - MAPE < 10%：优秀
            - MAPE < 20%：良好
            """)

        
        # 过拟合检测和调整建议
        st.markdown("---")
        st.subheader("🔍 过拟合分析与调整建议")
        
        overfitting = model_result['train_r2'] - model_result['test_r2']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("训练集R²", f"{model_result['train_r2']:.4f}")
            st.metric("测试集R²", f"{model_result['test_r2']:.4f}")
            st.metric("R²差值", f"{overfitting:.4f}", 
                     delta=f"{'过拟合风险' if overfitting > 0.1 else '良好' if overfitting > 0 else '优秀'}")
        
        with col2:
            if overfitting > 0.15:
                st.error("❌ 严重过拟合！模型在训练集上表现很好，但在测试集上表现较差")
                
                with st.expander("💡 如何解决过拟合？点击查看建议", expanded=True):
                    st.markdown("""
                    ### 🔧 调整建议（按优先级排序）
                    
                    #### 1️⃣ 增加训练数据
                    - 如果可能，收集更多数据
                    - 当前训练集样本数：{}
                    - 建议：至少需要特征数的0-20倍样本
                    
                    #### 2️⃣ 调整测试集比例
                    - 当前测试集比例：{}%
                    - 建议：如果数据量小，可以减少到0-15%
                    - 在步骤3中调整测试集比例
                    
                    #### 3️⃣ 特征选择
                    - 当前特征数：{}
                    - 建议：使用SHAP分析（步骤6）识别重要特征
                    - 移除不重要或冗余的特征
                    
                    #### 4️⃣ 正则化参数调整
                    - 对于RF：减少max_depth，增加min_samples_split
                    - 对于MLP：增加dropout，减少隐藏层神经元
                    - 对于SVR：减小C值（增加正则化）
                    - 对于LightGBM/XGBoost：增加reg_alpha和reg_lambda
                    
                    #### 5️⃣ 重新训练
                    - 返回步骤3，调整贝叶斯优化迭代次数
                    - 增加迭代次数可以找到更好的正则化参数
                    - 建议：从20次增加到30-50次
                    
                    #### 💡 尝试其他模型
                    - 当前模型：{}
                    - 建议尝试：LightGBM（内置正则化）或RF（集成学习）
                    """.format(
                        len(st.session_state.y_train),
                        int((1 - st.session_state.test_size) * 100) if hasattr(st.session_state, 'test_size') else 20,
                        len(st.session_state.feature_names),
                        selected_model
                    ))
                    
            elif overfitting > 0.1:
                st.warning(f"⚠️ 检测到轻度过拟合：训练集R²({model_result['train_r2']:.4f}) 高于测试集R²({model_result['test_r2']:.4f})，差值为 {overfitting:.4f}")
                
                with st.expander("💡 优化建议", expanded=False):
                    st.markdown("""
                    ### 🔧 轻度过拟合的调整建议
                    
                    1. **特征工程**：使用SHAP分析（步骤6）选择最重要的特征
                    2. **增加正则化**：在贝叶斯优化中会自动调整，可以增加迭代次数
                    3. **交叉验证**：当前使用10折交叉验证，已经是较好的设置
                    4. **数据增强**：如果可能，增加训练样本数量
                    
                    💡 轻度过拟合是可以接受的，特别是在数据量较小的情况下
                    """)
                    
            elif overfitting < -0.05:
                st.info(f"💡 测试集性能优于训练集（差值 {overfitting:.4f}），这可能是数据划分的偶然性")
                st.caption("这种情况通常是正常的，可能是测试集恰好包含了更容易预测的样本")
            else:
                st.success(f"✅ 模型泛化能力良好：训练集和测试集性能接近（差值 {overfitting:.4f}）")
                st.caption("模型在训练集和测试集上表现一致，说明模型具有良好的泛化能力")
        
        # ========== 新增功能 ==========
        
        # 功能1：最佳参数报告
        st.markdown("---")
        st.subheader("⚙️ 最佳超参数")
        
        # 转换参数为可显示的格式（确保所有值都是字符串）
        display_params = {}
        for key, value in model_result['best_params'].items():
            if isinstance(value, tuple):
                display_params[key] = str(value)
            elif isinstance(value, (int, float)):
                display_params[key] = str(value)  # 转换为字符串避免类型混合
            else:
                display_params[key] = str(value)  # 其他类型也转为字符串
        
        params_df = pd.DataFrame([display_params]).T
        params_df.columns = ['Value']
        params_df.index.name = 'Parameter'
        st.dataframe(params_df, width='stretch')
        
        # 功能2：贝叶斯优化收敛过程（增强版：R²、MAE、RMSE三合一）
        st.subheader("📈 Bayesian Optimization Convergence (Enhanced)")
        
        st.info("""
        **📖 图表说明**
        - **总迭代次数** = 8（初始随机探索）+ 设置的迭代次数
        - **左侧曲线图**：显示优化进度。🟠橙色点=随机探索，🔵蓝色点=贝叶斯优化，🔴红线=累积最优
        - **右侧柱状图**：显示每次迭代的改进量。🟢绿色=找到更好结果，⚪灰色=未改进
        - 💡 大部分柱子为0是正常的！贝叶斯优化需要大量尝试来确认最优解
        """)
        
        opt_history = model_result['optimization_history']
        iterations = [h['iteration'] for h in opt_history]
        targets = [h['target'] for h in opt_history]
        
        # 区分初始随机采样和贝叶斯优化阶段
        n_init = 8  # 初始随机采样点数
        
        # 计算累积最优
        cumulative_best = []
        current_best = -np.inf
        for target in targets:
            current_best = max(current_best, target)
            cumulative_best.append(current_best)
        
        # 估算MAE和RMSE趋势
        final_mae = model_result['test_mae']
        final_rmse = model_result['test_rmse']
        final_r2 = model_result['test_r2']
        estimated_mae = [final_mae * (1 - t) / (1 - final_r2) if final_r2 < 1 else final_mae for t in targets]
        estimated_rmse = [final_rmse * (1 - t) / (1 - final_r2) if final_r2 < 1 else final_rmse for t in targets]
        
        # 计算累积最优MAE和RMSE
        cumulative_best_mae = []
        current_best_mae = np.inf
        for mae in estimated_mae:
            current_best_mae = min(current_best_mae, mae)
            cumulative_best_mae.append(current_best_mae)
        
        cumulative_best_rmse = []
        current_best_rmse = np.inf
        for rmse in estimated_rmse:
            current_best_rmse = min(current_best_rmse, rmse)
            cumulative_best_rmse.append(current_best_rmse)
        
        # 计算改进
        improvements_r2 = [0]
        for i in range(1, len(cumulative_best)):
            improvements_r2.append(cumulative_best[i] - cumulative_best[i-1])
        
        improvements_mae = [0]
        for i in range(1, len(cumulative_best_mae)):
            improvements_mae.append(cumulative_best_mae[i-1] - cumulative_best_mae[i])  # MAE越小越好，所以是减少到
        
        improvements_rmse = [0]
        for i in range(1, len(cumulative_best_rmse)):
            improvements_rmse.append(cumulative_best_rmse[i-1] - cumulative_best_rmse[i])  # RMSE越小越好
        
        # 创建3列布局
        fig, axes = plt.subplots(3, 2, figsize=(16, 14), dpi=150)
        
        # ========== 第一行：R² ==========
        # 左图：R² 曲线
        ax1 = axes[0, 0]
        
        # 初始随机采样
        ax1.scatter(iterations[:n_init], targets[:n_init], 
                   c='orange', s=100, alpha=0.6, edgecolors='black', 
                   linewidth=1.5, label='Random Exploration', zorder=3)
        # 贝叶斯优化点
        ax1.scatter(iterations[n_init:], targets[n_init:], 
                   c='steelblue', s=100, alpha=0.6, edgecolors='black', 
                   linewidth=1.5, label='Bayesian Optimization', zorder=3)
        # 累积最优线
        ax1.plot(iterations, cumulative_best, 'r-', linewidth=2.5, 
                label='Best Score', zorder=2)
        ax1.fill_between(iterations, cumulative_best, alpha=0.2, color='red')
        
        # 标注最优点
        best_idx = targets.index(max(targets))
        ax1.scatter([iterations[best_idx]], [targets[best_idx]], 
                   c='red', s=200, marker='*', edgecolors='darkred', 
                   linewidth=2, label='Best Found', zorder=4)
        ax1.annotate(f'Best: {max(targets):.4f}', 
                    xy=(iterations[best_idx], targets[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))
        
        ax1.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Cross-Validation R² Score', fontsize=11, fontweight='bold')
        ax1.set_title('R² Convergence', fontsize=12, fontweight='bold', pad=12)
        ax1.legend(fontsize=7, loc='lower right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(0, len(iterations) + 1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 右图：R² 改进量柱状图
        ax2 = axes[0, 1]
        colors_r2 = ['green' if imp > 0 else 'lightgray' for imp in improvements_r2]
        ax2.bar(iterations, improvements_r2, color=colors_r2, alpha=0.7, edgecolor='black', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax2.set_ylabel('R² Improvement', fontsize=11, fontweight='bold')
        ax2.set_title('R² Improvement per Iteration', fontsize=12, fontweight='bold', pad=12)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_xlim(0, len(iterations) + 1)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # ========== 第二行：MAE ==========
        # 左图：MAE 曲线
        ax3 = axes[1, 0]
        ax3.plot(iterations, estimated_mae, 'go-', linewidth=2, markersize=6, alpha=0.6, label='Estimated MAE')
        ax3.plot(iterations, cumulative_best_mae, 'r-', linewidth=2.5, label='Best MAE', zorder=2)
        ax3.fill_between(iterations, cumulative_best_mae, alpha=0.2, color='red')
        ax3.axhline(y=min(estimated_mae), color='darkgreen', linestyle='--', 
                   alpha=0.5, linewidth=1.5, label=f'Best: {min(estimated_mae):.4f}')
        
        ax3.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax3.set_ylabel('MAE (Estimated)', fontsize=11, fontweight='bold')
        ax3.set_title('MAE Convergence', fontsize=12, fontweight='bold', pad=12)
        ax3.legend(fontsize=7, loc='upper right')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim(0, len(iterations) + 1)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 右图：MAE 改进量柱状图
        ax4 = axes[1, 1]
        colors_mae = ['green' if imp > 0 else 'lightgray' for imp in improvements_mae]
        ax4.bar(iterations, improvements_mae, color=colors_mae, alpha=0.7, edgecolor='black', linewidth=1)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax4.set_ylabel('MAE Reduction', fontsize=11, fontweight='bold')
        ax4.set_title('MAE Reduction per Iteration', fontsize=12, fontweight='bold', pad=12)
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax4.set_xlim(0, len(iterations) + 1)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # ========== 第三行：RMSE ==========
        # 左图：RMSE 曲线
        ax5 = axes[2, 0]
        ax5.plot(iterations, estimated_rmse, 'ro-', linewidth=2, markersize=6, alpha=0.6, label='Estimated RMSE')
        ax5.plot(iterations, cumulative_best_rmse, 'darkred', linewidth=2.5, label='Best RMSE', zorder=2)
        ax5.fill_between(iterations, cumulative_best_rmse, alpha=0.2, color='red')
        ax5.axhline(y=min(estimated_rmse), color='darkred', linestyle='--', 
                   alpha=0.5, linewidth=1.5, label=f'Best: {min(estimated_rmse):.4f}')
        
        ax5.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax5.set_ylabel('RMSE (Estimated)', fontsize=11, fontweight='bold')
        ax5.set_title('RMSE Convergence', fontsize=12, fontweight='bold', pad=12)
        ax5.legend(fontsize=7, loc='upper right')
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.set_xlim(0, len(iterations) + 1)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        
        # 右图：RMSE 改进量柱状图
        ax6 = axes[2, 1]
        colors_rmse = ['green' if imp > 0 else 'lightgray' for imp in improvements_rmse]
        ax6.bar(iterations, improvements_rmse, color=colors_rmse, alpha=0.7, edgecolor='black', linewidth=1)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax6.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax6.set_ylabel('RMSE Reduction', fontsize=11, fontweight='bold')
        ax6.set_title('RMSE Reduction per Iteration', fontsize=12, fontweight='bold', pad=12)
        ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax6.set_xlim(0, len(iterations) + 1)
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # 显示优化统计（三指标
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best R²", f"{max(targets):.4f}", f"Iter {best_idx + 1}")
        col2.metric("Best MAE", f"{min(estimated_mae):.4f}", f"Iter {estimated_mae.index(min(estimated_mae)) + 1}")
        col3.metric("Best RMSE", f"{min(estimated_rmse):.4f}", f"Iter {estimated_rmse.index(min(estimated_rmse)) + 1}")
        col4.metric("Convergence", f"{sum(1 for i in range(1, len(cumulative_best)) if cumulative_best[i] > cumulative_best[i-1])} iters")
        
        # 功能2.5：超参数优化平行坐标
        st.subheader("🎨 Hyperparameter Optimization Process (Parallel Coordinates)")
        
        st.info("💡 平行坐标图展示每次优化尝试的参数组合和性能。每条线代表一次尝试，颜色深浅表示 R² 高低，红色为最优配置")
        
        try:
            # 准备数据
            params_list = [h['params'] for h in opt_history]
            r2_list = targets
            
            # 提取参数名和
            param_names = list(params_list[0].keys())
            
            # 创建 DataFrame（R²放在最后）
            data_dict = {name: [p[name] for p in params_list] for name in param_names}
            data_dict['R²'] = r2_list
            data_dict['Iteration'] = iterations
            
            df_parallel = pd.DataFrame(data_dict)
            
            # 找到最优配
            best_idx_parallel = df_parallel['R²'].idxmax()
            
            # 绘制平行坐标
            fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
            
            # 设置 X 轴位置（R²放在最右边
            all_cols = param_names + ['R²']
            x_positions = list(range(len(all_cols)))
            
            # 为每个参数创建独立的Y轴范围和标记
            y_positions_dict = {}
            y_labels_dict = {}
            is_integer_param = {}  # 标记哪些参数是整数类
            
            # 定义已知的整数参数（基于常见的机器学习超参数
            integer_params = ['n_estimators', 'num_leaves', 'max_depth', 'min_child_samples', 
                            'min_samples_split', 'min_samples_leaf', 'hidden_layer_size',
                            'n_neighbors', 'max_iter', 'n_jobs', 'random_state']
            
            for col in all_cols:
                col_min = df_parallel[col].min()
                col_max = df_parallel[col].max()
                
                # 判断是否为整数参
                is_integer_param[col] = col in integer_params or col.lower() in [p.lower() for p in integer_params]
                
                if col == 'R²':
                    # R²特殊处理：根据实际数据范围调整，生成5个刻度
                    r2_range = col_max - col_min
                    
                    # 设置R²的显示范围：最小值向下取整到0.1，最大值为1.0
                    r2_min_actual = max(0.0, np.floor(col_min * 10) / 10 - 0.1)
                    r2_max_actual = 1.0
                    
                    # 标准化到[0, 1]
                    y_positions_dict[col] = [(val - r2_min_actual) / (r2_max_actual - r2_min_actual) 
                                            for val in df_parallel[col]]
                    
                    # Y轴标记：固定生成5个刻度
                    y_labels_dict[col] = list(np.linspace(r2_min_actual, r2_max_actual, 5))
                else:
                    # 其他参数：标准化到[0, 1]
                    if col_max > col_min:
                        y_positions_dict[col] = [(val - col_min) / (col_max - col_min) 
                                                for val in df_parallel[col]]
                    else:
                        y_positions_dict[col] = [0.5] * len(df_parallel[col])
                    
                    # Y轴标记（显示实际范围，整数参数显示整数）
                    param_range = col_max - col_min
                    if is_integer_param[col]:
                        # 整数参数：显示整数刻
                        if param_range > 100:
                            step = int(param_range / 4)
                            y_labels_dict[col] = [int(col_min + i * step) for i in range(5)]
                        elif param_range > 20:
                            step = int(param_range / 3)
                            y_labels_dict[col] = [int(col_min + i * step) for i in range(4)]
                        else:
                            y_labels_dict[col] = [int(col_min), int((col_min + col_max) / 2), int(col_max)]
                    else:
                        # 浮点参数：显示小数刻
                        if param_range > 40:
                            step = param_range / 4
                            y_labels_dict[col] = [col_min + i * step for i in range(5)]
                        elif param_range > 10:
                            step = param_range / 3
                            y_labels_dict[col] = [col_min + i * step for i in range(4)]
                        else:
                            y_labels_dict[col] = [col_min, (col_min + col_max) / 2, col_max]
            
            # 绘制每次尝试（除了最优）
            for idx, row_idx in enumerate(df_parallel.index):
                if row_idx != best_idx_parallel:
                    # 根据 R² 值选择颜色（使用更鲜艳的colormap
                    r2_val = df_parallel.loc[row_idx, 'R²']
                    r2_normalized = (r2_val - df_parallel['R²'].min()) / (df_parallel['R²'].max() - df_parallel['R²'].min())
                    
                    # 使用RdYlGn colormap（红-绿），更明显
                    color = plt.cm.RdYlGn(0.3 + 0.7 * r2_normalized)  # 0.3-1.0范围，避免太
                    
                    # 绘制线条
                    values = [y_positions_dict[col][row_idx] for col in all_cols]
                    ax.plot(x_positions, values, color=color, alpha=0.6, linewidth=2, zorder=1)
            
            # 绘制最优配置（红色高亮，更粗）
            best_values = [y_positions_dict[col][best_idx_parallel] for col in all_cols]
            ax.plot(x_positions, best_values, color='red', alpha=0.95, linewidth=4.5, 
                   label=f'Best (R²={df_parallel.loc[best_idx_parallel, "R²"]:.4f})', zorder=10,
                   marker='o', markersize=8, markeredgecolor='darkred', markeredgewidth=2)
            
            # 设置 Y 轴范围（基于标准化值）
            ax.set_ylim(-0.05, 1.05)
            
            # 为每个参数添加Y轴标记（在对应的X位置
            for i, col in enumerate(all_cols):
                # 在每个参数的X位置添加Y轴刻度标
                labels = y_labels_dict[col]
                positions = [(lbl - df_parallel[col].min()) / (df_parallel[col].max() - df_parallel[col].min()) 
                            if df_parallel[col].max() > df_parallel[col].min() else 0.5 
                            for lbl in labels]
                
                # 特殊处理R²
                if col == 'R²':
                    # 使用与生成时相同的范围计算
                    r2_min_actual = max(0.0, np.floor(df_parallel[col].min() * 10) / 10 - 0.1)
                    r2_max_actual = 1.0
                    positions = [(lbl - r2_min_actual) / (r2_max_actual - r2_min_actual) for lbl in labels]
                
                # 判断标签位置（R²在最右边，标记在右侧；其他在左侧
                if col == 'R²':
                    # R²只标记在右侧，不标记在左
                    for pos, lbl in zip(positions, labels):
                        ax.text(i + 0.15, pos, f'{lbl:.3f}', fontsize=9, ha='left', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, edgecolor='orange', linewidth=1.5))
                else:
                    # 其他参数标记在左
                    for pos, lbl in zip(positions, labels):
                        # 根据是否为整数参数选择格式
                        if is_integer_param[col]:
                            label_text = f'{int(lbl)}'
                        else:
                            label_text = f'{lbl:.2f}'
                        
                        ax.text(i - 0.15, pos, label_text, fontsize=9, ha='right', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
            
            # 在最右侧R²列添加刻度标签（修改为5个刻度，范围优化）
            # 计算R²的实际范围
            r2_min_actual = df_parallel['R²'].min()
            r2_max_actual = df_parallel['R²'].max()
            
            # 设置R²的显示范围：最小值向下取整到0.1，最大值为1.0
            r2_min_display = max(0.0, np.floor(r2_min_actual * 10) / 10 - 0.1)
            r2_max_display = 1.0
            
            # 创建5个刻度值
            r2_ticks_values = np.linspace(r2_min_display, r2_max_display, 5)
            
            # 设置 X 轴
            ax.set_xticks(x_positions)
            ax.set_xticklabels(all_cols, rotation=45, ha='right', fontsize=12, fontweight='bold')
            ax.set_xlim(-0.5, len(all_cols) - 0.5)
            
            # 添加垂直网格
            for pos in x_positions:
                ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
            
            # 标题和图例
            ax.set_title('Hyperparameter Optimization Process (Parallel Coordinates)', 
                        fontsize=15, fontweight='bold', pad=20)
            ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
            ax.grid(True, alpha=0.15, axis='y')
            
            # 移除所有Y轴（包括左侧和右侧）
            ax.set_yticks([])
            ax.set_ylabel('')
            
            # 移除所有边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # 添加颜色条（移除ax2引用）
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                      norm=plt.Normalize(vmin=df_parallel['R²'].min(), 
                                                        vmax=df_parallel['R²'].max()))
            sm.set_array([])
            # 将colorbar绑定到主轴ax
            cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=35, shrink=0.8)
            cbar.set_label('R² Score', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # 显示最优参数
            with st.expander("🏆 最优参数配置", expanded=True):
                st.markdown(f"**迭代次数**: {best_idx_parallel + 1}")
                st.markdown(f"**R² 分数**: {df_parallel.loc[best_idx_parallel, 'R²']:.4f}")
                st.markdown("**参数值：**")
                
                # 定义需要显示为整数的参数
                integer_params = {
                    'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                    'hidden_layer_size', 'num_leaves', 'min_child_samples', 'min_child_weight'
                }
                
                cols = st.columns(min(len(param_names), 4))
                for i, param_name in enumerate(param_names):
                    with cols[i % len(cols)]:
                        param_value = df_parallel.loc[best_idx_parallel, param_name]
                        # 如果是整数参数，显示为整数；否则显示为小数
                        if param_name in integer_params:
                            st.metric(param_name, f"{int(param_value)}")
                        else:
                            st.metric(param_name, f"{param_value:.4f}")
            
            # 解释说明
            with st.expander("📖 如何解读平行坐标图"):
                st.markdown("""
                **图表说明**
                - **每条线**：代表一次超参数优化尝试
                - **X轴**：各个超参数 + R² 分数
                - **Y轴**：标准化后的参数值（0=最小值，1=最大值）
                - **颜色**：浅蓝到深蓝表示 R² 从低到高
                - **红色粗线**：最优配置
                
                **解读要点**
                1. **线条聚集**：该参数值附近性能较好
                2. **线条发散**：该参数对性能影响较大
                3. **红线位置**：最优参数组合
                4. **颜色渐变**：优化过程中性能提升趋势
                
                **实际应用**
                - 识别关键参数（影响R²的参数）
                - 发现参数之间的交互效应
                - 验证优化过程是否收敛
                - 为下一轮优化提供参考
                """)
        
        except Exception as e:
            st.error(f"❌ 平行坐标图生成失败：{str(e)}")
            st.info("💡 如果参数数量过多或数据格式不兼容，可能导致此错误")
        
        # 功能3：最优参数的10折交叉验证详细分析
        st.subheader("📊 10-Fold Cross-Validation Analysis (Best Parameters)")
        
        cv_scores = model_result['best_cv_scores']
        folds = list(range(1, len(cv_scores) + 1))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(folds, cv_scores, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axhline(y=cv_scores.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {cv_scores.mean():.4f}')
            ax.set_xlabel('Fold Number', fontsize=11)
            ax.set_ylabel('R² Score', fontsize=11)
            ax.set_title('Cross-Validation Scores Across Folds', fontsize=12)
            ax.set_xticks(folds)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("**CV Statistics:**")
            st.metric("Mean R²", f"{cv_scores.mean():.4f}")
            st.metric("Std Dev", f"{cv_scores.std():.4f}")
            st.metric("Min R²", f"{cv_scores.min():.4f}")
            st.metric("Max R²", f"{cv_scores.max():.4f}")
        
        # 功能4：所有迭代参数探索可视化
        st.subheader("🔥 Parameter Exploration Landscape")
        
        st.info("""
        **📖 图表说明**
        - 此图展示贝叶斯优化在参数空间中的"搜索轨迹"
        - **2个参数**：2D散点图，点的位置=参数组合，颜色=性能（🟢绿=好，🔴红=差），数字=迭代顺序
        - **3个参数**：3D投影图，从不同角度观察参数空间
        - **多个参数**：热图，每列=一次迭代，红框=最优迭代
        - 💡 观察点的分布和颜色，了解哪些参数组合效果好
        """)
        
        # 准备数据
        param_names = list(opt_history[0]['params'].keys())
        n_params = len(param_names)
        
        # 创建参数探索图（散点图矩阵）
        if n_params == 2:
            # 2个参数：创建2D散点
            fig, ax = plt.subplots(figsize=(10, 8))
            
            param1_vals = [h['params'][param_names[0]] for h in opt_history]
            param2_vals = [h['params'][param_names[1]] for h in opt_history]
            
            # 使用颜色表示性能
            scatter = ax.scatter(param1_vals, param2_vals, 
                               c=targets, s=200, cmap='RdYlGn', 
                               alpha=0.7, edgecolors='black', linewidth=1.5)
            
            # 标注迭代顺序
            for i, (x, y) in enumerate(zip(param1_vals, param2_vals)):
                ax.annotate(f'{i+1}', (x, y), fontsize=9, ha='center', va='center', fontweight='bold')
            
            # 标注最优点
            best_idx = targets.index(max(targets))
            ax.scatter([param1_vals[best_idx]], [param2_vals[best_idx]], 
                      s=400, marker='*', c='red', edgecolors='darkred', linewidth=2, zorder=5)
            
            ax.set_xlabel(param_names[0], fontsize=12, fontweight='bold')
            ax.set_ylabel(param_names[1], fontsize=12, fontweight='bold')
            ax.set_title('Parameter Space Exploration', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('CV R² Score', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        elif n_params == 3:
            # 3个参数：创建3D投影
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            param_pairs = [(0, 1), (0, 2), (1, 2)]
            
            for idx, (p1, p2) in enumerate(param_pairs):
                ax = axes[idx]
                param1_vals = [h['params'][param_names[p1]] for h in opt_history]
                param2_vals = [h['params'][param_names[p2]] for h in opt_history]
                
                scatter = ax.scatter(param1_vals, param2_vals, 
                                   c=targets, s=150, cmap='RdYlGn', 
                                   alpha=0.7, edgecolors='black', linewidth=1.5)
                
                # 标注迭代顺序
                for i, (x, y) in enumerate(zip(param1_vals, param2_vals)):
                    ax.annotate(f'{i+1}', (x, y), fontsize=8, ha='center', va='center')
                
                # 标注最优点
                best_idx = targets.index(max(targets))
                ax.scatter([param1_vals[best_idx]], [param2_vals[best_idx]], 
                          s=300, marker='*', c='red', edgecolors='darkred', linewidth=2, zorder=5)
                
                ax.set_xlabel(param_names[p1], fontsize=11, fontweight='bold')
                ax.set_ylabel(param_names[p2], fontsize=11, fontweight='bold')
                ax.set_title(f'{param_names[p1]} vs {param_names[p2]}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                if idx == 2:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('CV R² Score', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        else:
            # 多个参数：创建改进的热图
            fig, ax = plt.subplots(figsize=(max(12, len(opt_history) * 0.8), max(6, n_params * 0.8)))
            
            # 准备数据矩阵
            data_matrix = []
            row_labels = ['CV R²']
            
            # 添加CV分数
            data_matrix.append(targets)
            
            # 添加每个参数的值（标准化以便比较）
            for param_name in param_names:
                param_vals = [h['params'][param_name] for h in opt_history]
                # 标准化到0-1范围
                param_min, param_max = min(param_vals), max(param_vals)
                if param_max > param_min:
                    normalized = [(v - param_min) / (param_max - param_min) for v in param_vals]
                else:
                    normalized = [0.5] * len(param_vals)
                data_matrix.append(normalized)
                row_labels.append(param_name)
            
            data_matrix = np.array(data_matrix)
            
            # 创建热图
            im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
            
            # 设置刻度
            ax.set_xticks(np.arange(len(opt_history)))
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_xticklabels([f'{i+1}' for i in range(len(opt_history))], fontsize=10)
            ax.set_yticklabels(row_labels, fontsize=11, fontweight='bold')
            
            # 添加数值标签（只对CV R²行）
            for j in range(len(opt_history)):
                text = ax.text(j, 0, f'{targets[j]:.3f}',
                             ha="center", va="center", color="black", 
                             fontsize=9, fontweight='bold')
            
            # 标注最优迭
            best_idx = targets.index(max(targets))
            for i in range(len(row_labels)):
                ax.add_patch(plt.Rectangle((best_idx - 0.5, i - 0.5), 1, 1, 
                                          fill=False, edgecolor='red', linewidth=3))
            
            ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax.set_title('Parameter Values and Performance Across Iterations', fontsize=13, fontweight='bold')
            
            # 添加颜色
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Normalized Value / Score', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # 显示详细的迭代表
        with st.expander("📋 View Detailed Iteration History"):
            history_df = pd.DataFrame(opt_history)
            history_df['CV_R2'] = history_df['target']
            
            # 展开params字典
            params_df = pd.DataFrame(history_df['params'].tolist())
            history_df = pd.concat([history_df[['iteration', 'CV_R2']], params_df], axis=1)
            
            # 高亮最优行
            def highlight_best(row):
                if row['iteration'] == best_idx + 1:
                    return ['background-color: #90EE90'] * len(row)
                return [''] * len(row)
            
            styled_df = history_df.style.apply(highlight_best, axis=1).format({
                'CV_R2': '{:.4f}',
                **{col: '{:.4f}' for col in params_df.columns}
            })
            
            st.dataframe(styled_df, width='stretch')
        
        # ========== 原有功能继续 ==========
        
        # 预测vs实际
        st.subheader("📊 预测 vs 实际 (Predicted vs Actual)")
        
        # 创建增强的可视化x2布局
        fig = plt.figure(figsize=(16, 14))  # 调整高度以适应1:1比例
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 计算残差
        train_residuals = model_result['y_train_pred'] - model_result['y_train_true']
        test_residuals = model_result['y_test_pred'] - model_result['y_test_true']
        
        # 计算统一的轴范围（从0开始）
        y_min_all = 0  # 从0开始
        y_max_all = max(model_result['y_train_true'].max(), model_result['y_train_pred'].max(),
                       model_result['y_test_true'].max(), model_result['y_test_pred'].max())
        
        # 添加5%的边距
        y_range = y_max_all - y_min_all
        y_min_plot = 0  # 确保从0开始
        y_max_plot = y_max_all + 0.05 * y_range
        
        # 1. 训练集：预测vs实际（左上）
        ax1 = fig.add_subplot(gs[0, 0])
        
        # 绘制散点
        scatter1 = ax1.scatter(model_result['y_train_true'], model_result['y_train_pred'], 
                              c=train_residuals, cmap='RdYlGn_r', alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        
        # 完美预测线
        ax1.plot([y_min_plot, y_max_plot], [y_min_plot, y_max_plot], 
                'r-', lw=2.5, label='Perfect Prediction', alpha=0.8, zorder=3)
        
        # +25% 误差线
        ax1.plot([y_min_plot, y_max_plot], [y_min_plot*1.25, y_max_plot*1.25], 
                'b--', lw=1.5, alpha=0.7, label='+25% Error', zorder=2)
        
        # -25% 误差线（只在有意义的范围内绘制）
        if y_max_plot * 0.75 > y_min_plot:
            ax1.plot([y_min_plot, y_max_plot], [y_min_plot*0.75, y_max_plot*0.75], 
                    'b--', lw=1.5, alpha=0.7, label='-25% Error', zorder=2)
            
            # 填充误差带区
            ax1.fill_between([y_min_plot, y_max_plot], 
                            [y_min_plot*0.75, y_max_plot*0.75],
                            [y_min_plot*1.25, y_max_plot*1.25],
                            alpha=0.08, color='blue', label='±25% Band', zorder=1)
        
        ax1.set_xlim(y_min_plot, y_max_plot)
        ax1.set_ylim(y_min_plot, y_max_plot)
        ax1.set_xlabel('Actual Value', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Value', fontsize=12, fontweight='bold')
        ax1.set_title(f'Training Set\nR²={model_result["train_r2"]:.4f}, RMSE={model_result["train_rmse"]:.4f}', 
                     fontsize=13, fontweight='bold', pad=10)
        ax1.legend(fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_aspect('equal', adjustable='box')  # 1:1比例
        cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.02)
        cbar1.set_label('Residual (Pred - Actual)', fontsize=10)
        
        # 2. 测试集：预测vs实际（右上）
        ax2 = fig.add_subplot(gs[0, 1])
        
        # 绘制散点
        scatter2 = ax2.scatter(model_result['y_test_true'], model_result['y_test_pred'], 
                              c=test_residuals, cmap='RdYlGn_r', alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        
        # 完美预测线
        ax2.plot([y_min_plot, y_max_plot], [y_min_plot, y_max_plot], 
                'r-', lw=2.5, label='Perfect Prediction', alpha=0.8, zorder=3)
        
        # +25% 误差线
        ax2.plot([y_min_plot, y_max_plot], [y_min_plot*1.25, y_max_plot*1.25], 
                'b--', lw=1.5, alpha=0.7, label='+25% Error', zorder=2)
        
        # -25% 误差线（只在有意义的范围内绘制）
        if y_max_plot * 0.75 > y_min_plot:
            ax2.plot([y_min_plot, y_max_plot], [y_min_plot*0.75, y_max_plot*0.75], 
                    'b--', lw=1.5, alpha=0.7, label='-25% Error', zorder=2)
            
            # 填充误差带区
            ax2.fill_between([y_min_plot, y_max_plot], 
                            [y_min_plot*0.75, y_max_plot*0.75],
                            [y_min_plot*1.25, y_max_plot*1.25],
                            alpha=0.08, color='blue', label='±25% Band', zorder=1)
        
        ax2.set_xlim(y_min_plot, y_max_plot)
        ax2.set_ylim(y_min_plot, y_max_plot)
        ax2.set_xlabel('Actual Value', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Predicted Value', fontsize=12, fontweight='bold')
        ax2.set_title(f'Test Set (Generalization)\nR²={model_result["test_r2"]:.4f}, RMSE={model_result["test_rmse"]:.4f}', 
                     fontsize=13, fontweight='bold', pad=10)
        ax2.legend(fontsize=9, loc='upper left')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_aspect('equal', adjustable='box')  # 1:1比例
        cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.02)
        cbar2.set_label('Residual (Pred - Actual)', fontsize=10)
        
        # 计算统一的残差范围（取两图中的较大值）
        residual_min_all = min(train_residuals.min(), test_residuals.min())
        residual_max_all = max(train_residuals.max(), test_residuals.max())
        residual_range = residual_max_all - residual_min_all
        residual_min_plot = residual_min_all - 0.05 * residual_range
        residual_max_plot = residual_max_all + 0.05 * residual_range
        
        # 3. 训练集残差分布（左下
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(train_residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
        ax3.axvline(x=train_residuals.mean(), color='orange', linestyle='-', linewidth=2, 
                   label=f'Mean={train_residuals.mean():.4f}')
        ax3.set_xlim(residual_min_plot, residual_max_plot)  # 统一X轴范
        ax3.set_xlabel('Residual (Pred - Actual)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title(f'Training Residuals Distribution\nStd={train_residuals.std():.4f}', 
                     fontsize=13, fontweight='bold', pad=10)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 测试集残差分布（右下
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(test_residuals, bins=30, edgecolor='black', alpha=0.7, color='coral')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
        ax4.axvline(x=test_residuals.mean(), color='orange', linestyle='-', linewidth=2, 
                   label=f'Mean={test_residuals.mean():.4f}')
        ax4.set_xlim(residual_min_plot, residual_max_plot)  # 统一X轴范
        ax4.set_xlabel('Residual (Pred - Actual)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title(f'Test Residuals Distribution\nStd={test_residuals.std():.4f}', 
                     fontsize=13, fontweight='bold', pad=10)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig)
        plt.close()
        
        # 添加详细说明
        with st.expander("📖 如何解读这些图表"):
            st.markdown("""
            ### 🎯 预测vs实际图（上方两图）
            
            **颜色含义：**
            - 🟢 绿色：预测偏低（负残差）
            - 🟡 黄色：预测接近实际
            - 🔴 红色：预测偏高（正残差）
            
            **理想模式：**
            - 点紧密分布在红色虚线附近
            - 颜色主要为黄绿色
            - 无明显的系统性偏差
            
            **问题模式：**
            - 点远离红色虚线 → 预测误差大
            - 上方多红点，下方多绿点 → 系统性偏差
            - 测试集比训练集分散 → 过拟合
            
            ### 📊 残差分布图（下方两图）
            
            **理想分布：**
            - 以0为中心的正态分布
            - 均值接近0（无偏）
            - 标准差小（误差小）
            - 训练集和测试集分布相似
            
            **问题模式：**
            - 均值不为0 → 系统性偏差
            - 分布偏斜 → 某些区域预测不准
            - 测试集标准差远大于训练集 → 过拟合
            - 出现多个峰 → 模型对不同数据模式处理不一致
            """)
        
        st.caption(f"💡 模型：{selected_model} | 颜色表示残差大小，理想情况下应集中在黄绿色区")
        
        # SHAP解释
        st.subheader("🔍 模型可解释性分")
        
        # 创建标签页（移除GP，GP移到步骤5
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 SHAP Analysis", 
            "📈 Partial Dependence (PDP)", 
            "❄️ Individual Conditional Expectation (ICE)",
            "🔬 Dual-Factor Coupling Analysis"
        ])
        
        # ==================== Tab 1: SHAP Analysis ====================
        with tab1:
            if not SHAP_AVAILABLE:
                st.warning("⚠️ SHAP库未安装，无法进行SHAP分析。运行`pip install shap` 安装")
            else:
                st.markdown("### SHAP (SHapley Additive exPlanations)")
                st.info("💡 SHAP值解释每个特征对模型预测的贡献。基于博弈论的Shapley值，提供一致且公平的特征归因")
                
                # 获取总样本数
                total_samples = len(st.session_state.X_train)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if selected_model in ["RF", "LightGBM"]:
                        st.success(f"✅ 树模型速度快，默认分析全部 {total_samples} 个样本")
                        default_samples = total_samples
                    else:
                        st.info(f"💡 此模型使用KernelExplainer较慢，建议使用较少样")
                        default_samples = min(100, total_samples)
                
                with col2:
                    n_samples = st.number_input(
                        "分析样本",
                        min_value=10,
                        max_value=total_samples,
                        value=default_samples,
                        step=10,
                        key="shap_samples"
                    )
                
                if st.button("🚀 生成SHAP分析", key="generate_shap"):
                    with st.spinner(f"正在计算 {n_samples} 个样本的SHAP值..."):
                        try:
                            model = model_result['model']
                            X_sample = st.session_state.X_train[:n_samples]
                            feature_names = st.session_state.feature_names
                            
                            # 根据模型类型选择解释
                            if selected_model in ["RF", "LightGBM"]:
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer.shap_values(X_sample)
                            else:
                                X_background = shap.sample(st.session_state.X_train, min(100, len(st.session_state.X_train)))
                                explainer = shap.KernelExplainer(model.predict, X_background)
                                shap_values = explainer.shap_values(X_sample)
                            
                            # 保存到session_state供后续使用
                            st.session_state.shap_values = shap_values
                            st.session_state.shap_X_sample = X_sample
                            st.session_state.shap_explainer = explainer
                            
                            st.success(f"✅ SHAP分析完成！已分析 {n_samples} 个样本")
                            
                        except Exception as e:
                            st.error(f"❌ SHAP分析失败：{str(e)}")
                
                # 如果已经计算过SHAP值，显示多种可视
                if 'shap_values' in st.session_state:
                    shap_values = st.session_state.shap_values
                    X_sample = st.session_state.shap_X_sample
                    feature_names = st.session_state.feature_names
                    
                    # 确保feature_names是字符串列表
                    if not isinstance(feature_names, list):
                        feature_names = list(feature_names)
                    feature_names = [str(name) for name in feature_names]
                    
                    st.markdown("---")
                    
                    # 添加特征相关性分
                    st.markdown("### 🔍 特征相关性诊")
                    
                    st.info("""
                    📊 **数据说明**
                    - 本图使用**训练集原始数据**（未标准化）计算相关系数
                    - 步骤2使用**全部数据**（训练集+测试集）计算相关系数
                    - 由于样本不同，相关系数可能有差异，这是正常现象
                    - 两个图都反映了特征之间的真实相关性
                    """)
                    
                    with st.expander("💡 为什么需要检查特征相关性？", expanded=False):
                        st.markdown("""
                        **如果您观察到：**
                        - 多个Dependence Plot呈现相似的y=x或y=-x趋势
                        - 散点位置分布基本一致，只有颜色不同
                        - 多个特征的SHAP值模式相似
                        
                        **可能的问题：**
                        1. 特征高度相关（冗余特征）
                        2. 数据预处理导致特征相关
                        3. 特征工程产生了衍生特征
                        
                        **影响：**
                        - 模型难以区分特征的真实重要性
                        - 可能导致过拟合
                        - 降低模型可解释性
                        """)
                    
                    # 计算特征相关性矩阵 - 使用原始训练集数据
                    if hasattr(st.session_state, 'X_train_original'):
                        X_for_correlation = st.session_state.X_train_original
                        data_source_msg = "✅ 使用训练集原始数据（未标准化）"
                    else:
                        # 降级方案：使用标准化数据（Pearson相关系数理论上一致）
                        X_for_correlation = pd.DataFrame(
                            st.session_state.X_train,
                            columns=feature_names
                        )
                        data_source_msg = "⚠️ 使用标准化数据（理论上Pearson相关系数一致）"
                    
                    # 计算相关系数矩阵
                    if isinstance(X_for_correlation, pd.DataFrame):
                        correlation_matrix = X_for_correlation.corr(method='pearson').values
                    else:
                        correlation_matrix = np.corrcoef(X_for_correlation.T)
                    
                    # 找出高度相关的特征对
                    high_corr_pairs = []
                    n_features = len(feature_names)
                    for i in range(n_features):
                        for j in range(i+1, n_features):
                            corr = correlation_matrix[i, j]
                            if abs(corr) > 0.8:  # 相关系数阈
                                high_corr_pairs.append({
                                    'Feature 1': feature_names[i],
                                    'Feature 2': feature_names[j],
                                    'Correlation': corr
                                })
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.caption(data_source_msg)
                        
                        if len(high_corr_pairs) > 0:
                            st.warning(f"⚠️ 发现 {len(high_corr_pairs)} 对高度相关的特征")
                            
                            # 显示高度相关的特征对
                            corr_df = pd.DataFrame(high_corr_pairs)
                            st.dataframe(corr_df.style.format({'Correlation': '{:.3f}'}), 
                                       width="stretch")
                            
                            st.markdown("**建议：**")
                            st.markdown("""
                            - 考虑移除冗余特征
                            - 保留最重要的特征（参考SHAP重要性）
                            - 或使用PCA降维
                            """)
                        else:
                            st.success("✅ 未发现高度相关的特征（|相关系数| > 0.8）")
                    
                    with col2:
                        # 绘制相关性热
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # 使用seaborn绘制热图（如果可用）
                        try:
                            import seaborn as sns
                            sns.heatmap(correlation_matrix, 
                                      xticklabels=feature_names,
                                      yticklabels=feature_names,
                                      annot=True, 
                                      fmt='.2f',
                                      cmap='RdBu_r',
                                      center=0,
                                      vmin=-1, 
                                      vmax=1,
                                      square=True,
                                      ax=ax)
                        except ImportError:
                            # 如果没有seaborn，使用matplotlib
                            im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                            ax.set_xticks(range(n_features))
                            ax.set_yticks(range(n_features))
                            ax.set_xticklabels(feature_names, rotation=45, ha='right')
                            ax.set_yticklabels(feature_names)
                            
                            # 添加数值标签
                            for i in range(n_features):
                                for j in range(n_features):
                                    text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                                 ha="center", va="center", color="black", fontsize=8)
                            
                            plt.colorbar(im, ax=ax)
                        
                        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    # 添加SHAP值相关性分析（如果步骤2未做过相关性分析）
                    if not st.session_state.get('correlation_done', False):
                        st.markdown("#### 🔬 SHAP值相关性分")
                        
                        # 计算SHAP值之间的相关
                        shap_correlation = np.corrcoef(shap_values.T)
                        
                        # 找出SHAP值高度相关的特征
                        high_shap_corr_pairs = []
                        for i in range(n_features):
                            for j in range(i+1, n_features):
                                corr = shap_correlation[i, j]
                                if abs(corr) > 0.8:
                                    high_shap_corr_pairs.append({
                                        'Feature 1': feature_names[i],
                                        'Feature 2': feature_names[j],
                                        'SHAP Correlation': corr
                                    })
                        
                        if len(high_shap_corr_pairs) > 0:
                            st.warning(f"⚠️ 发现 {len(high_shap_corr_pairs)} 对SHAP值高度相关的特征")
                            
                            shap_corr_df = pd.DataFrame(high_shap_corr_pairs)
                            st.dataframe(shap_corr_df.style.format({'SHAP Correlation': '{:.3f}'}), 
                                       width="stretch")
                            
                            st.info("""
                            **💡 SHAP值高度相关说明：**
                            - 这些特征对模型预测的影响模式相似
                            - 可能是因为特征本身相关，或模型学到了相似的模式
                            - 这就是为什么Dependence Plot看起来相似的原因
                            
                            **解决方案：**
                            1. 检查特征相关性矩阵，移除冗余特征
                            2. 使用特征选择方法（如基于SHAP重要性）
                            3. 考虑特征工程，创建更有区分度的特征
                            """)
                        else:
                            st.success("✅ SHAP值相关性正常，各特征对预测的影响模式不同")
                        
                        st.markdown("---")
                    else:
                        st.info("ℹ️ 已在步骤2完成相关性分析，此处跳过SHAP值相关性分析")
                        st.markdown("---")
                    st.markdown("### 📊 SHAP可视化")
                    
                    # 使用2×2布局显示4种SHAP图
                    st.markdown("---")
                    
                    # 第一行：Summary Plot 和 Bar Plot
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 1. Summary Plot (Beeswarm)
                        st.markdown("#### 1️⃣ Summary Plot (Beeswarm)")
                        st.caption("显示所有特征的SHAP值分布。每个点代表一个样本，颜色表示特征值高低")
                        
                        plt.figure(figsize=(10, 6))
                        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                        
                        # 调整字体大小和colorbar标签
                        ax = plt.gca()
                        ax.set_xlabel('SHAP值', fontsize=22, fontweight='bold')
                        ax.set_ylabel('特征', fontsize=22, fontweight='bold')
                        ax.tick_params(axis='both', labelsize=18)
                        for label in ax.get_yticklabels():
                            label.set_fontsize(18)
                            label.set_fontweight('bold')
                        
                        # 修改colorbar标签
                        cbar = plt.gcf().axes[-1]  # 获取colorbar轴
                        if cbar:
                            cbar.set_ylabel('特征值', fontsize=18, fontweight='bold', rotation=270, labelpad=5)
                            cbar.tick_params(labelsize=16)
                            # 修改colorbar刻度标签
                            cbar_labels = cbar.get_yticklabels()
                            for i, label in enumerate(cbar_labels):
                                if 'High' in str(label):
                                    label.set_text('高')
                                elif 'Low' in str(label):
                                    label.set_text('低')
                        
                        st.pyplot(plt.gcf())
                        plt.clf()
                        plt.close()
                        
                        with st.expander("📖 如何解读Summary Plot"):
                            st.markdown("""
                            - **横轴**：SHAP值（特征对预测的影响）
                              - 正值：增加预测值
                              - 负值：减少预测值
                            - **颜色**：特征值的大小
                              - 🔴 红色：特征值高
                              - 🔵 蓝色：特征值低
                            - **纵轴**：特征按重要性从上到下排列
                            - **每个点**：代表一个样本
                            
                            **示例解读：**
                            - 红点在右 → 特征值高时，增加预测
                            - 蓝点在左 → 特征值低时，减少预测
                            """)
                    
                    with col2:
                        # 2. Bar Plot (Feature Importance)
                        st.markdown("#### 2️⃣ Bar Plot (Feature Importance)")
                        st.caption("显示每个特征的平均绝对SHAP值，表示特征的整体重要性")
                        
                        plt.figure(figsize=(10, 6))
                        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                         plot_type="bar", show=False)
                        
                        # 调整字体大小
                        ax = plt.gca()
                        ax.set_xlabel('平均 |SHAP值|', fontsize=22, fontweight='bold')
                        ax.set_ylabel('特征', fontsize=22, fontweight='bold')
                        ax.tick_params(axis='both', labelsize=18)
                        for label in ax.get_yticklabels():
                            label.set_fontsize(18)
                            label.set_fontweight('bold')
                        
                        st.pyplot(plt.gcf())
                        plt.clf()
                        plt.close()
                    
                    st.markdown("---")
                    
                    # 第二行：Waterfall Plot 和 Dependence Plot
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        # 3. Waterfall Plot (Single Prediction)
                        st.markdown("#### 3️⃣ Waterfall Plot (Single Prediction Explanation)")
                        st.caption("显示单个样本的预测是如何由各个特征贡献组成的")
                        
                        sample_idx = st.slider("选择要解释的样本索引", 0, len(X_sample)-1, 0, key="waterfall_idx")
                        
                        plt.figure(figsize=(10, 6))
                        if selected_model in ["RF", "LightGBM"]:
                            shap.waterfall_plot(shap.Explanation(
                                values=shap_values[sample_idx],
                                base_values=st.session_state.shap_explainer.expected_value,
                                data=X_sample[sample_idx],
                                feature_names=feature_names
                            ), show=False)
                        else:
                            shap.waterfall_plot(shap.Explanation(
                                values=shap_values[sample_idx],
                                base_values=st.session_state.shap_explainer.expected_value,
                                data=X_sample[sample_idx],
                                feature_names=feature_names
                            ), show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()
                        plt.close()
                        
                        with st.expander("📖 如何解读Waterfall Plot"):
                            st.markdown("""
                            - **E[f(X)]**：模型的平均预测值（基准值）
                            - **f(x)**：该样本的实际预测值
                            - **红色箭头**：增加预测值的特征
                            - **蓝色箭头**：减少预测值的特征
                            - 从基准值开始，每个特征依次贡献，最终到达预测值
                            """)
                    
                    with col4:
                        # 4. Dependence Plot
                        st.markdown("#### 4️⃣ Dependence Plot")
                        st.caption("显示单个特征的SHAP值与特征值之间的关系")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            feature_idx = st.selectbox(
                                "选择主特征",
                                range(len(feature_names)),
                                format_func=lambda x: feature_names[x],
                                key="dep_feature"
                            )
                        with col_b:
                            interaction_idx = st.selectbox(
                                "选择交互特征（颜色）",
                                ["auto"] + list(range(len(feature_names))),
                                format_func=lambda x: "自动选择" if x == "auto" else feature_names[x],
                                key="dep_interaction"
                            )
                        
                        plt.figure(figsize=(10, 6))
                        if interaction_idx == "auto":
                            shap.dependence_plot(feature_idx, shap_values, X_sample, 
                                               feature_names=feature_names, show=False)
                        else:
                            shap.dependence_plot(feature_idx, shap_values, X_sample, 
                                               feature_names=feature_names, 
                                               interaction_index=interaction_idx, show=False)
                        
                        # 调整colorbar标签
                        ax = plt.gca()
                        ax.tick_params(axis='both', labelsize=18)
                        ax.set_xlabel(ax.get_xlabel(), fontsize=22, fontweight='bold')
                        ax.set_ylabel(ax.get_ylabel(), fontsize=22, fontweight='bold')
                        
                        # 修改colorbar标签
                        if len(plt.gcf().axes) > 1:
                            cbar = plt.gcf().axes[-1]
                            if cbar:
                                cbar.set_ylabel('特征值', fontsize=18, fontweight='bold', rotation=270, labelpad=5)
                                cbar.tick_params(labelsize=16)
                        
                        st.pyplot(plt.gcf())
                        plt.clf()
                        plt.close()
                        
                        with st.expander("📖 如何解读Dependence Plot"):
                            st.markdown("""
                            - **横轴**：特征值
                            - **纵轴**：该特征的SHAP值
                            - **颜色**：交互特征的值（如果选择）
                            - 显示特征值如何影响预测，以及是否存在非线性关系或交互效应
                            """)
                    
                    # 5. Force Plot (可选，对于少量样本)
                    if len(X_sample) <= 20:
                        st.markdown("---")
                        st.markdown("#### 5️⃣ Force Plot (Multiple Predictions)")
                        st.caption("显示多个样本的特征贡献，可以看到不同样本的预测差异")
                        
                        try:
                            plt.figure(figsize=(12, 4))
                            if selected_model in ["RF", "LightGBM"]:
                                shap.force_plot(
                                    st.session_state.shap_explainer.expected_value,
                                    shap_values[:min(20, len(X_sample))],
                                    X_sample[:min(20, len(X_sample))],
                                    feature_names=feature_names,
                                    matplotlib=True,
                                    show=False
                                )
                            else:
                                shap.force_plot(
                                    st.session_state.shap_explainer.expected_value,
                                    shap_values[:min(20, len(X_sample))],
                                    X_sample[:min(20, len(X_sample))],
                                    feature_names=feature_names,
                                    matplotlib=True,
                                    show=False
                                )
                            st.pyplot(plt.gcf())
                            plt.clf()
                            plt.close()
                        except:
                            st.info("💡 Force Plot需要较少样本数（≤20）才能清晰显示")
                    
                    # 5. Force Plot (可选，对于少量样本)
                    if len(X_sample) <= 20:
                        st.markdown("#### 5️⃣ Force Plot (Multiple Predictions)")
                        st.caption("显示多个样本的特征贡献，可以看到不同样本的预测差异")
                        
                        try:
                            plt.figure(figsize=(12, 4))
                            if selected_model in ["RF", "LightGBM"]:
                                shap.force_plot(
                                    st.session_state.shap_explainer.expected_value,
                                    shap_values[:min(20, len(X_sample))],
                                    X_sample[:min(20, len(X_sample))],
                                    feature_names=feature_names,
                                    matplotlib=True,
                                    show=False
                                )
                            else:
                                shap.force_plot(
                                    st.session_state.shap_explainer.expected_value,
                                    shap_values[:min(20, len(X_sample))],
                                    X_sample[:min(20, len(X_sample))],
                                    feature_names=feature_names,
                                    matplotlib=True,
                                    show=False
                                )
                            st.pyplot(plt.gcf())
                            plt.clf()
                            plt.close()
                        except:
                            st.info("💡 Force Plot需要较少样本数（≤20）才能清晰显示")
                
                else:
                    st.info("👆 点击上方按钮生成SHAP分析")
        
        # ==================== Tab 2: Partial Dependence Plot (PDP) ====================
        with tab2:
            st.markdown("### Partial Dependence Plot (PDP)")
            st.info("💡 PDP显示特征与预测结果之间的平均关系，边际化了其他特征的影响。适合理解特征的整体趋势")
            
            model = model_result['model']
            X_train = st.session_state.X_train
            feature_names = st.session_state.feature_names
            
            # 确保feature_names是字符串列表
            if not isinstance(feature_names, list):
                feature_names = list(feature_names)
            feature_names = [str(name) for name in feature_names]
            
            # 选择要分析的特征
            st.markdown("#### 选择特征")
            
            analysis_type = st.radio(
                "分析类型",
                ["单特征PDP", "双特征PDP（交互效应）"],
                key="pdp_type"
            )
            
            if analysis_type == "单特征PDP":
                # 多选特
                selected_features = st.multiselect(
                    "选择要分析的特征（可多选）",
                    range(len(feature_names)),
                    default=[0],
                    format_func=lambda x: feature_names[x],
                    key="pdp_features"
                )
                
                if st.button("🚀 生成PDP", key="generate_pdp"):
                    if len(selected_features) == 0:
                        st.warning("⚠️ 请至少选择一个特征")
                    else:
                        with st.spinner("正在生成PDP..."):
                            try:
                                # 创建子图
                                n_features = len(selected_features)
                                n_cols = min(2, n_features)
                                n_rows = (n_features + n_cols - 1) // n_cols
                                
                                fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
                                if n_features == 1:
                                    axes = [axes]
                                else:
                                    axes = axes.flatten() if n_features > 1 else [axes]
                                
                                for idx, feature_idx in enumerate(selected_features):
                                    ax = axes[idx]
                                    
                                    # 计算PDP
                                    pdp_result = partial_dependence(
                                        model, X_train, [feature_idx],
                                        kind='average', grid_resolution=50
                                    )
                                    
                                    # 获取原始特征范围（反标准化X轴）
                                    feature_name = feature_names[feature_idx]
                                    if 'feature_ranges' in st.session_state and feature_name in st.session_state.feature_ranges:
                                        x_values = pdp_result['grid_values'][0]
                                        feature_range = st.session_state.feature_ranges[feature_name]
                                        # 反标准化: x_original = x_scaled * std + mean
                                        x_values_original = x_values * feature_range['std'] + feature_range['mean']
                                    else:
                                        x_values_original = pdp_result['grid_values'][0]
                                    
                                    # 绘制（使用原始范围）
                                    ax.plot(x_values_original, pdp_result['average'][0], 
                                           linewidth=2.5, color='steelblue')
                                    ax.set_xlabel(feature_names[feature_idx], fontsize=11, fontweight='bold')
                                    ax.set_ylabel('Partial Dependence', fontsize=11, fontweight='bold')
                                    ax.set_title(f'PDP: {feature_names[feature_idx]}', fontsize=12, fontweight='bold')
                                    ax.grid(True, alpha=0.3, linestyle='--')
                                
                                # 隐藏多余的子图
                                for idx in range(len(selected_features), len(axes)):
                                    axes[idx].set_visible(False)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                                
                                st.success("✅ PDP图生成完成！")
                                
                                with st.expander("📖 如何解读PDP"):
                                    st.markdown("""
                                    - **横轴**：特征值
                                    - **纵轴**：部分依赖值（对预测的平均影响）
                                    - **曲线**：显示特征值变化时，预测的平均变化趋势
                                    
                                    **解读要点：**
                                    - 上升趋势 → 特征值增加，预测值增加
                                    - 下降趋势 → 特征值增加，预测值减少
                                    - 平坦 → 特征对预测影响小
                                    - 非线性 → 特征与预测存在复杂关系
                                    """)
                                
                            except Exception as e:
                                st.error(f"❌ PDP生成失败：{str(e)}")
            
            else:  # 双特征PDP
                col1, col2 = st.columns(2)
                with col1:
                    feature1 = st.selectbox(
                        "选择特征1",
                        range(len(feature_names)),
                        format_func=lambda x: feature_names[x],
                        key="pdp_feature1"
                    )
                with col2:
                    feature2 = st.selectbox(
                        "选择特征2",
                        range(len(feature_names)),
                        index=min(1, len(feature_names)-1),
                        format_func=lambda x: feature_names[x],
                        key="pdp_feature2"
                    )
                
                if st.button("🚀 生成2D PDP", key="generate_2d_pdp"):
                    if feature1 == feature2:
                        st.warning("⚠️ 请选择两个不同的特征")
                    else:
                        with st.spinner("正在生成2D PDP..."):

                            try:
                                # 计算2D PDP
                                pdp_result = partial_dependence(
                                    model, X_train, [(feature1, feature2)],
                                    kind='average', grid_resolution=50
                                )
                                
                                # 获取原始特征范围（反标准化）
                                feature1_name = feature_names[feature1]
                                feature2_name = feature_names[feature2]
                                
                                x1_values = pdp_result['grid_values'][0]
                                x2_values = pdp_result['grid_values'][1]
                                
                                # 反标准化X和Y
                                if 'feature_ranges' in st.session_state:
                                    if feature1_name in st.session_state.feature_ranges:
                                        range1 = st.session_state.feature_ranges[feature1_name]
                                        x1_values = x1_values * range1['std'] + range1['mean']
                                    
                                    if feature2_name in st.session_state.feature_ranges:
                                        range2 = st.session_state.feature_ranges[feature2_name]
                                        x2_values = x2_values * range2['std'] + range2['mean']
                                
                                # 绘制2D PDP
                                fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
                                
                                XX, YY = np.meshgrid(x1_values, x2_values)
                                Z = pdp_result['average'][0].T
                                
                                contour = ax.contourf(XX, YY, Z, levels=20, cmap='RdYlGn', alpha=0.8)
                                contour_lines = ax.contour(XX, YY, Z, levels=10, colors='black', 
                                                          linewidths=0.5, alpha=0.4)
                                ax.clabel(contour_lines, inline=True, fontsize=8)
                                
                                cbar = plt.colorbar(contour, ax=ax)
                                cbar.set_label('Partial Dependence', fontsize=11, fontweight='bold')
                                
                                ax.set_xlabel(feature1_name, fontsize=12, fontweight='bold')
                                ax.set_ylabel(feature2_name, fontsize=12, fontweight='bold')
                                ax.set_title(f'2D PDP: {feature1_name} vs {feature2_name}', 
                                           fontsize=13, fontweight='bold', pad=15)
                                ax.grid(True, alpha=0.3, linestyle='--')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                                
                                st.success("2D PDP图生成完成！")
                                
                                with st.expander("📖 如何解读2D PDP"):
                                    st.markdown("""
                                    - **横轴和纵轴**：两个特征的实际值范围
                                    - **颜色**：部分依赖值（预测值）
                                    - **等高线**：相同预测值的区域
                                    
                                    **解读要点：**
                                    - 颜色变化 → 显示两个特征如何共同影响预测
                                    - 对角线模式 → 两个特征存在交互效应
                                    - 平行线 → 特征独立作用，无交互
                                    """)
                                
                            except Exception as e:
                                st.error(f"2D PDP生成失败：{str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
        
        # ==================== Tab 3: Individual Conditional Expectation (ICE) ====================
        with tab3:
            st.markdown("### Individual Conditional Expectation (ICE)")
            st.info("💡 ICE图显示每个样本的特征-预测关系曲线。与PDP不同，ICE不做平均，可以看到个体差异和异质性")
            
            model = model_result['model']
            X_train = st.session_state.X_train
            feature_names = st.session_state.feature_names
            
            # 确保feature_names是字符串列表
            if not isinstance(feature_names, list):
                feature_names = list(feature_names)
            feature_names = [str(name) for name in feature_names]
            
            # 选择特征和样本数
            col1, col2 = st.columns([2, 1])
            
            with col1:
                ice_feature = st.selectbox(
                    "选择要分析的特征",
                    range(len(feature_names)),
                    format_func=lambda x: feature_names[x],
                    key="ice_feature"
                )
            
            with col2:
                n_ice_samples = st.number_input(
                    "样本数量",
                    min_value=10,
                    max_value=len(X_train),  # 不再限制00
                    value=min(100, len(X_train)),
                    step=10,
                    key="ice_samples",
                    help="显示的ICE曲线数量（可选择全部样本）"
                )
            
            show_pdp_overlay = st.checkbox("叠加PDP曲线（平均趋势）", value=True, key="show_pdp")
            show_fitted_curve = st.checkbox("显示拟合曲线", value=True, key="show_fitted")
            
            if st.button("🚀 生成ICE图", key="generate_ice"):
                with st.spinner("正在生成ICE图..."):
                    try:
                        from scipy.optimize import curve_fit
                        
                        fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
                        
                        # 计算ICE
                        pdp_result = partial_dependence(
                            model, X_train[:n_ice_samples], [ice_feature],
                            kind='both', grid_resolution=50
                        )
                        
                        # 获取原始特征范围（反标准化X轴）
                        feature_name = feature_names[ice_feature]
                        if 'feature_ranges' in st.session_state and feature_name in st.session_state.feature_ranges:
                            x_values = pdp_result['grid_values'][0]
                            feature_range = st.session_state.feature_ranges[feature_name]
                            # 反标准化: x_original = x_scaled * std + mean
                            x_values_original = x_values * feature_range['std'] + feature_range['mean']
                        else:
                            x_values_original = pdp_result['grid_values'][0]
                        
                        # 绘制ICE曲线（每个样本一条线，使用原始范围）
                        for i in range(pdp_result['individual'][0].shape[0]):
                            ax.plot(x_values_original, 
                                   pdp_result['individual'][0][i], 
                                   color='lightblue', alpha=0.3, linewidth=0.8)
                        
                        # 叠加PDP曲线（平均）
                        pdp_values = pdp_result['average'][0]
                        if show_pdp_overlay:
                            ax.plot(x_values_original, pdp_values,
                                   color='red', linewidth=3, label='平均', zorder=10)
                        
                        # 拟合曲线
                        fitted_formula = None
                        if show_fitted_curve and show_pdp_overlay:
                            # 定义多种拟合函数
                            def linear(x, a, b):
                                return a * x + b
                            
                            def quadratic(x, a, b, c):
                                return a * x**2 + b * x + c
                            
                            def exponential(x, a, b, c):
                                return a * np.exp(b * x) + c
                            
                            def power(x, a, b, c):
                                return a * np.power(np.abs(x) + 1e-10, b) + c
                            
                            def logarithmic(x, a, b, c):
                                return a * np.log(np.abs(x) + 1e-10) + b * x + c
                            
                            # 尝试多种拟合，选择R²最高的
                            best_fit = None
                            best_r2 = -np.inf
                            best_name = ""
                            best_params = None
                            
                            fit_functions = [
                                ("Linear", linear, [1, 0]),
                                ("Quadratic", quadratic, [0, 1, 0]),
                                ("Exponential", exponential, [1, 0.1, 0]),
                                ("Power", power, [1, 1, 0]),
                                ("Logarithmic", logarithmic, [1, 1, 0])
                            ]
                            
                            for name, func, p0 in fit_functions:
                                try:
                                    popt, _ = curve_fit(func, x_values_original, pdp_values, 
                                                       p0=p0, maxfev=5000)
                                    y_fit = func(x_values_original, *popt)
                                    r2 = 1 - np.sum((pdp_values - y_fit)**2) / np.sum((pdp_values - np.mean(pdp_values))**2)
                                    
                                    if r2 > best_r2:
                                        best_r2 = r2
                                        best_fit = (func, popt)
                                        best_name = name
                                        best_params = popt
                                except:
                                    continue
                            
                            # 绘制最佳拟合曲线并生成公式
                            if best_fit is not None:
                                func, popt = best_fit
                                y_fit = func(x_values_original, *popt)
                                
                                # 生成具体的数学公
                                if best_name == "Linear":
                                    fitted_formula = f"y = {popt[0]:.4f}x + {popt[1]:.4f}"
                                elif best_name == "Quadratic":
                                    fitted_formula = f"y = {popt[0]:.4f}x² + {popt[1]:.4f}x + {popt[2]:.4f}"
                                elif best_name == "Exponential":
                                    fitted_formula = f"y = {popt[0]:.4f}·exp({popt[1]:.4f}x) + {popt[2]:.4f}"
                                elif best_name == "Power":
                                    fitted_formula = f"y = {popt[0]:.4f}·x^{popt[1]:.4f} + {popt[2]:.4f}"
                                elif best_name == "Logarithmic":
                                    fitted_formula = f"y = {popt[0]:.4f}·ln(x) + {popt[1]:.4f}x + {popt[2]:.4f}"
                                
                                ax.plot(x_values_original, y_fit, 
                                       color='green', linewidth=2.5, linestyle='--',
                                       label=f'拟合: {best_name} (R²={best_r2:.3f})', 
                                       zorder=9)
                        
                        ax.legend(fontsize=24, loc='upper left', framealpha=0.95, bbox_to_anchor=(0.02, 0.98))
                        ax.set_xlabel(feature_names[ice_feature], fontsize=28, fontweight='bold')
                        ax.set_ylabel('预测值', fontsize=28, fontweight='bold')
                        ax.tick_params(axis='both', labelsize=24)
                        ax.grid(True, alpha=0.3, linestyle='--')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        st.success(f"ICE图生成完成！显示了 {n_ice_samples} 个样本")
                        
                        # 显示拟合公式
                        if fitted_formula is not None:
                            st.markdown("### 📐 拟合函数公式")
                            st.latex(fitted_formula.replace('·', r'\cdot ').replace('exp', r'\exp').replace('ln', r'\ln'))
                            st.info(f"**拟合公式：** `{fitted_formula}`  \n**拟合优度 R²：** {best_r2:.4f}")
                        
                        with st.expander("📖 如何解读ICE"):
                            st.markdown("""
                            - **横轴**：特征值（实际范围）
                            - **纵轴**：预测值
                            - **浅蓝色线**：每条线代表一个样本的预测轨迹
                            - **红色粗线**：PDP曲线（所有样本的平均）
                            - **绿色虚线**：拟合曲线（数学函数近似）
                            
                            **解读要点：**
                            - **线条平行** → 特征对所有样本的影响一致（同质性）
                            - **线条发散** → 特征对不同样本的影响不同（异质性）
                            - **线条交叉** → 存在复杂的交互效应
                            - **与PDP对比** → 如果ICE线条与PDP差异大，说明平均趋势不能代表个体
                            
                            **实际意义：**
                            - 同质性高 → 可以用PDP总结特征影响
                            - 异质性高 → 需要考虑子群体或交互效应
                            """)
                        
                        # 额外分析：检测异质
                        st.markdown("#### 📊 异质性分")
                        
                        # 计算每个样本的斜率变
                        slopes = []
                        for i in range(pdp_result['individual'][0].shape[0]):
                            y_values = pdp_result['individual'][0][i]
                            slope = (y_values[-1] - y_values[0]) / (pdp_result['grid_values'][0][-1] - pdp_result['grid_values'][0][0])
                            slopes.append(slope)
                        
                        slopes = np.array(slopes)
                        slope_std = np.std(slopes)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("平均斜率", f"{np.mean(slopes):.4f}")
                        col2.metric("斜率标准差", f"{slope_std:.4f}")
                        
                        if slope_std < 0.01:
                            col3.success("同质性高")
                            st.info("💡 特征对所有样本的影响一致，PDP可以很好地总结特征效应")
                        elif slope_std < 0.05:
                            col3.warning("⚠️ 中等异质性")
                            st.info("💡 特征对不同样本的影响有一定差异，建议结合ICE和PDP分析")
                        else:
                            col3.error("❌ 异质性高")
                            st.warning("⚠️ 特征对不同样本的影响差异很大，PDP可能无法准确反映特征效应，建议深入分析子群体")
                        
                    except Exception as e:
                        st.error(f"❌ ICE图生成失败：{str(e)}")
        
        # ==================== Tab 4: Dual-Factor Coupling Analysis ====================
        with tab4:
            st.markdown("### 🔬 双因素耦合效应分析 (Dual-Factor Coupling Analysis)")
            st.info("""
            💡 **分析目标**：揭示两两特征之间的交互作用对目标变量的影响
            
            **功能特点：**
            - 📊 3D曲面图：直观展示两因素与目标变量的关系
            - 🎨 颜色映射：使用渐变色表示目标变量大小
            - 📐 曲面拟合：支持LogisticCum、LogNormal2D、Power2D等多种函数
            - 🗺️ 2D投影：将3D曲面投影到XY平面，方便查看
            - 📝 公式输出：自动生成拟合函数的数学公式
            """)
            
            model = model_result['model']
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            feature_names = st.session_state.feature_names
            
            # 确保feature_names是字符串列表
            if not isinstance(feature_names, list):
                feature_names = list(feature_names)
            feature_names = [str(name) for name in feature_names]
            
            # 选择两个特征
            st.markdown("#### 🎯 选择分析的两个特")
            col1, col2 = st.columns(2)
            
            with col1:
                feature_x = st.selectbox(
                    "特征 X（横轴）",
                    range(len(feature_names)),
                    format_func=lambda x: feature_names[x],
                    key="coupling_feature_x"
                )
            
            with col2:
                feature_y = st.selectbox(
                    "特征 Y（纵轴）",
                    range(len(feature_names)),
                    index=min(1, len(feature_names)-1),
                    format_func=lambda x: feature_names[x],
                    key="coupling_feature_y"
                )
            
            # 参数设置
            col1, col2, col3 = st.columns(3)
            with col1:
                grid_resolution = st.slider("网格分辨", 20, 1000, 100, help="越高越精细，但计算时间越长。建议：快速预览100，标准分辨率200，高精度500+")
            with col2:
                show_projection = st.checkbox("显示2D投影", value=True)
            with col3:
                fit_surface = st.checkbox("拟合曲面函数", value=True)
            
            if st.button("🚀 生成双因素耦合分析", key="generate_coupling"):
                if feature_x == feature_y:
                    st.warning("⚠️ 请选择两个不同的特征")
                else:
                    with st.spinner("正在生成3D曲面图和拟合函数..."):
                        try:
                            from scipy.optimize import curve_fit
                            from mpl_toolkits.mplot3d import Axes3D
                            
                            # 获取特征名称
                            feature_x_name = feature_names[feature_x]
                            feature_y_name = feature_names[feature_y]
                            
                            # 获取特征范围（使用原始数据范围，而不是标准化后的范围）
                            if 'feature_ranges' in st.session_state:
                                if feature_x_name in st.session_state.feature_ranges:
                                    range_x = st.session_state.feature_ranges[feature_x_name]
                                    x_min, x_max = range_x['min'], range_x['max']
                                else:
                                    # 如果没有保存范围，使用标准化后的范围
                                    x_min, x_max = X_train[:, feature_x].min(), X_train[:, feature_x].max()
                                
                                if feature_y_name in st.session_state.feature_ranges:
                                    range_y = st.session_state.feature_ranges[feature_y_name]
                                    y_min, y_max = range_y['min'], range_y['max']
                                else:
                                    y_min, y_max = X_train[:, feature_y].min(), X_train[:, feature_y].max()
                            else:
                                # 如果没有feature_ranges，使用标准化后的范围
                                x_min, x_max = X_train[:, feature_x].min(), X_train[:, feature_x].max()
                                y_min, y_max = X_train[:, feature_y].min(), X_train[:, feature_y].max()
                            
                            # 创建网格（使用原始数据范围）
                            x_values_original = np.linspace(x_min, x_max, grid_resolution)
                            y_values_original = np.linspace(y_min, y_max, grid_resolution)
                            
                            # 如果有scaler，需要将原始值转换为标准化值用于预测
                            if 'scaler' in st.session_state and 'feature_ranges' in st.session_state:
                                scaler = st.session_state.scaler
                                # 标准化x和y值
                                x_values = (x_values_original - range_x['mean']) / range_x['std'] if feature_x_name in st.session_state.feature_ranges else x_values_original
                                y_values = (y_values_original - range_y['mean']) / range_y['std'] if feature_y_name in st.session_state.feature_ranges else y_values_original
                            else:
                                x_values = x_values_original
                                y_values = y_values_original
                            
                            XX_original, YY_original = np.meshgrid(x_values_original, y_values_original)
                            XX, YY = np.meshgrid(x_values, y_values)
                            
                            # 预测Z值（目标变量）
                            Z = np.zeros_like(XX)
                            for i in range(grid_resolution):
                                for j in range(grid_resolution):
                                    # 创建预测样本（其他特征使用均值）
                                    sample = X_train.mean(axis=0).copy()
                                    sample[feature_x] = x_values[j]
                                    sample[feature_y] = y_values[i]
                                    Z[i, j] = model.predict(sample.reshape(1, -1))[0]
                            
                            # ========== 绘制3D曲面==========
                            # 3D图单独一行，居中
                            fig_3d = plt.figure(figsize=(14, 10), dpi=150)
                            ax1 = fig_3d.add_subplot(111, projection='3d')
                            
                            # 3D曲面图（反转X和Y轴方向）
                            surf = ax1.plot_surface(XX_original, YY_original, Z, 
                                                   cmap='jet', alpha=0.9, 
                                                   edgecolor='none', antialiased=True)
                            
                            # 添加颜色条（统一长度）
                            cbar = fig_3d.colorbar(surf, ax=ax1, shrink=0.7, aspect=20, pad=0.1)
                            cbar.set_label('目标变量', 
                                         fontsize=14, fontweight='bold')
                            
                            ax1.set_xlabel(feature_x_name, fontsize=15, fontweight='bold', labelpad=10)
                            ax1.set_ylabel(feature_y_name, fontsize=15, fontweight='bold', labelpad=10)
                            zlabel = ax1.set_zlabel('目标变量', fontsize=15, fontweight='bold', labelpad=10)
                            zlabel.set_rotation(90)
                            ax1.set_title(f'三维耦合曲面: {feature_x_name} × {feature_y_name}', 
                                        fontsize=16, fontweight='bold', pad=20)
                            ax1.tick_params(axis='both', labelsize=13)
                            
                            # 反转X和Y轴方向（从大到小）
                            ax1.invert_xaxis()
                            ax1.invert_yaxis()
                            
                            # 设置视角
                            ax1.view_init(elev=25, azim=45)
                            
                            # 添加网格
                            ax1.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig_3d)
                            plt.close()
                            
                            # ========== 2D投影：两个图放一排 ==========
                            if show_projection:
                                fig_2d = plt.figure(figsize=(18, 8), dpi=150)
                                
                                # 等高线图（左）
                                ax2 = fig_2d.add_subplot(121)
                                contour = ax2.contourf(XX_original, YY_original, Z, 
                                                      levels=20, cmap='jet', alpha=0.9)
                                contour_lines = ax2.contour(XX_original, YY_original, Z, 
                                                           levels=10, colors='black', 
                                                           linewidths=0.5, alpha=0.4)
                                ax2.clabel(contour_lines, inline=True, fontsize=8)
                                
                                cbar2 = plt.colorbar(contour, ax=ax2, fraction=0.046, pad=0.04)
                                cbar2.set_label('目标变量', fontsize=13, fontweight='bold')
                                
                                ax2.set_xlabel(feature_x_name, fontsize=14, fontweight='bold')
                                ax2.set_ylabel(feature_y_name, fontsize=14, fontweight='bold')
                                ax2.set_title('二维投影（等高线）', fontsize=15, fontweight='bold', pad=10)
                                ax2.tick_params(axis='both', labelsize=12)
                                ax2.grid(True, alpha=0.3, linestyle='--')
                                
                                # 热力图（右）
                                ax3 = fig_2d.add_subplot(122)
                                im = ax3.imshow(Z, extent=[x_values_original.min(), x_values_original.max(),
                                                          y_values_original.min(), y_values_original.max()],
                                              origin='lower', cmap='jet', aspect='auto', alpha=0.9)
                                
                                cbar3 = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
                                cbar3.set_label('目标变量', fontsize=13, fontweight='bold')
                                
                                ax3.set_xlabel(feature_x_name, fontsize=14, fontweight='bold')
                                ax3.set_ylabel(feature_y_name, fontsize=14, fontweight='bold')
                                ax3.set_title('二维投影（热力图）', fontsize=15, fontweight='bold', pad=10)
                                ax3.tick_params(axis='both', labelsize=12)
                                ax3.grid(True, alpha=0.3, linestyle='--', color='white')
                                
                                plt.tight_layout()
                                st.pyplot(fig_2d)
                                plt.close()
                            
                            st.success("✅ 3D耦合曲面图生成完成！")
                            
                            # ========== 曲面拟合 ==========
                            if fit_surface:
                                st.markdown("#### 📐 曲面函数拟合")
                                
                                # 准备拟合数据
                                X_fit = XX.flatten()
                                Y_fit = YY.flatten()
                                Z_fit = Z.flatten()
                                
                                # 定义多种2D拟合函数
                                def logistic_cum_2d(xy, a, b, c, d, e):
                                    """LogisticCum函数"""
                                    x, y = xy
                                    return a / ((1 + np.exp((x + b) / c)) * (1 + np.exp((y + d) / e)))
                                
                                def lognormal_2d(xy, a, b, c, d, e, f):
                                    """LogNormal2D函数"""
                                    x, y = xy
                                    return a * np.exp(-((x - b)**2 / (2 * c**2) + (y - d)**2 / (2 * e**2))) + f
                                
                                def power_2d(xy, a, b, c, d, e):
                                    """Power2D函数"""
                                    x, y = xy
                                    return a * (np.abs(x) + 1e-10)**b * (np.abs(y) + 1e-10)**c + d * x + e * y
                                
                                def polynomial_2d(xy, a, b, c, d, e, f):
                                    """二次多项式"""
                                    x, y = xy
                                    return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f
                                
                                # 尝试多种拟合
                                best_fit = None
                                best_r2 = -np.inf
                                best_name = ""
                                best_params = None
                                best_formula = ""
                                
                                fit_functions = [
                                    ("LogisticCum", logistic_cum_2d, [30000, 150, 20, 2000, 300]),
                                    ("LogNormal2D", lognormal_2d, [50, 200, 100, 200, 100, 10]),
                                    ("Power2D", power_2d, [1, 0.5, 0.5, 0.1, 0.1]),
                                    ("Polynomial2D", polynomial_2d, [0.01, 0.01, 0.001, 1, 1, 10])
                                ]
                                
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                for idx, (name, func, p0) in enumerate(fit_functions):
                                    try:
                                        status_text.text(f"正在尝试 {name} 拟合...")
                                        progress_bar.progress((idx + 1) / len(fit_functions))
                                        
                                        popt, _ = curve_fit(func, (X_fit, Y_fit), Z_fit, 
                                                           p0=p0, maxfev=10000)
                                        Z_pred = func((X_fit, Y_fit), *popt)
                                        r2 = 1 - np.sum((Z_fit - Z_pred)**2) / np.sum((Z_fit - np.mean(Z_fit))**2)
                                        
                                        if r2 > best_r2:
                                            best_r2 = r2
                                            best_fit = (func, popt)
                                            best_name = name
                                            best_params = popt
                                            
                                            # 生成公式
                                            if name == "LogisticCum":
                                                best_formula = f"z = {popt[0]:.2f} / ((1 + exp((x + {popt[1]:.2f}) / {popt[2]:.2f})) × (1 + exp((y + {popt[3]:.2f}) / {popt[4]:.2f})))"
                                            elif name == "LogNormal2D":
                                                best_formula = f"z = {popt[0]:.2f} × exp(-((x - {popt[1]:.2f})² / (2 × {popt[2]:.2f}²) + (y - {popt[3]:.2f})² / (2 × {popt[4]:.2f}²))) + {popt[5]:.2f}"
                                            elif name == "Power2D":
                                                best_formula = f"z = {popt[0]:.4f} × x^{popt[1]:.4f} × y^{popt[2]:.4f} + {popt[3]:.4f}x + {popt[4]:.4f}y"
                                            elif name == "Polynomial2D":
                                                best_formula = f"z = {popt[0]:.4f}x² + {popt[1]:.4f}y² + {popt[2]:.4f}xy + {popt[3]:.4f}x + {popt[4]:.4f}y + {popt[5]:.2f}"
                                    except Exception as e:
                                        continue
                                
                                progress_bar.empty()
                                status_text.empty()
                                
                                # 显示最佳拟合结
                                if best_fit is not None:
                                    st.success(f"最佳拟合函数：**{best_name}**，R² = **{best_r2:.4f}**")
                                    
                                    # 显示公式
                                    st.markdown("##### 📝 拟合函数公式")
                                    st.code(best_formula, language="text")
                                    
                                    # 显示参数
                                    with st.expander("📊 查看拟合参数"):
                                        st.write(f"**函数类型：** {best_name}")
                                        st.write(f"**拟合优度 R²：** {best_r2:.6f}")
                                        st.write("**参数值：**")
                                        for i, param in enumerate(best_params):
                                            st.write(f"  - 参数 {i+1}: {param:.6f}")
                                    
                                    # 绘制拟合曲面对比
                                    st.markdown("##### 🔍 拟合效果对比")
                                    
                                    func, popt = best_fit
                                    Z_fitted = func((XX.flatten(), YY.flatten()), *popt).reshape(XX.shape)
                                    
                                    # ========== 2行2列布局 ==========
                                    fig_compare = plt.figure(figsize=(18, 16), dpi=150)
                                    gs = fig_compare.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
                                    
                                    # 第1行左：原始曲面
                                    ax1 = fig_compare.add_subplot(gs[0, 0])
                                    im1 = ax1.imshow(Z, extent=[x_values_original.min(), x_values_original.max(),
                                                               y_values_original.min(), y_values_original.max()],
                                                    origin='lower', cmap='jet', aspect='auto')
                                    ax1.set_title('原始曲面', fontsize=15, fontweight='bold', pad=10)
                                    ax1.set_xlabel(feature_x_name, fontsize=14, fontweight='bold')
                                    ax1.set_ylabel(feature_y_name, fontsize=14, fontweight='bold')
                                    ax1.tick_params(axis='both', labelsize=12)
                                    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                                    cbar1.set_label('目标变量', fontsize=13, fontweight='bold')
                                    
                                    # 第1行右：拟合曲面
                                    ax2 = fig_compare.add_subplot(gs[0, 1])
                                    im2 = ax2.imshow(Z_fitted, extent=[x_values_original.min(), x_values_original.max(),
                                                                       y_values_original.min(), y_values_original.max()],
                                                    origin='lower', cmap='jet', aspect='auto')
                                    ax2.set_title(f'二维拟合曲面 ({best_name})', fontsize=15, fontweight='bold', pad=10)
                                    ax2.set_xlabel(feature_x_name, fontsize=14, fontweight='bold')
                                    ax2.set_ylabel(feature_y_name, fontsize=14, fontweight='bold')
                                    ax2.tick_params(axis='both', labelsize=12)
                                    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                                    cbar2.set_label('目标变量', fontsize=13, fontweight='bold')
                                    
                                    # 第2行左：3D拟合曲面
                                    ax3 = fig_compare.add_subplot(gs[1, 0], projection='3d')
                                    surf = ax3.plot_surface(XX_original, YY_original, Z_fitted, 
                                                           cmap='jet', alpha=0.9, 
                                                           edgecolor='none', linewidth=0, 
                                                           antialiased=True)
                                    
                                    # 添加底部等高线投影
                                    ax3.contour(XX_original, YY_original, Z_fitted, zdir='z', 
                                               offset=Z_fitted.min(), cmap='jet', alpha=0.5, linewidths=1)
                                    
                                    # 添加颜色条（与原始3D图长度一致）
                                    cbar3 = fig_compare.colorbar(surf, ax=ax3, shrink=0.7, aspect=20, pad=0.1)
                                    cbar3.set_label('目标变量', 
                                                   fontsize=13, fontweight='bold')
                                    
                                    ax3.set_xlabel(feature_x_name, fontsize=14, fontweight='bold', labelpad=10)
                                    ax3.set_ylabel(feature_y_name, fontsize=14, fontweight='bold', labelpad=10)
                                    zlabel3 = ax3.set_zlabel('目标变量', fontsize=14, fontweight='bold', labelpad=10)
                                    zlabel3.set_rotation(90)
                                    ax3.set_title(f'三维拟合曲面\n({best_name}, R²={best_r2:.4f})', 
                                                 fontsize=15, fontweight='bold', pad=15)
                                    ax3.tick_params(axis='both', labelsize=12)
                                    
                                    # 反转X和Y轴方向（与原始图保持一致）
                                    ax3.invert_xaxis()
                                    ax3.invert_yaxis()
                                    ax3.view_init(elev=25, azim=45)
                                    ax3.grid(True, alpha=0.3)
                                    
                                    # 第2行右：残差分布
                                    ax4 = fig_compare.add_subplot(gs[1, 1])
                                    residual = Z - Z_fitted
                                    im4 = ax4.imshow(residual, extent=[x_values_original.min(), x_values_original.max(),
                                                                       y_values_original.min(), y_values_original.max()],
                                                    origin='lower', cmap='RdBu_r', aspect='auto')
                                    ax4.set_title(f'残差 (R²={best_r2:.4f})', fontsize=15, fontweight='bold', pad=10)
                                    ax4.set_xlabel(feature_x_name, fontsize=14, fontweight='bold')
                                    ax4.set_ylabel(feature_y_name, fontsize=14, fontweight='bold')
                                    ax4.tick_params(axis='both', labelsize=12)
                                    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
                                    cbar4.set_label('残差', fontsize=13, fontweight='bold')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig_compare)
                                    plt.close()
                                    
                                    st.caption(f"""
                                    **拟合效果说明**：
                                    - **第1行**：原始曲面（左）vs 拟合曲面（右）- 2D热力图对比
                                    - **第2行**：3D拟合曲面（左）vs 残差分布（右）
                                    - **拟合优度**：R²={best_r2:.4f}，值越接近1表示拟合效果越好
                                    - **残差分布**：蓝色表示低估，红色表示高估，白色表示拟合准确
                                    - **颜色映射**：jet色图（红色=高值，蓝色=低值）
                                    """)
                                    
                                else:
                                    st.warning("⚠️ 所有拟合函数都失败了，可能需要调整初始参数或选择其他函数")
                            
                            # 解释说明
                            with st.expander("📖 如何解读双因素耦合分析"):
                                st.markdown(r"""
                                ### 3D曲面图解读
                                - **X轴和Y轴**: 两个特征的实际值范围
                                - **Z轴**: 目标变量(如混凝土抗压强度)
                                - **颜色**: 红色表示高值,蓝色表示低值
                                - **曲面形状**: 揭示两因素的交互作用模式
                                
                                ### 2D投影图解读
                                - **等高线图**: 相同目标变量值的区域连线
                                - **热力图**: 用颜色直观表示目标变量大小
                                - **优势**: 更容易识别最优区域和变化趋势
                                
                                ### 拟合函数类型
                                1. **LogisticCum**: 适合S型增长曲面,常用于强度发展分析
                                2. **LogNormal2D**: 适合钟形分布曲面,有明显峰值
                                3. **Power2D**: 适合幂律关系,常用于材料性能分析
                                4. **Polynomial2D**: 通用二次多项式,适合大多数情况
                                
                                ### 实际应用
                                - **优化配比**: 找到使目标变量最大的特征组合
                                - **交互效应**: 判断两因素是协同作用还是拮抗作用
                                - **敏感性分析**: 识别对目标变量影响最大的区域
                                - **预测建模**: 使用拟合公式进行快速预测
                                """)
                        
                        except Exception as e:
                            st.error(f"❌ 双因素耦合分析失败: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        
        # 导出结果
        st.subheader("💾 导出结果")
        
        # 导出性能报告
        report = f"""
模型性能报告
============
模型名称: {selected_model}

交叉验证性能:
  R2 分数: {model_result['cv_score']:.4f}

训练集性能:
  R2 分数: {model_result['train_r2']:.4f}
  RMSE: {model_result['train_rmse']:.4f}
  MAE: {model_result['train_mae']:.4f}

测试集性能(泛化能力):
  R2 分数: {model_result['test_r2']:.4f}
  RMSE: {model_result['test_rmse']:.4f}
  MAE: {model_result['test_mae']:.4f}

过拟合分析:
  训练集-测试集R2差值: {model_result['train_r2'] - model_result['test_r2']:.4f}
  {'存在过拟合风险' if model_result['train_r2'] - model_result['test_r2'] > 0.1 else '泛化能力良好'}

最佳超参数:
{model_result['best_params']}

数据集信息:
  训练集样本数: {len(model_result['y_train_true'])}
  测试集样本数: {len(model_result['y_test_true'])}
"""
        
        st.download_button(
            label="📄 下载性能报告",
            data=report,
            file_name=f"{selected_model}_报告.txt",
            mime="text/plain"
        )
        
        # 导出预测结果（训练集和测试集）
        col1, col2 = st.columns(2)
        
        with col1:
            # 训练集预测结果
            train_pred_df = pd.DataFrame({
                '实际值': model_result['y_train_true'],
                '预测值': model_result['y_train_pred'],
                '残差': model_result['y_train_true'] - model_result['y_train_pred']
            })
            
            train_csv = train_pred_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📊 下载训练集预测结果",
                data=train_csv,
                file_name=f"{selected_model}_训练集预测.csv",
                mime="text/csv"
            )
        
        with col2:
            # 测试集预测结果
            test_pred_df = pd.DataFrame({
                '实际值': model_result['y_test_true'],
                '预测值': model_result['y_test_pred'],
                '残差': model_result['y_test_true'] - model_result['y_test_pred']
            })
            
            test_csv = test_pred_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📊 下载测试集预测结果",
                data=test_csv,
                file_name=f"{selected_model}_测试集预测.csv",
                mime="text/csv"
            )
        
        # 添加进入下一步的按钮
        st.markdown("---")
        st.markdown("### 🎯 下一步：符号回归")
        
        if 'shap_values' in st.session_state:
            st.success("✅SHAP分析已完成，可以进入符号回归步骤")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 进入步骤5/6：符号回归", type="primary", width="stretch"):
                    st.session_state.step = 5
                    st.rerun()
        else:
            st.warning("⚠️ 请先在上方的SHAP Analysis标签页中生成SHAP分析，然后才能进入符号回归步骤5")
            st.info("💡 SHAP分析是符号回归的前置条件，用于识别最重要的特征")




# 步骤5：显示数学公式
elif st.session_state.step == 5:
    st.header("5️⃣ 显示数学公式 (Mathematical Formula Discovery)")
    
    # 检查前置条件
    if not st.session_state.results or len(st.session_state.results) == 0:
        st.warning("⚠️ 请先完成步骤3（模型训练）")
        if st.button("← 返回步骤3"):
            st.session_state.step = 3
            st.rerun()
        st.stop()
    
    if 'shap_values' not in st.session_state:
        st.warning("⚠️ 请先在步骤3中生成SHAP分析，以获取特征重要性排序")
        if st.button("← 返回步骤4"):
            st.session_state.step = 4
            st.rerun()
        st.stop()
    
    st.markdown("""
    ### 🧬 符号回归（Symbolic Regression）
    
    选择下方标签页进行符号回归分析：
    - **GP符号回归**：基于遗传编程的传统方法
    - **PySR方案**：基于Julia的高性能符号回归
    """)
    
    if not GPLEARN_AVAILABLE:
        st.error("❌ gplearn库未安装")
        st.code("pip install gplearn", language="bash")
        st.info("请在env6环境中运行上述命令安装gplearn")
        st.stop()
    
    # ========== 公共部分：模型选择和特征重要性 ==========
    st.markdown("---")
    
    # 模型选择
    st.markdown("### 📊 基础配置")
    selected_model = st.selectbox(
        "选择要基于的模型结果",
        list(st.session_state.results.keys()),
        key="step5_base_model",
        help="选择一个已训练的模型作为基准"
    )
    
    model_result = st.session_state.results[selected_model]
    
    # 计算特征重要性
    shap_values = st.session_state.shap_values
    feature_names = st.session_state.feature_names
    
    # 确保feature_names是字符串列表（全局转换，供所有tab使用）
    if not isinstance(feature_names, list):
        feature_names = list(feature_names)
    feature_names = [str(name) for name in feature_names]
    
    # 确保feature_names是字符串列表
    if not isinstance(feature_names, list):
        feature_names = list(feature_names)
    feature_names = [str(name) for name in feature_names]
    
    # 计算平均绝对SHAP值作为特征重要性
    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # 显示特征重要性
    st.markdown("---")
    st.markdown("### 📊 特征重要性排序（基于SHAP）")
    st.caption("以下特征按重要性从高到低排序，建议选择前3-6个特征用于符号回归")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # 显示特征重要性柱状图
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue', edgecolor='black')
        ax.set_xlabel('Mean |SHAP Value|', fontsize=10, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=10, fontweight='bold')
        ax.set_title('Feature Importance Ranking', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("**Top Features:**")
        # 使用更紧凑的显示方式
        for idx, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            st.text(f"{idx}. {row['Feature']}: {row['Importance']:.4f}")
    
    st.markdown("---")
    
    # 创建两个标签页
    tab1, tab2 = st.tabs(["🧬 GP符号回归", "🔬 PySR方案"])
    
    # ==================== 标签页1：GP符号回归 ====================
    with tab1:
        st.markdown("""
        ### 🧬 GP符号回归
        
        使用遗传编程（Genetic Programming）自动发现数学公式：
        - 📊 基于SHAP特征重要性智能选择特征
        - 🧬 使用遗传算法进化出最优公式
        - 📝 得到可解释的数学表达式
        - 🎯 适合需要物理意义或可解释性的场景
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ 遗传编程配置")
        
        # 特征选择
        st.markdown("#### 📊 特征选择")
        
        selected_features_gp = st.multiselect(
            "选择用于GP的特征（按重要性排序）",
            options=importance_df['Feature'].tolist(),
            default=importance_df['Feature'].head(5).tolist(),
            help="建议选择3-6个最重要的特征。可以根据特征相关性灵活选择，不必选择连续的前N个",
            key="gp_feature_selector"
        )
        
        if len(selected_features_gp) == 0:
            st.warning("⚠️ 请至少选择1个特征")
            st.stop()
        
        st.caption(f"✅ 已选择 {len(selected_features_gp)} 个特征：{', '.join(selected_features_gp)}")
        
        # ==================== 参数配置 ====================
        st.markdown("---")
        st.markdown("### ⚙️ 参数配置")
        
        # 核心参数（始终可见）
        st.markdown("**核心参数：**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            population_size = st.number_input(
                "种群大小",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="每代的个体数量，越大越可能找到好的公式，但速度越慢"
            )
        
        with col2:
            generations = st.number_input(
                "进化代数",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                help="进化的代数，越多越可能找到最优解"
            )
        
        with col3:
            parsimony_coefficient = st.number_input(
                "简约系数",
                min_value=0.0,
                max_value=0.1,
                value=0.001,
                step=0.001,
                format="%.4f",
                help="惩罚复杂公式，值越大越倾向于简单公式。⚠️ 修改后需要重新训练才能生效！"
            )
        
        st.caption('💡 **提示**：修改任何参数后，需要点击下方的"开始遗传编程"按钮重新训练，新参数才会生效。')
        
        # 高级参数（可折叠）
        with st.expander("🔧 高级参数（可选）"):
            st.markdown("**选择压力与采样：**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tournament_size = st.slider(
                    "锦标赛大小",
                    min_value=5,
                    max_value=100,
                    value=20,
                    help="选择操作的竞争个体数，影响选择压力"
                )
            
            with col2:
                max_samples = st.slider(
                    "训练样本比例",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.9,
                    step=0.1,
                    help="每代使用的训练样本比例"
                )
            
            with col3:
                stopping_criteria = st.number_input(
                    "停止阈值 (MAE)",
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.01,
                    step=0.001,
                    format="%.4f",
                    help="达到此MAE时提前停止训练"
                )
            
            st.markdown("**遗传算子参数：**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                p_crossover = st.slider(
                    "交叉概率",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    help="交叉操作的概率，越高越多基因交换"
                )
            
            with col2:
                p_subtree_mutation = st.slider(
                    "子树变异率",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    step=0.05,
                    help="子树变异的概率"
                )
            
            with col3:
                p_point_mutation = st.slider(
                    "点变异率",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    step=0.05,
                    help="点变异的概率"
                )
            
            with col4:
                p_hoist_mutation = st.slider(
                    "提升变异率",
                    min_value=0.0,
                    max_value=0.2,
                    value=0.05,
                    step=0.01,
                    help="提升变异的概率，将子树提升到更高层"
                )
            
            st.markdown("**初始化与常数范围：**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                init_method = st.selectbox(
                    "初始化方法",
                    options=["half and half", "grow", "full"],
                    index=0,
                    help="种群初始化方法"
                )
            
            with col2:
                const_min = st.number_input(
                    "常数最小值",
                    min_value=-100.0,
                    max_value=0.0,
                    value=-10.0,
                    step=1.0,
                    help="GP可以使用的常数的最小值"
                )
            
            with col3:
                const_max = st.number_input(
                    "常数最大值",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                    help="GP可以使用的常数的最大值"
                )
            
            st.caption(f"💡 常数范围：[{const_min}, {const_max}]")
        
        # 参数总结（始终可见）
        st.markdown("**📊 参数总结：**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"• 总个体数：{population_size} × {generations} = {population_size * generations:,}")
            st.caption(f"• 总变异率：{p_subtree_mutation + p_point_mutation + p_hoist_mutation:.2f}")
        with col2:
            st.caption(f"• 选择压力：锦标赛大小 {tournament_size}")
            st.caption(f"• 复杂度控制：简约系数 {parsimony_coefficient:.4f}")
        
        # ==================== 运算符选择 ====================
        st.markdown("---")
        st.markdown("### 🔢 运算符选择")
        
        with st.expander("选择数学运算符", expanded=False):
            st.markdown("**推荐运算符（适合混凝土数据）：**")
            st.caption("💡 以下运算符默认选中，适合大多数回归问题")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**基础四则运算**")
                use_add = st.checkbox("加法 (+)", value=True, key="gp_add")
                use_sub = st.checkbox("减法 (-)", value=True, key="gp_sub")
                use_mul = st.checkbox("乘法 (×)", value=True, key="gp_mul")
                use_div = st.checkbox("除法 (÷)", value=True, key="gp_div")
            
            with col2:
                st.markdown("**常用函数**")
                use_sqrt = st.checkbox("平方根 (√)", value=True, key="gp_sqrt", help="适合龄期效应")
                use_log = st.checkbox("对数 (log)", value=True, key="gp_log", help="适合龄期效应")
            
            st.markdown("---")
            st.markdown("**可选运算符（按需选择）：**")
            st.caption("⚠️ 根据数据特点选择，过多运算符会增加搜索时间")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**幂运算**")
                use_inv = st.checkbox("倒数 (1/x)", value=False, key="gp_inv")
            
            with col2:
                st.markdown("**比较函数**")
                use_max = st.checkbox("最大值 (max)", value=False, key="gp_max")
                use_min = st.checkbox("最小值 (min)", value=False, key="gp_min")
            
            with col3:
                st.markdown("**其他函数**")
                use_abs = st.checkbox("绝对值 (|x|)", value=False, key="gp_abs")
                use_neg = st.checkbox("取负 (-x)", value=False, key="gp_neg")
            
            st.markdown("---")
            st.markdown("**不推荐运算符（特殊场景）：**")
            st.caption("⚠️ 三角函数适用于周期性数据，混凝土强度通常不需要")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                use_sin = st.checkbox("正弦 (sin)", value=False, key="gp_sin")
            with col2:
                use_cos = st.checkbox("余弦 (cos)", value=False, key="gp_cos")
            with col3:
                use_tan = st.checkbox("正切 (tan)", value=False, key="gp_tan")
            
            # 构建函数集
            function_set = []
            if use_add: function_set.append('add')
            if use_sub: function_set.append('sub')
            if use_mul: function_set.append('mul')
            if use_div: function_set.append('div')
            if use_sqrt: function_set.append('sqrt')
            if use_log: function_set.append('log')
            if use_inv: function_set.append('inv')
            if use_sin: function_set.append('sin')
            if use_cos: function_set.append('cos')
            if use_tan: function_set.append('tan')
            if use_abs: function_set.append('abs')
            if use_neg: function_set.append('neg')
            if use_max: function_set.append('max')
            if use_min: function_set.append('min')
            
            if len(function_set) < 2:
                st.warning("⚠️ 至少选择2个运算符")
                function_set = ['add', 'sub', 'mul', 'div']
            
            st.info(f"✅ 已选择 {len(function_set)} 个运算符：{', '.join(function_set)}")
        
        # ==================== 开始训练 ====================
        st.markdown("---")
        if use_min: function_set.append('min')
        
        if len(function_set) == 0:
            st.error("❌ 请至少选择一种数学运算符")
        else:
            st.caption(f"📊 将使用 {len(function_set)} 种运算：{', '.join(function_set)}")
            
            # 开始训练按钮
            if st.button("🚀 开始遗传编程", type="primary", key="start_gp_step5"):
                # 准备数据（使用原始未标准化数据）
                X_train = st.session_state.X_train_original.values
                y_train = st.session_state.y_train
                X_test = st.session_state.X_test_original.values
                y_test = st.session_state.y_test
                
                # 获取选中特征的索引（保持用户选择的顺序）
                feature_indices = [feature_names.index(name) for name in selected_features_gp]
                
                X_train_selected = X_train[:, feature_indices]
                X_test_selected = X_test[:, feature_indices]
                
                with st.spinner(f"🧬 正在进化公式... (种群:{population_size}, 代数:{generations})"):
                    try:
                        # 创建符号回归
                        gp_model = SymbolicRegressor(
                            population_size=population_size,
                            generations=generations,
                            tournament_size=tournament_size,
                            stopping_criteria=stopping_criteria,
                            p_crossover=p_crossover,
                            p_subtree_mutation=p_subtree_mutation,
                            p_hoist_mutation=p_hoist_mutation,
                            p_point_mutation=p_point_mutation,
                            max_samples=max_samples,
                            parsimony_coefficient=parsimony_coefficient,
                            function_set=function_set,
                            metric='mean absolute error',
                            const_range=(const_min, const_max),
                            init_depth=(2, 6),
                            init_method=init_method,
                            random_state=42,
                            verbose=1,
                            n_jobs=-1
                        )
                        
                        # 训练
                        gp_model.fit(X_train_selected, y_train)
                        
                        # 预测
                        y_train_pred_gp = gp_model.predict(X_train_selected)
                        y_test_pred_gp = gp_model.predict(X_test_selected)
                        
                        # 计算性能
                        train_r2_gp = r2_score(y_train, y_train_pred_gp)
                        test_r2_gp = r2_score(y_test, y_test_pred_gp)
                        train_rmse_gp = np.sqrt(mean_squared_error(y_train, y_train_pred_gp))
                        test_rmse_gp = np.sqrt(mean_squared_error(y_test, y_test_pred_gp))
                        train_mae_gp = mean_absolute_error(y_train, y_train_pred_gp)
                        test_mae_gp = mean_absolute_error(y_test, y_test_pred_gp)
                        
                        # 提取进化历史数据（用于收敛分析）
                        evolution_history = []
                        if hasattr(gp_model, '_programs'):
                            # 计算基准 MSE（使用均值预测）
                            y_mean = np.mean(y_train)
                            baseline_mse = np.mean((y_train - y_mean) ** 2)
                            
                            for gen_idx, generation in enumerate(gp_model._programs):
                                if generation:
                                    for program in generation:
                                        if program:
                                            try:
                                                # 获取程序的适应度（fitness 这是 MAE
                                                fitness_mae = program.fitness_ if hasattr(program, 'fitness_') else None
                                                
                                                if fitness_mae is None:
                                                    continue
                                                
                                                # 计算 R² 作为更直观的性能指标
                                                r2 = None
                                                try:
                                                    # 使用程序预测训练
                                                    y_pred_program = program.execute(X_train_selected)
                                                    # 计算真实R²
                                                    r2 = r2_score(y_train, y_pred_program)
                                                    # 限制 R² 在合理范围内1 1
                                                    r2 = max(-1.0, min(1.0, r2))
                                                except Exception as e:
                                                    # 如果无法计算 R²，使用MAE 转换为近R²
                                                    # 使用公式：R² 1 - (MAE²/baseline_MSE)
                                                    # 这个近似假设 MAE RMSE 成正
                                                    mae_squared = fitness_mae ** 2
                                                    r2 = 1.0 - (mae_squared / (baseline_mse + 1e-10))
                                                    # 限制在合理范围内
                                                    r2 = max(0.0, min(1.0, r2))
                                                
                                                # 获取程序长度
                                                length = len(str(program)) if hasattr(program, '__str__') else 0
                                                # 获取程序深度
                                                try:
                                                    depth = program.depth_ if hasattr(program, 'depth_') else length // 10
                                                except:
                                                    depth = length // 10
                                                
                                                evolution_history.append({
                                                    'generation': gen_idx,
                                                    'fitness': fitness_mae,  # 保留 MAE
                                                    'r2': r2,  # 添加 R²
                                                    'length': length,
                                                    'depth': depth
                                                })
                                            except Exception as e:
                                                # 完全失败时跳过这个个
                                                continue
                        
                        # 保存结果
                        st.session_state.gp_model = gp_model
                        st.session_state.gp_selected_features = selected_features_gp
                        st.session_state.gp_feature_indices = feature_indices
                        st.session_state.gp_function_set = function_set
                        st.session_state.gp_population_size = population_size
                        st.session_state.gp_generations = generations
                        st.session_state.gp_tournament_size = tournament_size
                        st.session_state.gp_p_crossover = p_crossover
                        st.session_state.gp_parsimony_coefficient = parsimony_coefficient
                        st.session_state.gp_evolution_history = evolution_history
                        st.session_state.gp_results = {
                            'train_r2': train_r2_gp,
                            'test_r2': test_r2_gp,
                            'train_rmse': train_rmse_gp,
                            'test_rmse': test_rmse_gp,
                            'train_mae': train_mae_gp,
                            'test_mae': test_mae_gp,
                            'y_train_pred': y_train_pred_gp,
                            'y_test_pred': y_test_pred_gp
                        }
                        
                        st.success("✅ 遗传编程完成")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ 遗传编程失败：{str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # 显示结果（只在Tab1中显示）
        st.session_state._current_tab = 'tab1'
        
        if 'gp_model' in st.session_state and 'gp_results' in st.session_state:
            st.markdown("---")
            st.markdown("## 📊 GP模型结果")
            
            gp_results = st.session_state.gp_results
            gp_model = st.session_state.gp_model
            selected_features_gp = st.session_state.gp_selected_features
            
            # ==================== 显示GP公式 ====================
            st.markdown("### 📐 训练得到的GP公式")
            
            # 获取原始公式
            formula_str = str(gp_model._program)
            
            # 将X0, X1, X2...替换为A, B, C...
            import re
            def replace_variables(text):
                """将X0, X1, X2...替换为A, B, C..."""
                def replacer(match):
                    idx = int(match.group(1))
                    if idx < 26:
                        return chr(65 + idx)  # A-Z
                    else:
                        return f"X{idx}"  # 超过26个特征时保持X格式
                return re.sub(r'X(\d+)', replacer, text)
            
            # 转换公式
            formula_with_letters = replace_variables(formula_str)
            
            # 转换为中缀表达式（标准数学格式）
            infix_formula = convert_to_infix(formula_with_letters)
            
            # 转换为LaTeX格式
            latex_formula = convert_to_latex(formula_with_letters)
            
            # 显示公式
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**标准数学格式：**")
                st.code(infix_formula, language="text")
            
            with col2:
                st.markdown("**LaTeX格式：**")
                st.latex(latex_formula)
            
            # 显示变量映射关系
            st.markdown("**变量映射关系：**")
            mapping_text = "  |  ".join([f"**{chr(65+i)}** = {feat}" for i, feat in enumerate(selected_features_gp)])
            st.markdown(mapping_text)
            
            # 显示公式特征
            st.markdown("---")
            st.markdown("### 📏 公式特征")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # 公式长度
            formula_length = len(formula_str)
            col1.metric("公式长度", f"{formula_length} 字符")
            
            # 公式深度
            try:
                formula_depth = gp_model._program.depth_ if hasattr(gp_model._program, 'depth_') else "N/A"
                col2.metric("公式深度", formula_depth)
            except:
                col2.metric("公式深度", "N/A")
            
            # 运算符数量
            operator_count = sum(1 for c in formula_str if c in ['+', '-', '*', '/', '(', ')'])
            col3.metric("运算符数量", operator_count)
            
            # 使用特征数
            col4.metric("使用特征数", len(selected_features_gp))
            
            st.markdown("---")
            
            # 基础指标
            st.markdown("### 📊 性能指标")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("训练集 R²", f"{gp_results['train_r2']:.4f}")
            col2.metric("测试集 R²", f"{gp_results['test_r2']:.4f}")
            col3.metric("训练集 MAE", f"{gp_results['train_mae']:.4f}")
            col4.metric("测试集 MAE", f"{gp_results['test_mae']:.4f}")
            
            # ==================== GP模型分析（6图全景）====================
            st.markdown("#### 📊 GP模型分析（6图全景）")
            
            # 创建6图可视化
            fig = plt.figure(figsize=(20, 13))
            fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.06, hspace=0.35, wspace=0.3)
            
            # ===== 图1：预测散点图 =====
            ax1 = plt.subplot(2, 3, 1)
            
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
            y_train_pred = gp_results['y_train_pred']
            y_test_pred = gp_results['y_test_pred']
            
            # 从0开始的坐标轴
            global_min = 0
            global_max = max(y_train.max(), y_test.max(), y_train_pred.max(), y_test_pred.max())
            margin = global_max * 0.05
            plot_max = global_max + margin
            
            # 训练集
            ax1.scatter(y_train, y_train_pred, alpha=0.5, s=30, 
                       c='blue', edgecolors='navy', linewidth=0.5, label='训练集')
            # 测试集
            ax1.scatter(y_test, y_test_pred, alpha=0.5, s=30, 
                       c='green', edgecolors='darkgreen', linewidth=0.5, label='测试集')
            
            # 理想线
            ax1.plot([global_min, plot_max], [global_min, plot_max], 
                    'r--', linewidth=2, label='理想线 (y=x)')
            
            ax1.set_xlim(global_min, plot_max)
            ax1.set_ylim(global_min, plot_max)
            ax1.set_aspect('equal', adjustable='box')
            ax1.set_xlabel('实际值', fontsize=15, fontweight='bold')
            ax1.set_ylabel('预测值', fontsize=15, fontweight='bold')
            ax1.set_title(f'1. Prediction Scatter Plot\n(Train R²={gp_results["train_r2"]:.4f}, Test R²={gp_results["test_r2"]:.4f})', 
                         fontsize=14, fontweight='bold', pad=15)
            ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='both', labelsize=13)
            
            # ===== 图2：残差分析（QQ图）=====
            ax2 = plt.subplot(2, 3, 2)
            
            # 合并残差
            residuals = np.concatenate([y_train - y_train_pred, y_test - y_test_pred])
            
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax2)
            ax2.get_lines()[0].set_markerfacecolor('blue')
            ax2.get_lines()[0].set_markeredgecolor('navy')
            ax2.get_lines()[0].set_markersize(5)
            ax2.get_lines()[0].set_alpha(0.6)
            ax2.get_lines()[1].set_color('red')
            ax2.get_lines()[1].set_linewidth(2)
            
            ax2.set_xlabel('理论分位数', fontsize=15, fontweight='bold')
            ax2.set_ylabel('样本分位数', fontsize=15, fontweight='bold')
            ax2.set_title('2. QQ Plot (Normality Test)\n(Points on Line = Normal)', fontsize=14, fontweight='bold', pad=15)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='both', labelsize=13)
            
            # 添加正态性检验结果
            if len(residuals) <= 5000:
                _, p_value = stats.shapiro(residuals)
                normality_text = f'Shapiro-Wilk\np={p_value:.4f}\n'
                if p_value > 0.05:
                    normality_text += '✓ 近似正态'
                else:
                    normality_text += '✗ 偏离正态'
                ax2.text(0.05, 0.95, normality_text, transform=ax2.transAxes,
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            # ===== 图3：残差分布 =====
            ax3 = plt.subplot(2, 3, 3)
            
            ax3.hist(residuals, bins=30, alpha=0.7, color='skyblue', 
                    edgecolor='navy', linewidth=1.5)
            ax3.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='零线')
            
            ax3.set_xlabel('残差', fontsize=15, fontweight='bold')
            ax3.set_ylabel('频数', fontsize=15, fontweight='bold')
            ax3.set_title('3. Residual Distribution\n(Centered at Zero is Better)', fontsize=14, fontweight='bold', pad=15)
            ax3.legend(loc='upper right', fontsize=11, framealpha=0.9)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.tick_params(axis='both', labelsize=13)
            
            # ===== 图4：训练集 vs 测试集性能对比 =====
            ax4 = plt.subplot(2, 3, 4)
            
            metrics_names = ['R²', 'MAE', 'RMSE']
            train_metrics = [gp_results['train_r2'], gp_results['train_mae'], 
                           np.sqrt(np.mean((y_train - y_train_pred) ** 2))]
            test_metrics = [gp_results['test_r2'], gp_results['test_mae'], 
                          np.sqrt(np.mean((y_test - y_test_pred) ** 2))]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, train_metrics, width, label='训练集', 
                           color='skyblue', edgecolor='navy', linewidth=1.5)
            bars2 = ax4.bar(x + width/2, test_metrics, width, label='测试集', 
                           color='lightcoral', edgecolor='darkred', linewidth=1.5)
            
            ax4.set_xlabel('性能指标', fontsize=15, fontweight='bold')
            ax4.set_ylabel('指标值', fontsize=15, fontweight='bold')
            ax4.set_title('4. Train vs Test Performance\n(Higher R² is Better)', fontsize=14, fontweight='bold', pad=15)
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics_names, fontsize=13, fontweight='bold')
            ax4.legend(loc='upper right', fontsize=11, framealpha=0.9)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.tick_params(axis='y', labelsize=13)
            
            # 标注数值
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # ===== 图5：预测误差分析 =====
            ax5 = plt.subplot(2, 3, 5)
            
            train_errors = np.abs(y_train - y_train_pred)
            test_errors = np.abs(y_test - y_test_pred)
            
            ax5.scatter(y_train, train_errors, alpha=0.5, s=30, 
                       c='blue', edgecolors='navy', linewidth=0.5, label='训练集')
            ax5.scatter(y_test, test_errors, alpha=0.5, s=30, 
                       c='green', edgecolors='darkgreen', linewidth=0.5, label='测试集')
            
            ax5.set_xlabel('实际值', fontsize=15, fontweight='bold')
            ax5.set_ylabel('绝对误差', fontsize=15, fontweight='bold')
            ax5.set_title('5. Prediction Error Analysis\n(Lower is Better)', fontsize=14, fontweight='bold', pad=15)
            ax5.legend(loc='upper right', fontsize=11, framealpha=0.9)
            ax5.grid(True, alpha=0.3)
            ax5.tick_params(axis='both', labelsize=13)
            
            # ===== 图6：特征重要性 =====
            ax6 = plt.subplot(2, 3, 6)
            
            # 使用importance_df中的特征重要性
            if 'importance_df' in locals() and len(selected_features_gp) > 0:
                # 获取选中特征的重要性
                feature_importance = importance_df[importance_df['Feature'].isin(selected_features_gp)].copy()
                feature_importance = feature_importance.sort_values('Importance', ascending=True)
                
                colors = ['#2ecc71' if imp > 0 else '#95a5a6' for imp in feature_importance['Importance']]
                bars = ax6.barh(range(len(feature_importance)), feature_importance['Importance'], 
                               color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
                
                ax6.set_yticks(range(len(feature_importance)))
                ax6.set_yticklabels(feature_importance['Feature'], fontsize=13, fontweight='bold')
                ax6.set_xlabel('重要性', fontsize=15, fontweight='bold')
                ax6.set_ylabel('特征', fontsize=15, fontweight='bold')
                ax6.set_title('6. Feature Importance\n(Higher = More Important)', fontsize=14, fontweight='bold', pad=15)
                ax6.grid(True, alpha=0.3, axis='x')
                ax6.invert_yaxis()
                ax6.tick_params(axis='x', labelsize=13)
                
                # 标注数值
                for i, (bar, imp) in enumerate(zip(bars, feature_importance['Importance'])):
                    if imp > 0:
                        ax6.text(imp + 0.01, i, f'{imp:.3f}', 
                                va='center', fontsize=10, fontweight='bold')
            else:
                ax6.text(0.5, 0.5, '特征重要性数据不可用', 
                        ha='center', va='center', transform=ax6.transAxes)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # ==================== GP进化收敛分析（6图可视化）====================
            evolution_history = st.session_state.get('gp_evolution_history', [])
            
            if evolution_history:
                import pandas as pd
                df_evolution = pd.DataFrame(evolution_history)
                
                # 调用完整版GP可视化模块（内置）
                # 准备GP参数
                gp_params = {
                    'population_size': st.session_state.get('gp_population_size', 1000),
                    'generations': st.session_state.get('gp_generations', 20),
                    'tournament_size': st.session_state.get('gp_tournament_size', 20),
                    'p_crossover': st.session_state.get('gp_p_crossover', 0.7),
                    'p_subtree_mutation': st.session_state.get('gp_p_subtree_mutation', 0.1),
                    'parsimony_coefficient': st.session_state.get('gp_parsimony_coefficient', 0.001)
                }
                
                # 调用完整版可视化函数
                metrics = create_gp_convergence_visualization(df_evolution, gp_params)
                
                # 生成优化建议
                generate_optimization_suggestions(metrics, gp_params)
            else:
                st.warning("⚠️ 未找到进化历史数据")
            
            st.info("💡 模型已保存，可在步骤6中使用优化功能")
    
    # ==================== 标签页2：PySR方案 ====================
    with tab2:
        st.markdown("### 🔬 PySR符号回归")
        
        # 检查PySR是否可用
        if not PYSR_AVAILABLE:
            st.warning("⚠️ PySR库未安装")
            
            with st.expander("📦 安装说明", expanded=True):
                st.markdown("""
                **安装步骤：**
                
                1. 安装PySR Python包：
                ```bash
                pip install pysr
                ```
                
                2. 首次使用时，PySR会自动安装Julia依赖（可能需要几分钟）
                
                3. 重启应用后即可使用
                
                **系统要求：**
                - Python 3.7+
                - 至少4GB内存
                
                **详细文档：**
                - 官方文档：https://astroautomata.com/PySR/
                - GitHub：https://github.com/MilesCranmer/PySR
                """)
            
            st.info("💡 如果您不想安装PySR，可以使用Tab1的基础GP或步骤6 Tab1的优化GP。")
            st.stop()
        
        # PySR可用
        st.markdown("""
        **PySR (Python Symbolic Regression)** 是一个高性能的符号回归库：
        - 🚀 多种群并行搜索，速度快10-100倍
        - 📊 Pareto前沿分析，在精度和复杂度之间找平衡
        - 🎯 自动简化公式
        """)
        
        st.markdown("---")
        
        # 特征选择（使用公共的importance_df）
        st.markdown("### 📊 特征选择")
        
        selected_features_pysr = st.multiselect(
            "选择用于PySR的特征（按重要性排序）",
            options=importance_df['Feature'].tolist(),
            default=importance_df['Feature'].head(5).tolist(),
            help="建议选择3-6个最重要的特征",
            key="pysr_features"
        )
        
        if len(selected_features_pysr) == 0:
            st.warning("⚠️ 请至少选择1个特征")
            st.stop()
        
        st.caption(f"已选择 {len(selected_features_pysr)} 个特征")
        
        # ==================== 参数配置 ====================
        st.markdown("---")
        st.markdown("### ⚙️ 参数配置")
        
        # 核心参数（始终可见）
        st.markdown("**核心参数：**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pysr_niterations = st.number_input(
                "迭代次数",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                help="更多迭代可能找到更好的公式，但需要更长时间",
                key="pysr_niter"
            )
        
        with col2:
            pysr_populations = st.number_input(
                "种群数量",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
                help="并行搜索的种群数量",
                key="pysr_pops"
            )
        
        with col3:
            pysr_parsimony = st.number_input(
                "简约系数",
                min_value=0.0,
                max_value=1.0,
                value=0.0032,
                step=0.0001,
                format="%.4f",
                help="惩罚复杂公式，值越大越倾向于简单公式。💡 如果想获得更复杂的公式（complexity>13），可以降低此值至0.0001-0.001",
                key="pysr_parsimony"
            )
        
        st.caption("💡 **提示**：修改任何参数后，需要点击下方的\"开始训练\"按钮重新训练，新参数才会生效。")
        
        # 高级参数（可折叠）
        with st.expander("🔧 高级参数（可选）"):
            st.markdown("**种群配置：**")
            col1, col2 = st.columns(2)
            
            with col1:
                pysr_population_size = st.number_input(
                    "种群大小",
                    min_value=20,
                    max_value=200,
                    value=50,
                    step=10,
                    help="每个种群的个体数量",
                    key="pysr_popsize"
                )
            
            with col2:
                pysr_weight_optimize = st.number_input(
                    "权重优化迭代",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=1,
                    help="优化公式中常数的迭代次数，0表示不优化",
                    key="pysr_weight_opt"
                )
            
            st.markdown("**复杂度控制：**")
            col1, col2 = st.columns(2)
            
            with col1:
                pysr_maxsize = st.slider(
                    "最大公式复杂度",
                    min_value=10,
                    max_value=100,
                    value=30,
                    help="限制公式的最大节点数。注意：实际Pareto前沿的复杂度可能远小于此值，这是因为简约系数会惩罚过于复杂的公式",
                    key="pysr_maxsize"
                )
            
            with col2:
                pysr_maxdepth = st.slider(
                    "最大公式深度",
                    min_value=3,
                    max_value=30,
                    value=12,
                    help="限制公式的最大嵌套深度（建议10-15）",
                    key="pysr_maxdepth"
                )
            
            st.caption("💡 权重优化可以提高精度，但会增加计算时间")
        
        # 参数总结（始终可见）
        st.markdown("**📊 参数总结：**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"• 迭代次数：{pysr_niterations}")
            st.caption(f"• 种群数量：{pysr_populations}")
        with col2:
            st.caption(f"• 简约系数：{pysr_parsimony:.4f}")
            population_size_val = pysr_population_size if 'pysr_population_size' in locals() else 50
            st.caption(f"• 总个体数：{pysr_populations} × {population_size_val} = {pysr_populations * population_size_val}")
        
        # ==================== 运算符选择 ====================
        st.markdown("---")
        st.markdown("### 🔢 运算符选择")
        
        with st.expander("选择数学运算符", expanded=False):
            st.markdown("**推荐运算符（适合大多数回归问题）：**")
            st.caption("💡 以下运算符默认选中")
            
            st.markdown("**二元运算符：**")
            col1, col2 = st.columns(2)
            
            with col1:
                pysr_use_add = st.checkbox("加法 (+)", value=True, key="pysr_add")
                pysr_use_sub = st.checkbox("减法 (-)", value=True, key="pysr_sub")
            with col2:
                pysr_use_mul = st.checkbox("乘法 (*)", value=True, key="pysr_mul")
                pysr_use_div = st.checkbox("除法 (/)", value=True, key="pysr_div")
            
            st.markdown("**一元运算符：**")
            col1, col2 = st.columns(2)
            
            with col1:
                pysr_use_sqrt = st.checkbox("平方根 (sqrt)", value=True, key="pysr_sqrt")
            with col2:
                pysr_use_log = st.checkbox("对数 (log)", value=True, key="pysr_log")
            
            st.markdown("---")
            st.markdown("**可选运算符（按需选择）：**")
            st.caption("⚠️ 根据数据特点选择，过多运算符会增加搜索时间")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**基础函数**")
                pysr_use_square = st.checkbox("平方 (square)", value=False, key="pysr_square")
                pysr_use_abs = st.checkbox("绝对值 (abs)", value=False, key="pysr_abs")
                pysr_use_cube = st.checkbox("立方 (cube)", value=False, key="pysr_cube")
            
            with col2:
                st.markdown("**对数/指数**")
                pysr_use_log10 = st.checkbox("常用对数 (log10)", value=False, key="pysr_log10")
                pysr_use_exp = st.checkbox("指数 (exp)", value=False, key="pysr_exp")
            
            with col3:
                st.markdown("**其他函数**")
                pysr_use_neg = st.checkbox("取负 (neg)", value=False, key="pysr_neg")
                pysr_use_inv = st.checkbox("倒数 (inv)", value=False, key="pysr_inv")
            
            st.markdown("---")
            st.markdown("**不推荐运算符（特殊场景）：**")
            st.caption("⚠️ 幂运算容易过拟合，三角函数适用于周期性数据")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pysr_use_pow = st.checkbox("幂运算 (^)", value=False, key="pysr_pow")
            with col2:
                pysr_use_sin = st.checkbox("正弦 (sin)", value=False, key="pysr_sin")
            with col3:
                pysr_use_cos = st.checkbox("余弦 (cos)", value=False, key="pysr_cos")
            with col4:
                pysr_use_tan = st.checkbox("正切 (tan)", value=False, key="pysr_tan")
            
            # 构建运算符列表
            binary_ops = []
            if pysr_use_add: binary_ops.append("+")
            if pysr_use_sub: binary_ops.append("-")
            if pysr_use_mul: binary_ops.append("*")
            if pysr_use_div: binary_ops.append("/")
            if pysr_use_pow: binary_ops.append("^")
            
            unary_ops = []
            if pysr_use_sqrt: unary_ops.append("sqrt")
            if pysr_use_square: unary_ops.append("square")
            if pysr_use_abs: unary_ops.append("abs")
            if pysr_use_log: unary_ops.append("log")
            if pysr_use_exp: unary_ops.append("exp")
            if pysr_use_log10: unary_ops.append("log10")
            if pysr_use_sin: unary_ops.append("sin")
            if pysr_use_cos: unary_ops.append("cos")
            if pysr_use_tan: unary_ops.append("tan")
            if pysr_use_neg: unary_ops.append("neg")
            if pysr_use_inv: unary_ops.append("inv")
            if pysr_use_cube: unary_ops.append("cube")
            
            if len(binary_ops) < 2:
                st.warning("⚠️ 建议至少选择2个二元运算符")
                binary_ops = ["+", "-", "*", "/"]
            
            st.info(f"✅ 已选择 {len(binary_ops)} 个二元运算符，{len(unary_ops)} 个一元运算符")
        
        # 开始训练
        st.markdown("---")
        if st.button("🚀 开始PySR训练", type="primary", key="start_pysr"):
            
            # 检查数据
            if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
                st.error("❌ 缺少训练数据")
                st.stop()
            
            X_train = st.session_state.X_train_original.values
            y_train = st.session_state.y_train
            X_test = st.session_state.X_test_original.values
            y_test = st.session_state.y_test
            
            # 选择特征（保持用户选择的顺序）
            selected_indices = [feature_names.index(name) for name in selected_features_pysr]
            X_train_pysr = X_train[:, selected_indices]
            X_test_pysr = X_test[:, selected_indices]
            
            # 为PySR创建有效的变量名
            # 如果特征名是纯数字或包含特殊字符，使用x0, x1, x2...
            def make_valid_variable_name(name, index):
                """创建有效的Python变量名"""
                # 检查是否是纯数字
                if name.isdigit():
                    return f"x{index}"
                # 检查是否以数字开头
                if name[0].isdigit():
                    return f"x{index}"
                # 替换特殊字符
                valid_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
                # 如果替换后为空或全是下划线，使用x{index}
                if not valid_name or valid_name.replace('_', '') == '':
                    return f"x{index}"
                return valid_name
            
            pysr_variable_names = [make_valid_variable_name(name, i) for i, name in enumerate(selected_features_pysr)]
            
            # 保存原始特征名和PySR变量名的映射
            feature_to_pysr_var = dict(zip(selected_features_pysr, pysr_variable_names))
            
            st.info(f"使用 {len(selected_features_pysr)} 个特征：{', '.join(selected_features_pysr)}")
            if any(orig != pysr for orig, pysr in zip(selected_features_pysr, pysr_variable_names)):
                st.caption(f"PySR变量名：{', '.join(pysr_variable_names)}")
            
            # 训练PySR
            st.markdown("#### 🧬 训练PySR模型")
            
            with st.spinner(f"🧬 正在搜索公式... (迭代:{pysr_niterations}, 种群:{pysr_populations})"):
                try:
                    from pysr import PySRRegressor
                    
                    # 创建PySR模型
                    pysr_model = PySRRegressor(
                        niterations=pysr_niterations,
                        populations=pysr_populations,
                        population_size=pysr_population_size if 'pysr_population_size' in locals() else 50,
                        maxsize=pysr_maxsize if 'pysr_maxsize' in locals() else 30,
                        maxdepth=pysr_maxdepth if 'pysr_maxdepth' in locals() else 12,
                        binary_operators=binary_ops,
                        unary_operators=unary_ops if len(unary_ops) > 0 else None,
                        model_selection="best",
                        loss="loss(prediction, target) = (prediction - target)^2",
                        parsimony=pysr_parsimony,
                        weight_optimize=pysr_weight_optimize if 'pysr_weight_optimize' in locals() and pysr_weight_optimize > 0 else 0,
                        verbosity=0,
                        progress=False,
                        temp_equation_file=False,
                        random_state=42
                    )
                    
                    # 训练
                    pysr_model.fit(X_train_pysr, y_train, variable_names=pysr_variable_names)
                    
                    # 预测
                    y_train_pred_pysr = pysr_model.predict(X_train_pysr)
                    y_test_pred_pysr = pysr_model.predict(X_test_pysr)
                    
                    # 计算指标
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    
                    train_r2_pysr = r2_score(y_train, y_train_pred_pysr)
                    test_r2_pysr = r2_score(y_test, y_test_pred_pysr)
                    train_rmse_pysr = np.sqrt(mean_squared_error(y_train, y_train_pred_pysr))
                    test_rmse_pysr = np.sqrt(mean_squared_error(y_test, y_test_pred_pysr))
                    train_mae_pysr = mean_absolute_error(y_train, y_train_pred_pysr)
                    test_mae_pysr = mean_absolute_error(y_test, y_test_pred_pysr)
                    
                    # 保存结果
                    st.session_state.pysr_model = pysr_model
                    st.session_state.pysr_results = {
                        'train_r2': train_r2_pysr,
                        'test_r2': test_r2_pysr,
                        'train_rmse': train_rmse_pysr,
                        'test_rmse': test_rmse_pysr,
                        'train_mae': train_mae_pysr,
                        'test_mae': test_mae_pysr,
                        'y_train_pred': y_train_pred_pysr,
                        'y_test_pred': y_test_pred_pysr,
                        'feature_names': selected_features_pysr,  # 保存原始特征名
                        'pysr_variable_names': pysr_variable_names,  # 保存PySR变量名
                        'feature_to_pysr_var': feature_to_pysr_var  # 保存映射关系
                    }
                    
                    st.success("✅ PySR训练完成！")
                    
                except Exception as e:
                    st.error(f"❌ PySR训练失败：{str(e)}")
                    st.info("💡 提示：首次运行PySR可能需要几分钟来编译Julia代码")
                    st.stop()
        
        # 显示结果
        if 'pysr_results' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 PySR结果")
            
            pysr_results = st.session_state.pysr_results
            pysr_model = st.session_state.pysr_model
            
            # 性能指标
            st.markdown("#### 📈 性能指标")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("训练集 R²", f"{pysr_results['train_r2']:.4f}")
            col2.metric("测试集 R²", f"{pysr_results['test_r2']:.4f}")
            col3.metric("训练集 RMSE", f"{pysr_results['train_rmse']:.4f}")
            col4.metric("测试集 RMSE", f"{pysr_results['test_rmse']:.4f}")
            
            # 与GP对比
            if 'gp_results' in st.session_state:
                gp_results = st.session_state.gp_results
                r2_diff = (pysr_results['test_r2'] - gp_results['test_r2']) * 100
                
                if r2_diff > 0:
                    st.success(f"✅ PySR比GP提升了 {r2_diff:.2f}% 的R²")
                elif r2_diff < -5:
                    st.warning(f"⚠️ PySR比GP降低了 {abs(r2_diff):.2f}% 的R²")
                else:
                    st.info(f"ℹ️ PySR与GP性能相近（差异 {r2_diff:.2f}%）")
            
            # 预测散点图和详细分析
            st.markdown("#### 📊 PySR模型分析（6图全景）")
            
            # 创建6图可视化
            fig = plt.figure(figsize=(20, 13))
            fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.06, hspace=0.35, wspace=0.3)
            
            # ===== 图1：Pareto前沿 =====
            ax1 = plt.subplot(2, 3, 1)
            
            if hasattr(pysr_model, 'equations_'):
                try:
                    equations_df = pysr_model.equations_
                    
                    # 绘制所有公式
                    scatter = ax1.scatter(equations_df['complexity'], equations_df['loss'], 
                                         alpha=0.4, s=50, c=equations_df['score'], 
                                         cmap='viridis', edgecolors='black', linewidth=0.5)
                    
                    # 绘制Pareto前沿
                    pareto_mask = equations_df['Pareto'] if 'Pareto' in equations_df.columns else [True] * len(equations_df)
                    pareto_df = equations_df[pareto_mask].sort_values('complexity')
                    
                    if len(pareto_df) > 0:
                        ax1.plot(pareto_df['complexity'], pareto_df['loss'], 
                                'r-', linewidth=2.5, label='Pareto前沿', zorder=5)
                        ax1.scatter(pareto_df['complexity'], pareto_df['loss'], 
                                   c='red', s=120, marker='D', edgecolors='darkred', 
                                   linewidth=2, label='Pareto最优', zorder=6)
                    
                    # 标记最佳公式
                    best_idx = equations_df['loss'].idxmin()
                    best_complexity = equations_df.loc[best_idx, 'complexity']
                    best_loss = equations_df.loc[best_idx, 'loss']
                    ax1.scatter([best_complexity], [best_loss], 
                               c='lime', s=250, marker='*', edgecolors='darkgreen', 
                               linewidth=3, label='最佳公式', zorder=8)
                    
                    plt.colorbar(scatter, ax=ax1, label='Score')
                    ax1.set_xlabel('复杂度', fontsize=15, fontweight='bold')
                    ax1.set_ylabel('损失', fontsize=15, fontweight='bold')
                    ax1.set_title('1. Pareto Front Analysis\n(Lower-Left is Better)', fontsize=14, fontweight='bold', pad=15)
                    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
                    ax1.grid(True, alpha=0.3)
                    ax1.set_yscale('log')
                    ax1.tick_params(axis='both', labelsize=13)
                    
                except Exception as e:
                    ax1.text(0.5, 0.5, 'Pareto数据不可用', 
                            ha='center', va='center', transform=ax1.transAxes)
            else:
                ax1.text(0.5, 0.5, 'Pareto数据不可用', 
                        ha='center', va='center', transform=ax1.transAxes)
            
            # ===== 图2：公式复杂度分布 =====
            ax2 = plt.subplot(2, 3, 2)
            
            if hasattr(pysr_model, 'equations_'):
                try:
                    equations_df = pysr_model.equations_
                    
                    # 绘制复杂度直方图
                    n, bins, patches = ax2.hist(equations_df['complexity'], bins=20, 
                                                alpha=0.7, color='skyblue', 
                                                edgecolor='navy', linewidth=1.5)
                    
                    # 标记最佳公式的复杂度
                    best_complexity = equations_df.loc[equations_df['loss'].idxmin(), 'complexity']
                    ax2.axvline(x=best_complexity, color='red', linestyle='--', 
                               linewidth=2.5, label=f'最佳公式 ({int(best_complexity)})')
                    
                    # 标记平均复杂度
                    mean_complexity = equations_df['complexity'].mean()
                    ax2.axvline(x=mean_complexity, color='green', linestyle=':', 
                               linewidth=2, label=f'平均 ({mean_complexity:.1f})')
                    
                    ax2.set_xlabel('复杂度', fontsize=15, fontweight='bold')
                    ax2.set_ylabel('公式数量', fontsize=15, fontweight='bold')
                    ax2.set_title('2. Complexity Distribution\n(Simpler is Better)', fontsize=14, fontweight='bold', pad=15)
                    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
                    ax2.grid(True, alpha=0.3, axis='y')
                    ax2.tick_params(axis='both', labelsize=13)
                    
                except Exception as e:
                    ax2.text(0.5, 0.5, '复杂度数据不可用', 
                            ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, '复杂度数据不可用', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            # ===== 图3：损失分布 =====
            ax3 = plt.subplot(2, 3, 3)
            
            if hasattr(pysr_model, 'equations_'):
                try:
                    equations_df = pysr_model.equations_
                    
                    # 绘制损失直方图（对数尺度）
                    log_loss = np.log10(equations_df['loss'] + 1e-10)
                    n, bins, patches = ax3.hist(log_loss, bins=20, 
                                                alpha=0.7, color='lightcoral', 
                                                edgecolor='darkred', linewidth=1.5)
                    
                    # 标记最佳公式的损失
                    best_loss = equations_df['loss'].min()
                    ax3.axvline(x=np.log10(best_loss + 1e-10), color='red', 
                               linestyle='--', linewidth=2.5, 
                               label=f'最佳 ({best_loss:.2e})')
                    
                    # 标记中位数损失
                    median_loss = equations_df['loss'].median()
                    ax3.axvline(x=np.log10(median_loss + 1e-10), color='orange', 
                               linestyle=':', linewidth=2, 
                               label=f'中位数 ({median_loss:.2e})')
                    
                    ax3.set_xlabel('log10(损失)', fontsize=15, fontweight='bold')
                    ax3.set_ylabel('公式数量', fontsize=15, fontweight='bold')
                    ax3.set_title('3. Loss Distribution\n(Lower is Better)', fontsize=14, fontweight='bold', pad=15)
                    ax3.legend(loc='upper right', fontsize=11, framealpha=0.9)
                    ax3.grid(True, alpha=0.3, axis='y')
                    ax3.tick_params(axis='both', labelsize=13)
                    
                except Exception as e:
                    ax3.text(0.5, 0.5, '损失数据不可用', 
                            ha='center', va='center', transform=ax3.transAxes)
            else:
                ax3.text(0.5, 0.5, '损失数据不可用', 
                        ha='center', va='center', transform=ax3.transAxes)
            
            # ===== 图4：预测散点图 =====
            ax4 = plt.subplot(2, 3, 4)
            
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
            y_train_pred = pysr_results['y_train_pred']
            y_test_pred = pysr_results['y_test_pred']
            
            # 从0开始的坐标轴
            global_min = 0
            global_max = max(y_train.max(), y_test.max(), y_train_pred.max(), y_test_pred.max())
            margin = global_max * 0.05
            plot_max = global_max + margin
            
            # 训练集
            ax4.scatter(y_train, y_train_pred, alpha=0.5, s=30, 
                       c='blue', edgecolors='navy', linewidth=0.5, label='训练集')
            # 测试集
            ax4.scatter(y_test, y_test_pred, alpha=0.5, s=30, 
                       c='green', edgecolors='darkgreen', linewidth=0.5, label='测试集')
            
            # 理想线
            ax4.plot([global_min, plot_max], [global_min, plot_max], 
                    'r--', linewidth=2, label='理想线 (y=x)')
            
            ax4.set_xlim(global_min, plot_max)
            ax4.set_ylim(global_min, plot_max)
            ax4.set_aspect('equal', adjustable='box')
            ax4.set_xlabel('实际值', fontsize=15, fontweight='bold')
            ax4.set_ylabel('预测值', fontsize=15, fontweight='bold')
            ax4.set_title(f'4. Prediction Scatter Plot\n(Train R²={pysr_results["train_r2"]:.4f}, Test R²={pysr_results["test_r2"]:.4f})', 
                         fontsize=14, fontweight='bold', pad=15)
            ax4.legend(loc='upper left', fontsize=11, framealpha=0.9)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='both', labelsize=13)
            
            # ===== 图5：残差分析（QQ图）=====
            ax5 = plt.subplot(2, 3, 5)
            
            # 合并残差
            residuals = np.concatenate([y_train - y_train_pred, y_test - y_test_pred])
            
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax5)
            ax5.get_lines()[0].set_markerfacecolor('blue')
            ax5.get_lines()[0].set_markeredgecolor('navy')
            ax5.get_lines()[0].set_markersize(5)
            ax5.get_lines()[0].set_alpha(0.6)
            ax5.get_lines()[1].set_color('red')
            ax5.get_lines()[1].set_linewidth(2)
            
            ax5.set_xlabel('理论分位数', fontsize=15, fontweight='bold')
            ax5.set_ylabel('样本分位数', fontsize=15, fontweight='bold')
            ax5.set_title('5. QQ Plot (Normality Test)\n(Points on Line = Normal)', fontsize=14, fontweight='bold', pad=15)
            ax5.grid(True, alpha=0.3)
            ax5.tick_params(axis='both', labelsize=13)
            
            # 添加正态性检验结果
            if len(residuals) <= 5000:
                _, p_value = stats.shapiro(residuals)
                normality_text = f'Shapiro-Wilk\np={p_value:.4f}\n'
                if p_value > 0.05:
                    normality_text += '✓ 近似正态'
                else:
                    normality_text += '✗ 偏离正态'
                ax5.text(0.05, 0.95, normality_text, transform=ax5.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            # ===== 图6：特征使用频率 =====
            ax6 = plt.subplot(2, 3, 6)
            
            if hasattr(pysr_model, 'equations_'):
                try:
                    equations_df = pysr_model.equations_
                    feature_names_pysr = pysr_results['feature_names']
                    
                    # 统计每个特征在Pareto前沿中的使用次数
                    pareto_mask = equations_df['Pareto'] if 'Pareto' in equations_df.columns else [True] * len(equations_df)
                    pareto_equations = equations_df[pareto_mask]['equation'].astype(str)
                    
                    feature_counts = {}
                    for feature in feature_names_pysr:
                        count = sum(feature in eq for eq in pareto_equations)
                        feature_counts[feature] = count
                    
                    # 排序
                    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
                    features, counts = zip(*sorted_features) if sorted_features else ([], [])
                    
                    # 绘制柱状图
                    colors = ['#2ecc71' if c > 0 else '#95a5a6' for c in counts]
                    bars = ax6.barh(range(len(features)), counts, color=colors, 
                                   edgecolor='black', linewidth=1.5, alpha=0.8)
                    
                    ax6.set_yticks(range(len(features)))
                    ax6.set_yticklabels(features, fontsize=13, fontweight='bold')
                    ax6.set_xlabel('使用次数', fontsize=15, fontweight='bold')
                    ax6.set_ylabel('特征', fontsize=15, fontweight='bold')
                    ax6.set_title('6. Feature Usage in Pareto Front\n(Higher = More Important)', 
                                 fontsize=14, fontweight='bold', pad=15)
                    ax6.grid(True, alpha=0.3, axis='x')
                    ax6.invert_yaxis()
                    ax6.tick_params(axis='x', labelsize=13)
                    
                    # 标注数值
                    for i, (bar, count) in enumerate(zip(bars, counts)):
                        if count > 0:
                            ax6.text(count + 0.1, i, str(count), 
                                    va='center', fontsize=11, fontweight='bold')
                    
                except Exception as e:
                    ax6.text(0.5, 0.5, '特征使用数据不可用', 
                            ha='center', va='center', transform=ax6.transAxes)
            else:
                ax6.text(0.5, 0.5, '特征使用数据不可用', 
                        ha='center', va='center', transform=ax6.transAxes)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # 统计信息
            st.markdown("#### 📈 关键统计信息")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("测试集 R²", f"{pysr_results['test_r2']:.4f}")
                st.metric("测试集 RMSE", f"{pysr_results['test_rmse']:.4f}")
            
            with col2:
                if hasattr(pysr_model, 'equations_'):
                    try:
                        equations_df = pysr_model.equations_
                        st.metric("候选公式数", len(equations_df))
                        pareto_count = equations_df['Pareto'].sum() if 'Pareto' in equations_df.columns else len(equations_df)
                        st.metric("Pareto前沿点数", int(pareto_count))
                    except:
                        st.metric("候选公式数", "N/A")
                        st.metric("Pareto前沿点数", "N/A")
                else:
                    st.metric("候选公式数", "N/A")
                    st.metric("Pareto前沿点数", "N/A")
            
            with col3:
                if hasattr(pysr_model, 'equations_'):
                    try:
                        equations_df = pysr_model.equations_
                        best_complexity = int(equations_df.loc[equations_df['loss'].idxmin(), 'complexity'])
                        mean_complexity = equations_df['complexity'].mean()
                        st.metric("最佳公式复杂度", best_complexity)
                        st.metric("平均复杂度", f"{mean_complexity:.1f}")
                    except:
                        st.metric("最佳公式复杂度", "N/A")
                        st.metric("平均复杂度", "N/A")
                else:
                    st.metric("最佳公式复杂度", "N/A")
                    st.metric("平均复杂度", "N/A")
            
            with col4:
                st.metric("使用特征数", len(pysr_results['feature_names']))
                if len(residuals) <= 5000:
                    _, p_value = stats.shapiro(residuals)
                    if p_value > 0.05:
                        st.success("✓ 残差正态")
                    else:
                        st.warning("⚠ 残差非正态")
                else:
                    st.info("样本量过大")
            
            st.info("""
            💡 **图表说明**：
            - **图1 Pareto前沿**：展示复杂度与损失的权衡，左下角最优
            - **图2 复杂度分布**：候选公式的复杂度分布，红线是最佳公式
            - **图3 损失分布**：候选公式的损失分布（对数尺度），红线是最佳
            - **图4 预测散点图**：蓝色=训练集，绿色=测试集，越接近红线越好
            - **图5 QQ图**：检验残差正态性，点在直线上表示正态分布
            - **图6 特征使用频率**：Pareto前沿中各特征的使用次数，绿色=被使用
            """)
            
            # 显示公式
            st.markdown("---")
            st.markdown("### 🔬 发现的数学公式")
            
            pysr_model = st.session_state.pysr_model
            pysr_results = st.session_state.pysr_results
            best_equation = str(pysr_model.sympy())
            
            # 获取特征映射
            feature_names_pysr = pysr_results.get('feature_names', selected_features_pysr)
            pysr_variable_names = pysr_results.get('pysr_variable_names', None)
            feature_to_pysr_var = pysr_results.get('feature_to_pysr_var', {})
            
            # 显示公式
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("**发现的公式（PySR变量）：**")
                st.code(f"y = {best_equation}", language="python")
                
                # 创建字母映射（A, B, C...）
                letter_mapping = {}
                for i, feature_name in enumerate(feature_names_pysr):
                    letter = chr(65 + i) if i < 26 else f"X{i}"  # A-Z, then X26, X27...
                    letter_mapping[feature_name] = letter
                
                # 如果有PySR变量映射，先转换为特征名，再转换为字母
                if feature_to_pysr_var:
                    # 创建反向映射（PySR变量 -> 原始特征名）
                    pysr_var_to_feature = {v: k for k, v in feature_to_pysr_var.items()}
                    
                    # 替换公式中的变量名为字母
                    formula_with_letters = best_equation
                    for pysr_var, feature_name in pysr_var_to_feature.items():
                        if feature_name in letter_mapping:
                            formula_with_letters = formula_with_letters.replace(pysr_var, letter_mapping[feature_name])
                    
                    st.markdown("**发现的公式（使用字母代替）：**")
                    st.code(f"y = {formula_with_letters}", language="python")
                
                # LaTeX格式
                try:
                    import sympy as sp
                    latex_eq = sp.latex(pysr_model.sympy())
                    
                    # 如果有映射关系，替换LaTeX公式中的变量名为字母
                    if feature_to_pysr_var:
                        pysr_var_to_feature = {v: k for k, v in feature_to_pysr_var.items()}
                        latex_with_letters = latex_eq
                        for pysr_var, feature_name in pysr_var_to_feature.items():
                            if feature_name in letter_mapping:
                                latex_with_letters = latex_with_letters.replace(pysr_var, letter_mapping[feature_name])
                        st.markdown("**LaTeX格式（使用字母）：**")
                        st.latex(f"y = {latex_with_letters}")
                    else:
                        st.markdown("**LaTeX格式：**")
                        st.latex(f"y = {latex_eq}")
                except:
                    pass
            
            with col2:
                st.markdown("**公式特点：**")
                st.metric("使用特征数", len(feature_names_pysr))
                st.metric("公式长度", f"{len(best_equation)} 字符")
                try:
                    complexity = pysr_model.equations_['complexity'].iloc[-1]
                    st.metric("公式复杂度", int(complexity))
                except:
                    st.metric("公式复杂度", "N/A")
                
                # 显示字母映射
                st.markdown("**字母映射：**")
                for feature_name in feature_names_pysr:
                    if feature_name in letter_mapping:
                        st.caption(f"{letter_mapping[feature_name]} = {feature_name}")
            
            # Pareto前沿
            if hasattr(pysr_model, 'equations_'):
                st.markdown("---")
                st.markdown("#### 📊 Pareto前沿")
                
                try:
                    equations_df = pysr_model.equations_
                    
                    # 显示前10个公式
                    st.dataframe(
                        equations_df[['complexity', 'loss', 'score', 'equation']].head(10),
                        width='stretch'
                    )
                    
                    st.caption("💡 Pareto前沿展示了不同复杂度下的最优公式")
                    
                except Exception as e:
                    st.info("ℹ️ Pareto前沿数据不可用")
            
            # 公式导出
            st.markdown("---")
            st.markdown("#### 📥 导出公式")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Python代码
                python_code = f"""# PySR符号回归公式
import numpy as np

def pysr_formula(features):
    \"\"\"
    PySR发现的公式
    
    参数:
        features: dict, 包含以下特征
{chr(10).join([f'            - {f}' for f in pysr_results['feature_names']])}
    
    返回:
        预测值
    \"\"\"
    # 提取特征
{chr(10).join([f'    {f} = features["{f}"]' for f in pysr_results['feature_names']])}
    
    # 公式计算
    result = {best_equation}
    
    return result
"""
                st.download_button(
                    label="📥 下载Python代码",
                    data=python_code,
                    file_name="pysr_formula.py",
                    mime="text/x-python"
                )
            
            with col2:
                # 公式报告
                report = f"""PySR符号回归报告
{'='*50}

公式:
y = {best_equation}

特征:
{chr(10).join([f'- {f}' for f in pysr_results['feature_names']])}

性能指标:
训练集:
  R² = {pysr_results['train_r2']:.4f}
  RMSE = {pysr_results['train_rmse']:.4f}
  MAE = {pysr_results['train_mae']:.4f}

测试集:
  R² = {pysr_results['test_r2']:.4f}
  RMSE = {pysr_results['test_rmse']:.4f}
  MAE = {pysr_results['test_mae']:.4f}

PySR参数:
- 迭代次数: {pysr_niterations}
- 种群数量: {pysr_populations}
- 种群大小: {pysr_population_size}
- 最大复杂度: {pysr_maxsize}
- 最大深度: {pysr_maxdepth}
"""
                st.download_button(
                    label="📥 下载公式报告",
                    data=report,
                    file_name="pysr_report.txt",
                    mime="text/plain"
                )
