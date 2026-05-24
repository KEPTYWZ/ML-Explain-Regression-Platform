@echo off
echo ========================================
echo 机器学习分析平台 - 依赖安装脚本
echo ========================================
echo.

echo 正在安装必需的依赖包...
echo.

pip install streamlit pandas numpy matplotlib seaborn scikit-learn bayesian-optimization

echo.
echo ========================================
echo 正在安装可选的依赖包...
echo ========================================
echo.

echo 安装 LightGBM...
pip install lightgbm

echo.
echo 安装 XGBoost...
pip install xgboost

echo.
echo 安装 SHAP...
pip install shap

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 运行以下命令启动应用：
echo    streamlit run app.py
echo.
echo 或双击 run.bat 文件
echo.
pause
