============================================================
北方华创温控部件拓扑优化项目 - 技术数据说明
============================================================

本文件夹用于提交项目源代码、优化数据集、COMSOL模型及最终几何文件。

============================================================
一、目录总览
============================================================

[src]
作用：项目源代码目录，包含圆柱簇与 zigzag 簇的 COMSOL 仿真、优化搜索、结果解析和代理模型分析代码。

[data_set]
作用：优化过程数据集目录，保存两类结构族的 trial 记录，可用于复现实验统计、Pareto 分析和代理模型训练。

[STEP&STL]
作用：COMSOL 模型与最终几何文件目录，包含 baseline、最佳模型的 mph 文件，以及最佳模型的 STEP/STL 几何文件。

============================================================
二、src/cylinder_family 文件说明
============================================================

cylinder_baseline.java
作用：圆柱簇 baseline 生命周期仿真脚本，在 COMSOL Java Shell 中运行，输出电压、温度、寿命和辐射功率指标。

cylinder_process_views.java
作用：圆柱簇过程可视化脚本，在 baseline 流程基础上生成 4 个退蚀阶段结果图，可切换查看温度场或 von Mises 应力场。

comsol_runner.py
作用：圆柱簇 Python-COMSOL 批量仿真接口，用于被优化脚本调用并返回每个候选半径组合的核心指标。

optuna_optimize.py
作用：圆柱簇贝叶斯优化脚本，调用 comsol_runner.py 批量搜索半径参数，并将每次试验写入 trials.csv。

parse_results.py
作用：圆柱簇优化结果解析脚本，用于读取 trial 数据、筛选可行解、绘制 Pareto 图和优化过程图。

surrogate_analysis.py
作用：圆柱簇代理模型与 SHAP 分析脚本，用随机森林近似仿真结果，并解释半径参数对功率和寿命的影响。

============================================================
三、src/zigzag_family 文件说明
============================================================

zigzag_baseline.java
作用：zigzag 簇 baseline 生命周期仿真脚本，在 COMSOL Java Shell 中运行，输出电压、温度、寿命和辐射功率指标。

zigzag_process_views.java
作用：zigzag 簇过程可视化脚本，在 baseline 流程基础上生成 4 个退蚀阶段结果图，可切换查看温度场或 von Mises 应力场。

zigzag_runner.py
作用：zigzag 簇 Python-COMSOL 批量仿真接口，输入 N、L_RUN、z_first 三个自由参数，并自动按体积守恒计算 side。

optuna_optimize.py
作用：zigzag 簇贝叶斯优化脚本，调用 zigzag_runner.py 搜索 N、L_RUN、z_first，并记录每次试验结果。

parse_results.py
作用：zigzag 簇优化结果解析脚本，用于读取 trial 数据、筛选可行解、绘制 Pareto 图和优化过程图。

surrogate_analysis.py
作用：zigzag 簇代理模型与 SHAP 分析脚本，用随机森林解释 N、L_RUN、z_first 对功率和寿命的影响。

============================================================
四、data_set 文件说明
============================================================

cylinder_trials.csv
作用：圆柱簇优化 trial 数据集，记录每组半径参数、COMSOL 求解结果、寿命、功率、状态和耗时。

zigzag_trials.csv
作用：zigzag 簇优化 trial 数据集，记录 N、L_RUN、z_first、side、COMSOL 求解结果、寿命、功率、状态和耗时。

============================================================
五、STEP&STL 文件说明
============================================================

STEP&STL/cylinder_family/cylinder_baseline.mph
作用：圆柱簇 baseline COMSOL 模型文件。

STEP&STL/cylinder_family/cylinder_best_model.mph
作用：圆柱簇最佳模型 COMSOL 模型文件。

STEP&STL/cylinder_family/cylinder_best_model.step
作用：圆柱簇最佳模型 STEP 几何文件，用于 CAD/CAE 软件查看或后续建模。

STEP&STL/cylinder_family/cylinder_best_model.stl
作用：圆柱簇最佳模型 STL 几何文件，用于三维网格查看、展示或快速导入。

STEP&STL/zigzag_family/zigzag_baseline.mph
作用：zigzag 簇 baseline COMSOL 模型文件。

STEP&STL/zigzag_family/zigzag_best_model.mph
作用：zigzag 簇最佳模型 COMSOL 模型文件。

STEP&STL/zigzag_family/zigzag_best_model.step
作用：zigzag 簇最佳模型 STEP 几何文件，用于 CAD/CAE 软件查看或后续建模。

STEP&STL/zigzag_family/zigzag_best_model.stl
作用：zigzag 簇最佳模型 STL 几何文件，用于三维网格查看、展示或快速导入。

============================================================
六、运行提示
============================================================

COMSOL Java 文件：建议在 COMSOL Desktop 的 Java Shell 中直接运行。

Python 优化脚本：需要本机安装 COMSOL、Python 依赖库 mph / jpype / optuna / numpy / matplotlib / scikit-learn / shap。

数据复现顺序：先查看 data_set 中的 trial 数据，再使用 parse_results.py 和 surrogate_analysis.py 复现统计图和解释分析。

几何查看顺序：优先打开 STEP 文件查看模型几何；如需完整物理场和结果树，打开对应 mph 文件。

============================================================
说明结束
============================================================
