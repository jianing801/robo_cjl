# DM10010 大腿装配体 GP 代理模型

本目录单独保存 DM-J10010（非 L）大腿装配体的 GP 拟合资料。它与 DM-J10010L 是不同型号，分别使用各自的 SW 数据和 GP 模型。

- 输入：上级目录的完整质量属性 CSV；使用 0.25 m 基准和 0.26–0.40 m 的 15 个 SW 扫描点。
- 输出目标：质量、质心 xyz、质心惯量张量的 6 个独立项，共 10 个字段。
- 模型：MATLAB fitrgp，Matérn 5/2 核；每个输出单独训练。
- 最终模型使用全部 16 个已采集点；精度通过嵌套 LOOCV 检查，不把训练误差当独立测试精度。
- 惯性符号：保留 SolidWorks 原始 Lxy/Lxz/Lyz；生成的 urdf_i* 输出列按标准惯性矩阵约定对交叉项取负号。
- 预测域：0.25–0.40 m。图中的 95% 带代表 GPR 模型预测不确定度，不代表实物误差。

运行 fit_dm10010_thigh_gp.m 即可重建模型、图和 CSV。需要 MATLAB Statistics and Machine Learning Toolbox（fitrgp）。
