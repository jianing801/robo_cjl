%% DM10010 大腿装配体：URDF 质量属性的 GP 拟合
% 数据来源：父目录的 DM10010_大腿装配体质量属性完整数据.csv
% 样本：0.25 m 基准 + 0.26–0.40 m 的 15 个 SolidWorks 扫描点
%
% 拟合目标为 URDF 惯性描述需要的 10 个量：
%   mass、质心 xyz、质心处惯性张量的 6 个独立分量
% 对有变化的输出分别建立 Matern 5/2 Gaussian Process Regression；
% 若某字段在全部输入点完全相同，则直接保留该常数，不伪造 GP 噪声。
% 所有已有点都参与建模；精度用嵌套式留一交叉验证（每折重拟合超参数）估计。
%
% SolidWorks 导出的 Lxy/Lxz/Lyz 按“正张量记数法”记录惯性积；
% 导出给标准 URDF 惯性矩阵时，对非对角项取负号。
% 当前转换假设“坐标系1”与 URDF 大腿 link 坐标系对齐。

clear; clc; close all;
rng(42, 'twister');

scriptDir = fileparts(mfilename('fullpath'));
inputFile = fullfile(scriptDir, '..', 'DM10010_大腿装配体质量属性完整数据.csv');
outputDir = fullfile(scriptDir, 'output');
if ~isfolder(outputDir)
    mkdir(outputDir);
end
if ~isfile(inputFile)
    error('未找到输入 CSV：%s', inputFile);
end

T = readtable(inputFile, 'VariableNamingRule', 'preserve');
x = double(T.length_m(:));
if height(T) ~= 16 || numel(unique(x)) ~= numel(x)
    error('预期输入为 16 个唯一腿长点（0.25 m 基准 + 15 个扫描点）。');
end
if any(x < 0.25 | x > 0.40)
    error('数据中存在超出 0.25–0.40 m 大腿长度范围的记录。');
end

properties = ["mass_kg", "com_x_m", "com_y_m", "com_z_m", ...
    "Lxx", "Lxy", "Lxz", "Lyy", "Lyz", "Lzz"];
labels = ["质量", "质心 X", "质心 Y", "质心 Z", ...
    "惯量 Lxx", "惯积 Lxy", "惯积 Lxz", "惯量 Lyy", "惯积 Lyz", "惯量 Lzz"];
units = ["kg", "m", "m", "m", repmat("kg·m²",1,6)];

xq = linspace(0.25, 0.40, 301)';
n = numel(x);
nProperties = numel(properties);
models = struct();
modelInfo = cell(nProperties, 14);
metricRows = cell(nProperties, 10);
looTable = table(string(T.record_type), x, double(T.delta_mm), ...
    'VariableNames', {'record_type','length_m','delta_mm'});
denseTable = table(xq, 'VariableNames', {'length_m'});
actualValues = zeros(n, nProperties);
looValues = zeros(n, nProperties);
denseMeans = zeros(numel(xq), nProperties);
denseStd = zeros(numel(xq), nProperties);

for k = 1:nProperties
    prop = properties(k);
    y = double(T.(char(prop)));
    y = y(:);
    actualValues(:,k) = y;

    [model, yCenter, yScale, isConstant] = fitOneGP(x, y);
    models.(char(prop)) = struct('model', model, ...
        'yCenter', yCenter, 'yScale', yScale, ...
        'isConstant', isConstant, 'property', char(prop), ...
        'units', char(units(k)), 'kernel', 'Matern52');

    [mu, sd] = predictOneGP(model, isConstant, y(1), yCenter, yScale, xq);
    denseMeans(:,k) = mu;
    denseStd(:,k) = sd;
    denseTable.(char(prop + "_mean")) = mu;
    denseTable.(char(prop + "_std")) = sd;

    % 嵌套 LOOCV：每个留出点都只使用其余点重新拟合 GPR。
    for i = 1:n
        trainMask = true(n,1);
        trainMask(i) = false;
        [foldModel, foldCenter, foldScale, foldConstant] = ...
            fitOneGP(x(trainMask), y(trainMask));
        if foldConstant
            looValues(i,k) = y(1);
        else
            looZ = predict(foldModel, x(i));
            looValues(i,k) = foldCenter + foldScale * looZ;
        end
    end

    err = looValues(:,k) - y;
    rmse = sqrt(mean(err.^2));
    mae = mean(abs(err));
    maxAbs = max(abs(err));
    sst = sum((y - mean(y)).^2);
    if sst > eps(max(1, max(abs(y))))
        r2 = 1 - sum(err.^2) / sst;
    elseif maxAbs == 0
        r2 = 1;
    else
        r2 = NaN;
    end
    peakScale = max(abs(y));
    if peakScale > 0
        nrmse = 100 * rmse / peakScale;
    else
        nrmse = 0;
    end
    metricRows(k,:) = {char(prop), char(labels(k)), char(units(k)), n, ...
        rmse, mae, maxAbs, r2, nrmse, isConstant};

    looTable.(char(prop + "_actual")) = y;
    looTable.(char(prop + "_loo_pred")) = looValues(:,k);

    if isConstant
        modelType = "constant: all SolidWorks points identical";
        kernelScale = NaN;
        sigma = 0;
        kernelParameters = "";
    else
        modelType = "GPR Matern 5/2";
        % For Matern 5/2, KernelParameters(1) is the fitted length scale.
        kernelScale = model.KernelInformation.KernelParameters(1);
        sigma = model.Sigma;
        kernelParameters = mat2str(model.KernelInformation.KernelParameters(:)', 8);
    end
    modelInfo(k,:) = {char(prop), char(labels(k)), char(units(k)), ...
        char(modelType), n, min(x), max(x), kernelScale, sigma, ...
        char(kernelParameters), yCenter, yScale, rmse, nrmse};
end

% 将 SolidWorks 原始正张量惯性积转成 URDF 的标准惯性矩阵非对角项。
denseTable.urdf_ixx_kg_m2 = denseTable.Lxx_mean;
denseTable.urdf_ixy_kg_m2 = -denseTable.Lxy_mean;
denseTable.urdf_ixz_kg_m2 = -denseTable.Lxz_mean;
denseTable.urdf_iyy_kg_m2 = denseTable.Lyy_mean;
denseTable.urdf_iyz_kg_m2 = -denseTable.Lyz_mean;
denseTable.urdf_izz_kg_m2 = denseTable.Lzz_mean;

% 连续区间物理合法性检查：正定与主惯量三角不等式。
minimumEigenvalue = inf;
minimumTriangleMargin = inf;
valid = true(numel(xq),1);
for i = 1:numel(xq)
    J = [denseTable.Lxx_mean(i), -denseTable.Lxy_mean(i), -denseTable.Lxz_mean(i); ...
        -denseTable.Lxy_mean(i), denseTable.Lyy_mean(i), -denseTable.Lyz_mean(i); ...
        -denseTable.Lxz_mean(i), -denseTable.Lyz_mean(i), denseTable.Lzz_mean(i)];
    eigsJ = sort(eig((J+J')/2));
    triangleMargin = eigsJ(1) + eigsJ(2) - eigsJ(3);
    minimumEigenvalue = min(minimumEigenvalue, eigsJ(1));
    minimumTriangleMargin = min(minimumTriangleMargin, triangleMargin);
    valid(i) = all(eigsJ > 0) && triangleMargin >= -1e-12 && denseTable.mass_kg_mean(i) > 0;
end
denseTable.physical_valid = valid;

% 输出逐点 LOOCV 指标与预测记录。
metricsTable = cell2table(metricRows, 'VariableNames', ...
    {'property','label_cn','unit','point_count','loo_rmse','loo_mae', ...
    'loo_max_abs_error','loo_r2','loo_nrmse_percent_peak','constant_data'});
metadataTable = cell2table(modelInfo, 'VariableNames', ...
    {'property','label_cn','unit','model_type','point_count','data_min_m', ...
    'data_max_m','kernel_scale','sigma_standardized','kernel_parameters', ...
    'y_center','y_scale','loo_rmse','loo_nrmse_percent_peak'});
writetable(metricsTable, fullfile(outputDir, 'gp_loo_cv_metrics.csv'));
writetable(metadataTable, fullfile(outputDir, 'gp_model_metadata.csv'));
writetable(looTable, fullfile(outputDir, 'gp_loo_predictions.csv'));
writetable(denseTable, fullfile(outputDir, 'gp_dense_predictions_301.csv'));

% 两张适合放入汇报的曲线图。
plotGroup(1:4, properties, labels, units, x, actualValues, xq, ...
    denseMeans, denseStd, metricsTable, '质量与质心 GP 拟合', ...
    fullfile(outputDir, 'DM10010_thigh_GP_mass_COM.png'), [2,2]);
plotGroup(5:10, properties, labels, units, x, actualValues, xq, ...
    denseMeans, denseStd, metricsTable, '质心惯性参数 GP 拟合', ...
    fullfile(outputDir, 'DM10010_thigh_GP_inertia.png'), [2,3]);

physicalTable = table(numel(xq), sum(valid), minimumEigenvalue, ...
    minimumTriangleMargin, all(valid), ...
    'VariableNames', {'dense_points','physically_valid_points', ...
    'minimum_eigenvalue_kg_m2','minimum_triangle_margin_kg_m2','all_valid'});
writetable(physicalTable, fullfile(outputDir, 'gp_physical_check.csv'));

summaryPath = fullfile(outputDir, 'README_CN.md');
writeReadme(summaryPath, metricsTable, physicalTable, T);

% 保存 MATLAB GPR 模型，便于后续直接调用 predict。
metadata = metadataTable;
save(fullfile(outputDir, 'DM10010_thigh_gpr_models.mat'), ...
    'models', 'metadata', 'properties', 'labels', 'units', 'x', 'T', '-v7.3');

fprintf('完成：%d 个输入点，%d 个质量/质心/惯量目标。\n', n, nProperties);
fprintf('所有 301 个区间采样点物理检查通过：%d\n', all(valid));
disp(metricsTable);
fprintf('输出目录：%s\n', outputDir);


function [model, yCenter, yScale, isConstant] = fitOneGP(x, y)
% 先对输出单独标准化，避免质量、坐标和惯量的数量级差异影响优化。
x = double(x(:));
y = double(y(:));
yCenter = mean(y);
rawScale = std(y, 1);
constantTolerance = max(1e-15, 32 * eps(max(1, max(abs(y)))));
isConstant = (max(y)-min(y)) <= constantTolerance;
if isConstant
    model = [];
    yScale = 1;
    return;
end
yScale = rawScale;
if ~isfinite(yScale) || yScale <= 0
    error('输出尺度无效，无法建立 GP。');
end
z = (y-yCenter)/yScale;
model = fitrgp(x, z, ...
    'KernelFunction', 'matern52', ...
    'BasisFunction', 'constant', ...
    'FitMethod', 'exact', ...
    'PredictMethod', 'exact', ...
    'Standardize', true);
end


function [mu, sd] = predictOneGP(model, isConstant, constantValue, yCenter, yScale, xq)
if isConstant
    mu = repmat(constantValue, size(xq));
    sd = zeros(size(xq));
else
    [zMean, zStd] = predict(model, xq);
    mu = yCenter + yScale .* zMean;
    sd = abs(yScale) .* zStd;
end
end


function plotGroup(indices, properties, labels, units, x, actual, xq, means, stds, metricsTable, titleText, filePath, layout)
fig = figure('Color','w','Position',[100 80 1500 880],'Visible','off');
tl = tiledlayout(layout(1), layout(2), 'TileSpacing','compact','Padding','compact');
blue = [0.05 0.32 0.61];
band = [0.66 0.82 0.94];
point = [0.12 0.16 0.20];
for j = 1:numel(indices)
    k = indices(j);
    ax = nexttile(tl);
    mu = means(:,k);
    sd = stds(:,k);
    lower = mu - 1.96*sd;
    upper = mu + 1.96*sd;
    fill(ax, [xq; flipud(xq)], [lower; flipud(upper)], band, ...
        'EdgeColor','none','FaceAlpha',0.58,'DisplayName','95% GP predictive interval');
    hold(ax,'on');
    plot(ax, xq, mu, '-', 'Color',blue,'LineWidth',2.2,'DisplayName','GP mean');
    scatter(ax, x, actual(:,k), 30, 'o','MarkerFaceColor',point, ...
        'MarkerEdgeColor','w','LineWidth',0.6,'DisplayName','SolidWorks data');
    grid(ax,'on'); box(ax,'on');
    ax.GridAlpha=0.15; ax.FontName='Arial'; ax.FontSize=9;
    if max(abs([actual(:,k); mu])) < 1e-2
        ytickformat(ax,'%.2e');
        ax.YAxis.Exponent = 0;
        if range(actual(:,k)) == 0 && all(sd == 0)
            centerValue = actual(1,k);
            halfSpan = max(abs(centerValue)*0.10, 1e-12);
            ylim(ax, centerValue + [-halfSpan, halfSpan]);
        end
    end
    xlim(ax,[0.25 0.40]);
    xlabel(ax,'大腿长度 L (m)','FontName','Microsoft YaHei');
    ylabel(ax,sprintf('%s (%s)',labels(k),units(k)),'FontName','Microsoft YaHei');
    rmse = metricsTable.loo_rmse(k);
    title(ax,sprintf('%s   LOOCV RMSE = %.3g',labels(k),rmse), ...
        'FontName','Microsoft YaHei','FontWeight','bold','FontSize',10);
    if j == 1
        legend(ax,'Location','best','FontSize',8,'Box','off');
    end
end
title(tl,titleText,'FontName','Microsoft YaHei','FontWeight','bold','FontSize',16);
exportgraphics(fig,filePath,'Resolution',240);
savefig(fig, replace(filePath,'.png','.fig'));
close(fig);
end


function writeReadme(path, metricsTable, physicalTable, T)
fid=fopen(path,'w','n','UTF-8');
if fid<0, error('无法写入拟合说明。'); end
cleanup=onCleanup(@() fclose(fid));
fprintf(fid,'# DM10010 大腿装配体 URDF 参数 GP 拟合\n\n');
fprintf(fid,'## 拟合目标与数据\n\n');
fprintf(fid,'拟合 URDF 惯性属性需要的 10 个输出：质量、质心 xyz、以及质心处惯性张量的 6 个独立分量（Lxx、Lxy、Lxz、Lyy、Lyz、Lzz）。\n\n');
fprintf(fid,'输入共 %d 个 SolidWorks 点：0.25 m 基准点与 0.26–0.40 m 的 15 个扫描点。预测域为 0.25–0.40 m，区间内 301 个密集采样点。\n\n',height(T));
fprintf(fid,'## GP 设置\n\n');
fprintf(fid,'- 每个有变化的输出分别建立精确 Gaussian Process Regression。\n');
fprintf(fid,'- Lxy 在 16 个 SolidWorks 点完全相同，因此按常数 2.5e-7 kg·m² 保存，不人为引入 GP 噪声。\n');
fprintf(fid,'- 协方差核：Matérn 5/2；常数均值项；输入和输出分别标准化。\n');
fprintf(fid,'- MATLAB fitrgp 对核尺度和噪声项进行边际似然估计。所有实测点用于最终模型。\n');
fprintf(fid,'- 精度用嵌套式留一交叉验证估计：每次留出 1 点，其余 15 点重新拟合；这不是独立外部测试集。\n');
fprintf(fid,'- 图中的阴影是模型的近似 95%% 预测区间，不是 SolidWorks 几何误差或实物测量不确定度。\n\n');
fprintf(fid,'## 单位与坐标/符号约定\n\n');
fprintf(fid,'长度 m，质量 kg，质心 m，惯量 kg·m²。质心处 L 张量以“坐标系1”表示。当前假设坐标系1与 URDF 大腿 link 坐标系一致；若不一致，需先旋转惯量张量并变换质心。\n\n');
fprintf(fid,'SolidWorks 的 Lxy/Lxz/Lyz 为正张量记数法下的惯性积；输出列 urdf_ixy、urdf_ixz、urdf_iyz 已取负号，其他惯量对角项不变。\n\n');
fprintf(fid,'## 物理约束检查\n\n');
fprintf(fid,'301 个预测点中有效 %d 点；最小惯量特征值 %.8g kg·m²，最小主惯量三角不等式裕量 %.8g kg·m²。整体通过：%d。\n\n', ...
    physicalTable.physically_valid_points(1),physicalTable.minimum_eigenvalue_kg_m2(1), ...
    physicalTable.minimum_triangle_margin_kg_m2(1),physicalTable.all_valid(1));
fprintf(fid,'## 交叉验证精度\n\n');
fprintf(fid,'| 输出 | 单位 | LOOCV RMSE | LOOCV MAE | 最大绝对误差 | LOOCV R² | 峰值归一化 RMSE |\n');
fprintf(fid,'|---|---|---:|---:|---:|---:|---:|\n');
for i=1:height(metricsTable)
    fprintf(fid,'| %s | %s | %.6g | %.6g | %.6g | %.6f | %.4f%% |\n', ...
        char(string(metricsTable.label_cn(i))),char(string(metricsTable.unit(i))),metricsTable.loo_rmse(i), ...
        metricsTable.loo_mae(i),metricsTable.loo_max_abs_error(i), ...
        metricsTable.loo_r2(i),metricsTable.loo_nrmse_percent_peak(i));
end
fprintf(fid,'\n## 文件\n\n');
fprintf(fid,'- DM10010_thigh_gpr_models.mat：训练好的 MATLAB GPR 模型。\n');
fprintf(fid,'- gp_dense_predictions_301.csv：区间均值、预测标准差及已转换符号的 URDF 惯性分量。\n');
fprintf(fid,'- gp_loo_cv_metrics.csv、gp_loo_predictions.csv：留一验证指标和逐点预测。\n');
fprintf(fid,'- DM10010_thigh_GP_mass_COM.png：质量与质心拟合图。\n');
fprintf(fid,'- DM10010_thigh_GP_inertia.png：质心惯量拟合图。\n');
fprintf(fid,'- fit_dm10010_thigh_gp.m：完整 MATLAB 拟合脚本。\n');
end
