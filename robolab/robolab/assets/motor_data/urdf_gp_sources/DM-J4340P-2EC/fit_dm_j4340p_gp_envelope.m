%% 从 CSV 批量生成约束 GP 扭矩—转速包络
% 先运行 run_dm_j4340p_peak_envelope.m。本脚本遍历参数表中 enabled=1
% 的型号，自动读取 output/<motor_id>/peak_envelope.csv。峰值平台是硬限幅，
% 只对平台后的下降段拟合 Matérn 3/2 GP，避免转折点被普通 GP 错误抹平。

clear; clc; close all; rng(42,'twister');
scriptDir=fileparts(mfilename('fullpath'));
parameterCsv=fullfile(scriptDir,'selected','motor_parameters.csv');
outputRoot=fullfile(scriptDir,'output');
if ~isfile(parameterCsv), error('找不到参数表：%s',parameterCsv); end
P=readtable(parameterCsv,'TextType','string','VariableNamingRule','preserve');
required=["motor_id","peak_output_torque_nm","no_load_output_speed_rpm", ...
    "gp_noise_ratio","gp_export_point_count","gp_cv_folds","enabled"];
checkColumns(P,required,parameterCsv);
if any(~isfinite(P.enabled) | ~ismember(P.enabled,[0 1]))
    error('enabled 只能填 0 或 1，不能留空。');
end
if numel(unique(P.motor_id)) ~= height(P)
    error('motor_id 不能重复，否则输出目录会被覆盖。');
end
P=P(P.enabled~=0,:);
selectedIds=strtrim(split(string(getenv('DM_MOTOR_IDS')),','));
if any(strlength(selectedIds)>0)
    if any(~ismember(selectedIds,P.motor_id))
        error('DM_MOTOR_IDS 包含不存在或 enabled=0 的型号。');
    end
    P=P(ismember(P.motor_id,selectedIds),:);
end
if isempty(P), error('没有 enabled=1 的电机。'); end
oldVisible=get(groot,'defaultFigureVisible'); set(groot,'defaultFigureVisible','off');
cleanupFigure=onCleanup(@() set(groot,'defaultFigureVisible',oldVisible));
manifestRows=cell(height(P),10);

for k=1:height(P)
    R=P(k,:); validateGpRow(R,k); motorId=R.motor_id; folderName=safeName(motorId);
    outDir=fullfile(outputRoot,folderName);
    inputCsv=fullfile(outDir,'peak_envelope.csv');
    outputCsv=fullfile(outDir,'gp_envelope.csv');
    if ~isfile(inputCsv)
        error('%s 缺少原始包络，请先运行第一个脚本：%s',motorId,inputCsv); end
    fprintf('\n[%d/%d] GP 拟合 %s\n',k,height(P),motorId);

    %% 1. 读取原始曲线
    raw=readtable(inputCsv,'TextType','string','VariableNamingRule','preserve');
    checkColumns(raw,["output_speed_rpm","output_speed_rad_s","peak_envelope_nm"],inputCsv);
    xRpm=raw.output_speed_rpm(:); xRadS=raw.output_speed_rad_s(:); yNm=raw.peak_envelope_nm(:);
    peakNm=R.peak_output_torque_nm; noLoadRpm=R.no_load_output_speed_rpm;
    useNaturalNoLoad=ismember('model_speed_status',R.Properties.VariableNames) && ...
        string(R.model_speed_status)=="mcb_natural_48v_no_anchor";
    useUserSpeedAnchor=ismember('model_speed_status',R.Properties.VariableNames) && ...
        string(R.model_speed_status)=="user_assumed_48v_anchor";
    if useNaturalNoLoad, noLoadRpm=xRpm(end); end
    if ~isfinite(noLoadRpm)||noLoadRpm<=0||abs(xRpm(end)-noLoadRpm)>max(1e-8,noLoadRpm*1e-10)
        error('%s: speed endpoint does not match the declared model mode.',motorId); end
    noiseRatio=R.gp_noise_ratio;
    sigma=max(1e-3,noiseRatio*peakNm);
    denseCount=round(R.gp_export_point_count);
    cvFolds=round(R.gp_cv_folds);
    if numel(xRpm)<3||any(diff(xRpm)<=0)||abs(xRpm(1))>1e-12
        error('%s：原始曲线必须从零开始、严格递增且至少有 3 点。',motorId); end
    if any(~isfinite(yNm))||any(yNm<0)||any(yNm>peakNm+1e-8)
        error('%s：扭矩必须为有限值并位于 [0, 峰值]。',motorId); end

    %% 2. 自动找到模型实际恒扭矩平台的最后一点
    % 厂家峰值是安全硬上限，但电磁模型在给定电流下未必恰好达到该数值。
    % 因此用原始包络的最大值识别平台，再继续用厂家峰值做上限检查。
    plateauNm=max(yNm);
    tol=max(1e-9,plateauNm*1e-9);
    cornerIdx=find(abs(yNm-plateauNm)<=tol,1,'last');
    if isempty(cornerIdx)||cornerIdx>=numel(xRpm)
        error('%s：无法识别平台及其后的下降段。',motorId); end
    cornerRpm=xRpm(cornerIdx); cornerRadS=xRadS(cornerIdx);
    xFit=xRadS(cornerIdx:end); yFit=yNm(cornerIdx:end);
    if numel(xFit)<max(6,cvFolds)
        error('%s：下降段 %d 点不足以进行 %d 折验证。',motorId,numel(xFit),cvFolds); end

    %% 3. GP 拟合
    % Matérn 3/2 比平方指数核更允许有限曲率。各型号统一使用 CSV 中的
    % 相对噪声比例，Sigma=峰值扭矩*比例，并保留 1e-3 N*m 数值下限。
    gpr=fitrgp(xFit,yFit,'BasisFunction','constant','KernelFunction','matern32', ...
        'Standardize',true,'FitMethod','exact','PredictMethod','exact', ...
        'Sigma',sigma,'SigmaLowerBound',max(1e-9,sigma*1e-6),'ConstantSigma',true);

    %% 4. 高密度预测，并强制物理约束
    % unique 同时保留均匀网格、全部原始点和精确转折点。
    outputSpeedRpm=unique([linspace(0,noLoadRpm,denseCount)';xRpm;cornerRpm]);
    outputSpeedRadS=outputSpeedRpm*(2*pi/60);
    gpMeanRawNm=plateauNm*ones(size(outputSpeedRpm)); gpStdNm=zeros(size(outputSpeedRpm));
    descending=outputSpeedRpm>cornerRpm;
    [gpMeanRawNm(descending),gpStdNm(descending)]=predict(gpr,outputSpeedRadS(descending));
    % 普通 GP 不保证非负、单调或峰值上限，所以必须做以下投影。
    peakEnvelopeNm=min(max(gpMeanRawNm,0),peakNm);
    peakEnvelopeNm(outputSpeedRpm<=cornerRpm)=plateauNm;
    peakEnvelopeNm=cummin(peakEnvelopeNm); peakEnvelopeNm(end)=0;
    lower95=min(max(gpMeanRawNm-1.96*gpStdNm,0),peakNm);
    upper95=min(max(gpMeanRawNm+1.96*gpStdNm,0),peakNm);
    lower95(outputSpeedRpm<=cornerRpm)=plateauNm; upper95(outputSpeedRpm<=cornerRpm)=plateauNm;
    source=repmat("Constrained GP mean",numel(outputSpeedRpm),1);
    source(outputSpeedRpm<=cornerRpm)="Current-limited model plateau";
    if useNaturalNoLoad
        source(end)="MCB natural 48 V no-load (not manufacturer verified)";
    elseif useUserSpeedAnchor
        source(end)="User-assumed 48 V no-load anchor (not manufacturer verified)";
    else
        source(end)="Manufacturer no-load anchor";
    end

    %% 5. 导出 Python 动态执行器可直接读取的 CSV
    % 运行时只使用 output_speed_rad_s 和 peak_envelope_nm，其他列用于诊断。
    gpTable=table(outputSpeedRpm,outputSpeedRadS,peakEnvelopeNm,gpMeanRawNm,gpStdNm, ...
        lower95,upper95,source,'VariableNames',{'output_speed_rpm','output_speed_rad_s', ...
        'peak_envelope_nm','gp_mean_unconstrained_nm','gp_std_nm','gp_lower95_nm', ...
        'gp_upper95_nm','point_source'});
    writetable(gpTable,outputCsv);

    %% 6. 训练误差和交叉验证误差
    trainPrediction=predict(gpr,xFit);
    trainRmse=sqrt(mean((trainPrediction-yFit).^2));
    trainMax=max(abs(trainPrediction-yFit));
    cvModel=crossval(gpr,'KFold',cvFolds); cvPrediction=kfoldPredict(cvModel);
    cvRmse=sqrt(mean((cvPrediction-yFit).^2)); cvMax=max(abs(cvPrediction-yFit));

    %% 7. 绘制原始点、约束均值和 95%% 区间
    close all; fig=figure('Color','w','Position',[100 100 980 620]);
    fill([outputSpeedRpm;flipud(outputSpeedRpm)],[lower95;flipud(upper95)], ...
        [0.80 0.88 1.00],'EdgeColor','none','FaceAlpha',0.45,'DisplayName','GP 95% interval'); hold on;
    plot(xRpm,yNm,'ko','MarkerSize',4.5,'MarkerFaceColor','k', ...
        'DisplayName',sprintf('Original points (%d)',height(raw)));
    plot(outputSpeedRpm,peakEnvelopeNm,'b-','LineWidth',2.4, ...
        'DisplayName',sprintf('Constrained GP (%d)',height(gpTable)));
    xline(cornerRpm,':',sprintf('Corner %.3f rpm',cornerRpm),'HandleVisibility','off');
    yline(peakNm,':','Manufacturer peak limit','HandleVisibility','off'); grid on;
    xlabel('Output speed (rpm)'); ylabel('Output torque (N m)');
    title(motorId+' constrained GP envelope','Interpreter','none'); legend('Location','southwest');
    xlim([0,1.02*noLoadRpm]); ylim([0,1.08*peakNm]);
    exportgraphics(fig,fullfile(outDir,'03_gp_envelope.png'),'Resolution',220);
    savefig(fig,fullfile(outDir,'03_gp_envelope.fig'));
    save(fullfile(outDir,'gp_envelope.mat'),'gpTable','gpr','raw','plateauNm','cornerRpm', ...
        'cornerRadS','trainRmse','trainMax','cvRmse','cvMax');

    %% 8. 保存报告和批处理清单
    report=fullfile(outDir,'gp_summary.txt'); fid=fopen(report,'w');
    if fid<0, error('无法写入：%s',report); end
    fprintf(fid,'Motor: %s\nGenerated: %s\nInput points: %d\n',motorId,char(datetime('now')),height(raw));
    fprintf(fid,'GP training points: %d\nExport points: %d\nKernel: Matern 3/2\n',numel(xFit),height(gpTable));
    fprintf(fid,'Noise ratio: %.9f\nSigma: %.9f N*m\nModel plateau: %.9f N*m\nManufacturer peak limit: %.9f N*m\nCorner: %.9f rpm\nTraining RMSE: %.9f N*m\n', ...
        noiseRatio,sigma,plateauNm,peakNm,cornerRpm,trainRmse);
    fprintf(fid,'%d-fold CV RMSE: %.9f N*m\nCV max error: %.9f N*m\n',cvFolds,cvRmse,cvMax);
    if useNaturalNoLoad
        fprintf(fid,'Speed endpoint: %.9f rpm; unanchored 48 V MCB model, not manufacturer measured.\n',noLoadRpm);
    elseif useUserSpeedAnchor
        fprintf(fid,'Speed endpoint: %.9f rpm; user-assumed 48 V anchor, voltage attribution not manufacturer verified.\n',noLoadRpm);
    end
    fclose(fid);
    manifestRows(k,:)={motorId,folderName,height(raw),numel(xFit),height(gpTable), ...
        cornerRpm,trainRmse,cvRmse,cvMax,string(outputCsv)};
    fprintf('完成：训练 RMSE %.6g，%d 折 CV RMSE %.6g N*m。\n',trainRmse,cvFolds,cvRmse);
end

manifest=cell2table(manifestRows,'VariableNames',{'motor_id','output_folder', ...
    'raw_point_count','gp_training_point_count','export_point_count','corner_speed_rpm', ...
    'training_rmse_nm','cv_rmse_nm','cv_max_abs_nm','gp_csv'});
manifestName='gp_envelope_manifest.csv';
if any(strlength(selectedIds)>0), manifestName='selected_gp_envelope_manifest.csv'; end
writetable(manifest,fullfile(outputRoot,manifestName));
disp(manifest); fprintf('\n全部 GP 拟合完成：%s\n',outputRoot);

%% 局部函数
function checkColumns(T,names,path)
missing=names(~ismember(names,string(T.Properties.VariableNames)));
if ~isempty(missing), error('%s 缺少字段：%s',path,strjoin(missing,', ')); end
end
function name=safeName(id)
name=regexprep(char(id),'[^A-Za-z0-9._-]','_');
if isempty(name), error('无效 motor_id：%s',id); end
end
function validateGpRow(R,k)
names=["peak_output_torque_nm","gp_noise_ratio", ...
    "gp_export_point_count","gp_cv_folds"];
for n=names
    v=R.(n); if ~isscalar(v)||~isfinite(v)||v<=0
        error('CSV 第 %d 行的 %s 必须为有限正数。',k+1,n); end
end
naturalMode=ismember('model_speed_status',R.Properties.VariableNames) && ...
    string(R.model_speed_status)=="mcb_natural_48v_no_anchor";
if ~naturalMode && (~isfinite(R.no_load_output_speed_rpm)||R.no_load_output_speed_rpm<=0)
    error('%s: missing manufacturer speed without explicit natural-model mode.',R.motor_id);
end
if R.gp_export_point_count<20||R.gp_cv_folds<2
    error('CSV 第 %d 行：导出点数至少 20，交叉验证至少 2 折。',k+1); end
if R.gp_noise_ratio>=1
    error('CSV 第 %d 行：gp_noise_ratio 必须小于 1。',k+1); end
end
