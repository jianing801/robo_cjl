%% 从 CSV 批量生成电机峰值扭矩—转速包络
% 本脚本不再把 DM-J4340P 参数写死在代码里。motor_parameters.csv 每行
% 代表一个型号，enabled=1 的行会被依次计算。新增电机只需增加 CSV 行。
% 输出目录：output/<motor_id>/，不同型号不会互相覆盖。

clear; clc; close all;
scriptDir = fileparts(mfilename('fullpath'));
parameterCsv = fullfile(scriptDir, 'selected', 'motor_parameters.csv');
outputRoot = fullfile(scriptDir, 'output');
if ~isfile(parameterCsv), error('找不到参数表：%s', parameterCsv); end
if ~exist(outputRoot, 'dir'), mkdir(outputRoot); end

%% 1. 读取参数表并校验表头
P = readtable(parameterCsv, 'TextType', 'string', 'VariableNamingRule', 'preserve');
required = ["motor_id","motor_type","pole_pairs","phase_resistance_ohm", ...
    "ld_h","lq_h","flux_pm_vs","viscous_damping_nm_per_rad_s", ...
    "rated_phase_current_peak_a","rotor_inertia_kgm2","dc_bus_voltage_v", ...
    "inverter_resistance_ohm","peak_phase_current_a","gear_ratio", ...
    "peak_output_torque_nm","no_load_output_speed_rpm","enabled"];
checkColumns(P, required, parameterCsv);
if any(~isfinite(P.enabled) | ~ismember(P.enabled,[0 1]))
    error('enabled 只能填 0 或 1，不能留空。');
end
if numel(unique(P.motor_id)) ~= height(P)
    error('motor_id 不能重复，否则输出目录会被覆盖。');
end
P = P(P.enabled ~= 0, :);       % enabled=0 可保留在库中但不参与本次计算
% 可选环境变量用于只重建指定型号，保留其他型号的历史结果。
selectedIds = strtrim(split(string(getenv('DM_MOTOR_IDS')), ','));
if any(strlength(selectedIds)>0)
    if any(~ismember(selectedIds,P.motor_id))
        error('DM_MOTOR_IDS 包含不存在或 enabled=0 的型号。');
    end
    P=P(ismember(P.motor_id,selectedIds),:);
end
if isempty(P), error('参数表中没有 enabled=1 的电机。'); end

oldVisible = get(groot, 'defaultFigureVisible');
set(groot, 'defaultFigureVisible', 'off');
cleanupFigure = onCleanup(@() set(groot, 'defaultFigureVisible', oldVisible));
manifestRows = cell(height(P), 6);

%% 2. 对参数表中的每个电机执行相同流程
for k = 1:height(P)
    R = P(k, :);
    validateRow(R, k);
    motorId = R.motor_id;
    folderName = safeName(motorId);
    outDir = fullfile(outputRoot, folderName);
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    fprintf('\n[%d/%d] 计算 %s\n', k, height(P), motorId);

    %% 2.1 CSV 字段映射到 MathWorks 电机/逆变器结构体
    % 模型将电流字段解释为相电流幅值，不使用驱动最大电流替代。
    % 新型号沿用达妙参数口径及 SPMSM 近似；这些约定不是独立实测结论。
    motor.model = char(motorId);
    motor.motorType = char(R.motor_type);
    motor.p = R.pole_pairs;
    motor.Rs = R.phase_resistance_ohm;
    motor.Ld = R.ld_h;
    motor.Lq = R.lq_h;
    motor.FluxPM = R.flux_pm_vs;
    motor.B = R.viscous_damping_nm_per_rad_s;
    motor.I_rated = R.rated_phase_current_peak_a;
    motor.J = R.rotor_inertia_kgm2;
    inverter.V_dc = R.dc_bus_voltage_v;
    inverter.R_board = R.inverter_resistance_ohm;
    peakCurrent = R.peak_phase_current_a;
    gearRatio = R.gear_ratio;
    peakOutputTorque = R.peak_output_torque_nm;
    noLoadOutputRpm = R.no_load_output_speed_rpm;
    useNaturalNoLoad = ismember('model_speed_status',R.Properties.VariableNames) && ...
        string(R.model_speed_status)=="mcb_natural_48v_no_anchor";
    useUserSpeedAnchor = ismember('model_speed_status',R.Properties.VariableNames) && ...
        string(R.model_speed_status)=="user_assumed_48v_anchor";

    %% 2.2 调用 Motor Control Blockset
    % actual：dq 电压方程保留 Rs；none：控制器弱磁策略未知，不作假设。
    close all;
    raw = mcbPMSMCharacteristics(motor, inverter, ...
        'voltageEquation','actual', 'FWCMethod','none', 'imax',peakCurrent, ...
        'driveCharacteristics',1, 'constraintCurves',0);
    nativeFig = findobj('Type','figure','Tag','DriveCharacteristics');
    if ~isempty(nativeFig)
        exportgraphics(nativeFig(1), fullfile(outDir,'01_mcb_motor_characteristics.png'), ...
            'Resolution',180);
    end

    %% 2.3 电机侧结果换算到减速器输出侧
    motorSpeedRpm = raw.wArray(:);
    motorTorqueNm = real(raw.TArray(:));
    if isempty(motorSpeedRpm), error('%s 没有返回有效曲线。', motorId); end
    if motorSpeedRpm(1) > 0       % 补零速点，保证仿真静止时可查表
        motorSpeedRpm = [0; motorSpeedRpm];
        motorTorqueNm = [motorTorqueNm(1); motorTorqueNm];
    end
    outputSpeedRpm = motorSpeedRpm / gearRatio;
    % MCB 只描述电磁模型，其自然空载转速不包含厂家固件、驱动器调制及系统限制。
    % 用厂家空载转速校准速度轴，同时完整保留 MCB 曲线的无量纲形状。
    mcbNoLoadOutputRpm = outputSpeedRpm(end);
    if ~(isfinite(mcbNoLoadOutputRpm) && mcbNoLoadOutputRpm > 0)
        error('%s：MCB 返回的空载转速无效。',motorId); end
    if useNaturalNoLoad
        % No manufacturer speed is asserted: retain the 48 V model endpoint.
        noLoadOutputRpm = mcbNoLoadOutputRpm;
        warning('Damiao:UnanchoredSpeed', ...
            '%s: using MCB natural 48 V endpoint %.6f rpm; factory speed voltage unconfirmed.', ...
            motorId,noLoadOutputRpm);
    end
    speedAxisScale = noLoadOutputRpm / mcbNoLoadOutputRpm;
    outputSpeedRpm = outputSpeedRpm * speedAxisScale;
    outputSpeedRadS = outputSpeedRpm * (2*pi/60);
    idealGearTorqueNm = motorTorqueNm * gearRatio;

    % PMSM 模型不知道固件限流、减速器强度和饱和上限，因此厂家峰值是硬约束。
    peakEnvelopeNm = min(max(idealGearTorqueNm,0), peakOutputTorque);
    pointSource = repmat("MathWorks MCB",numel(outputSpeedRpm),1);
    % 强制最后一点精确等于厂家空载锚点，避免浮点误差或微小负扭矩。
    outputSpeedRpm(end) = noLoadOutputRpm;
    outputSpeedRadS(end) = noLoadOutputRpm*2*pi/60;
    motorTorqueNm(end) = 0; idealGearTorqueNm(end) = 0; peakEnvelopeNm(end) = 0;
    if useNaturalNoLoad
        pointSource(end) = "MCB natural 48 V no-load (not manufacturer verified)";
    elseif useUserSpeedAnchor
        pointSource(end) = "MCB shape + user-assumed 48 V no-load anchor";
    else
        pointSource(end) = "MCB shape + manufacturer no-load anchor";
    end

    %% 2.4 排序、去重并导出统一 CSV
    [outputSpeedRpm, order] = sort(outputSpeedRpm);
    outputSpeedRadS=outputSpeedRadS(order); motorSpeedRpm=motorSpeedRpm(order);
    motorTorqueNm=motorTorqueNm(order); idealGearTorqueNm=idealGearTorqueNm(order);
    peakEnvelopeNm=peakEnvelopeNm(order); pointSource=pointSource(order);
    [outputSpeedRpm, idx] = unique(outputSpeedRpm,'last');
    outputSpeedRadS=outputSpeedRadS(idx); motorSpeedRpm=motorSpeedRpm(idx);
    motorTorqueNm=motorTorqueNm(idx); idealGearTorqueNm=idealGearTorqueNm(idx);
    peakEnvelopeNm=peakEnvelopeNm(idx); pointSource=pointSource(idx);
    curveTable = table(outputSpeedRpm,outputSpeedRadS,motorSpeedRpm,motorTorqueNm, ...
        idealGearTorqueNm,peakEnvelopeNm,pointSource,'VariableNames', ...
        {'output_speed_rpm','output_speed_rad_s','motor_speed_rpm', ...
        'mcb_motor_torque_nm','ideal_gear_output_torque_nm','peak_envelope_nm','point_source'});
    rawCsv = fullfile(outDir,'peak_envelope.csv');
    writetable(curveTable,rawCsv);
    save(fullfile(outDir,'peak_envelope.mat'),'curveTable','motor','inverter','raw', ...
        'gearRatio','peakOutputTorque','noLoadOutputRpm','mcbNoLoadOutputRpm','speedAxisScale');

    %% 2.5 输出侧曲线图
    fig=figure('Color','w','Position',[100 100 980 620]);
    plot(outputSpeedRpm,idealGearTorqueNm,'--','LineWidth',1.4, ...
        'DisplayName','MCB result after ideal gear'); hold on;
    plot(outputSpeedRpm,peakEnvelopeNm,'-','LineWidth',2.4, ...
        'DisplayName','Peak envelope with hard cap');
    xline(noLoadOutputRpm,':','No-load speed','HandleVisibility','off');
    yline(peakOutputTorque,':','Peak torque','HandleVisibility','off');
    grid on; xlabel('Output speed (rpm)'); ylabel('Output torque (N m)');
    title(motorId+' peak torque-speed envelope','Interpreter','none');
    legend('Location','southwest');
    exportgraphics(fig,fullfile(outDir,'02_output_peak_envelope.png'),'Resolution',220);
    savefig(fig,fullfile(outDir,'02_output_peak_envelope.fig'));

    %% 2.6 保存可追溯文字报告
    report=fullfile(outDir,'model_summary.txt'); fid=fopen(report,'w');
    if fid<0, error('无法写入：%s',report); end
    fprintf(fid,'Motor: %s\nGenerated: %s\nMATLAB: %s\n',motorId,char(datetime('now')),version);
    fprintf(fid,'Vdc: %.6f V\nPeak phase current: %.6f A\n',inverter.V_dc,peakCurrent);
    fprintf(fid,'Gear ratio: %.6f\nPeak cap: %.6f N*m\nNo-load: %.6f rpm\n', ...
        gearRatio,peakOutputTorque,noLoadOutputRpm);
    fprintf(fid,'MCB natural no-load: %.6f rpm\nSpeed-axis scale: %.9f\n', ...
        mcbNoLoadOutputRpm,speedAxisScale);
    if useNaturalNoLoad
        fprintf(fid,'Speed endpoint source: unanchored MCB natural 48 V model; not factory measured.\n');
        fprintf(fid,'Factory reported speed has no voltage label and is NOT used as a 48 V anchor.\n');
    elseif useUserSpeedAnchor
        fprintf(fid,'Speed endpoint source: user-assumed 48 V no-load anchor; voltage attribution not manufacturer verified.\n');
        fprintf(fid,'Rated phase current: %.6f A; see parameter-table revision provenance.\n',motor.I_rated);
    else
        fprintf(fid,'Speed endpoint source: manufacturer 48 V no-load anchor.\n');
    end
    fprintf(fid,'Points: %d\nField weakening: disabled\n',height(curveTable)); fclose(fid);
    if ismember('model_assumptions',R.Properties.VariableNames)
        fid=fopen(report,'a');
        fprintf(fid,'Model assumptions: %s\nPhase mapping: %s\n', ...
            R.model_assumptions,R.phase_parameter_mapping_status);
        fprintf(fid,'Not a manufacturer-measured full-speed envelope.\n');
        fprintf(fid,'Inertia axis and gearbox contribution are not independently verified.\n');
        fclose(fid);
    end
    manifestRows(k,:)={motorId,folderName,height(curveTable),peakOutputTorque,noLoadOutputRpm,string(rawCsv)};
end

%% 3. 汇总所有型号
manifest=cell2table(manifestRows,'VariableNames',{'motor_id','output_folder', ...
    'point_count','peak_output_torque_nm','no_load_output_speed_rpm','raw_csv'});
manifestName='raw_envelope_manifest.csv';
if any(strlength(selectedIds)>0), manifestName='selected_raw_envelope_manifest.csv'; end
writetable(manifest,fullfile(outputRoot,manifestName));
disp(manifest); fprintf('\n完成，输出目录：%s\n',outputRoot);

%% 局部函数
function checkColumns(T,names,path)
missing=names(~ismember(names,string(T.Properties.VariableNames)));
if ~isempty(missing), error('%s 缺少字段：%s',path,strjoin(missing,', ')); end
end
function name=safeName(id)
name=regexprep(char(id),'[^A-Za-z0-9._-]','_');
if isempty(name), error('无效 motor_id：%s',id); end
end
function validateRow(R,k)
% 原表“相间”与上位机 Rs/Ls 数值一致，手册称“相参数”。用户已接受
% 按达妙同一参数体系建模，但仍需明确记录该映射及 Ld=Lq 等假设。
if ismember('phase_parameter_mapping_status',R.Properties.VariableNames) && ...
        string(R.phase_parameter_mapping_status) == "pending_line_to_line_to_phase"
    error(['%s：相间 R/L 到相 Rs、Ld/Lq 的映射未确认。原表值已存入 ' ...
        'source_line_to_line_* 列；确认测量定义后再启用该型号。'],R.motor_id);
end
if ismember('phase_parameter_mapping_status',R.Properties.VariableNames) && ...
        string(R.phase_parameter_mapping_status) == "damiao_family_phase_mapping_assumed"
    if ~ismember('model_assumptions_accepted',R.Properties.VariableNames) || ...
            R.model_assumptions_accepted~=1
        error('%s：需显式接受参数模型假设再计算。',R.motor_id);
    end
    warning('Damiao:AssumedModel','%s：使用已接受的达妙参数模型假设，非全速实测包络。',R.motor_id);
end
if ismissing(R.motor_type) || strlength(strtrim(R.motor_type)) == 0
    error('CSV 第 %d 行的 motor_type 未确认，不能计算包络。',k+1); end
names=["pole_pairs","phase_resistance_ohm","ld_h","lq_h","flux_pm_vs", ...
 "rated_phase_current_peak_a","rotor_inertia_kgm2","dc_bus_voltage_v", ...
 "peak_phase_current_a","gear_ratio","peak_output_torque_nm"];
for n=names
    v=R.(n); if ~isscalar(v)||~isfinite(v)||v<=0
        error('CSV 第 %d 行的 %s 必须为有限正数。',k+1,n); end
end
naturalMode=ismember('model_speed_status',R.Properties.VariableNames) && ...
    string(R.model_speed_status)=="mcb_natural_48v_no_anchor";
if naturalMode
    if R.dc_bus_voltage_v~=48 || ~isnan(R.no_load_output_speed_rpm)
        error('%s: natural 48 V mode requires Vdc=48 and an unset (NaN) manufacturer speed.',R.motor_id);
    end
elseif ~isfinite(R.no_load_output_speed_rpm) || R.no_load_output_speed_rpm<=0
    error('%s: a finite positive manufacturer speed or explicit natural-model mode is required.',R.motor_id);
end
if ~isfinite(R.inverter_resistance_ohm) || R.inverter_resistance_ohm<0 || ...
        ~isfinite(R.viscous_damping_nm_per_rad_s) || R.viscous_damping_nm_per_rad_s<0
    error('CSV 第 %d 行的逆变器电阻和阻尼须为有限非负数。',k+1); end
end
