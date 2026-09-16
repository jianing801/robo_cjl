%% Constrained GP fit for the DM-J4340P-2EC peak torque-speed envelope
% The 27 N*m constant-torque region is a known hard cap. A Gaussian
% process is fitted only to the voltage-limited descending region, avoiding
% artificial smoothing across the physical corner-speed kink.

clear; clc; close all;
rng(42, 'twister');

scriptDir = fileparts(mfilename('fullpath'));
outputDir = fullfile(scriptDir, 'output');
inputCsv = fullfile(outputDir, 'DM_J4340P_peak_envelope.csv');
% The standalone MATLAB project stores generated data in ./output, while the
% Python package keeps this script beside its CSV files.
if ~isfile(inputCsv)
    outputDir = scriptDir;
    inputCsv = fullfile(outputDir, 'DM_J4340P_peak_envelope.csv');
end
outputCsv = fullfile(outputDir, 'DM_J4340P_gp_envelope.csv');

manufacturerPeakNm = 27.0;
manufacturerNoLoadRpm = 100.0;
raw = readtable(inputCsv, 'VariableNamingRule', 'preserve');
xRpm = raw.output_speed_rpm(:);
xRadS = raw.output_speed_rad_s(:);
yNm = raw.peak_envelope_nm(:);

if any(diff(xRpm) <= 0) || xRpm(1) ~= 0 || abs(xRpm(end) - manufacturerNoLoadRpm) > 1e-9
    error('Input curve must be strictly increasing from 0 to 100 rpm.');
end
if any(yNm < 0) || any(yNm > manufacturerPeakNm + 1e-9)
    error('Input torque values must lie in [0, 27] N*m.');
end

cornerIdx = find(abs(yNm - manufacturerPeakNm) <= 1e-9, 1, 'last');
cornerRpm = xRpm(cornerIdx);
cornerRadS = xRadS(cornerIdx);
xFit = xRadS(cornerIdx:end);
yFit = yNm(cornerIdx:end);

% Matern 3/2 is intentionally less smooth than a squared-exponential
% kernel and better represents the finite-curvature voltage-limited branch.
% Source points are deterministic model output, so observation noise is
% kept small and fixed; kernel parameters are estimated by likelihood.
gpr = fitrgp( ...
    xFit, yFit, ...
    'BasisFunction', 'constant', ...
    'KernelFunction', 'matern32', ...
    'Standardize', true, ...
    'FitMethod', 'exact', ...
    'PredictMethod', 'exact', ...
    'Sigma', 0.03, ...
    'SigmaLowerBound', 1e-6, ...
    'ConstantSigma', true);

% Dense runtime grid, retaining every original point and exact corner.
denseRpm = linspace(0, manufacturerNoLoadRpm, 401)';
outputSpeedRpm = unique([denseRpm; xRpm; cornerRpm]);
outputSpeedRadS = outputSpeedRpm .* (2*pi/60);

gpMeanRawNm = manufacturerPeakNm .* ones(size(outputSpeedRpm));
gpStdNm = zeros(size(outputSpeedRpm));
descending = outputSpeedRpm > cornerRpm;
[gpMeanRawNm(descending), gpStdNm(descending)] = predict(gpr, outputSpeedRadS(descending));

% Project the statistical mean onto a physically admissible hard envelope.
peakEnvelopeNm = min(max(gpMeanRawNm, 0), manufacturerPeakNm);
peakEnvelopeNm(outputSpeedRpm <= cornerRpm) = manufacturerPeakNm;
peakEnvelopeNm = cummin(peakEnvelopeNm);
peakEnvelopeNm(end) = 0;

gpLower95Nm = min(max(gpMeanRawNm - 1.96 .* gpStdNm, 0), manufacturerPeakNm);
gpUpper95Nm = min(max(gpMeanRawNm + 1.96 .* gpStdNm, 0), manufacturerPeakNm);
gpLower95Nm(outputSpeedRpm <= cornerRpm) = manufacturerPeakNm;
gpUpper95Nm(outputSpeedRpm <= cornerRpm) = manufacturerPeakNm;

pointSource = repmat("Constrained GP mean", numel(outputSpeedRpm), 1);
pointSource(outputSpeedRpm <= cornerRpm) = "Manufacturer peak plateau";
pointSource(end) = "Manufacturer no-load anchor";

gpTable = table( ...
    outputSpeedRpm, outputSpeedRadS, peakEnvelopeNm, gpMeanRawNm, gpStdNm, ...
    gpLower95Nm, gpUpper95Nm, pointSource, ...
    'VariableNames', { ...
    'output_speed_rpm', 'output_speed_rad_s', 'peak_envelope_nm', ...
    'gp_mean_unconstrained_nm', 'gp_std_nm', 'gp_lower95_nm', ...
    'gp_upper95_nm', 'point_source'});
writetable(gpTable, outputCsv);

% Diagnostics are computed only on the branch learned by the GP.
[trainMean, ~] = predict(gpr, xFit);
trainRmseNm = sqrt(mean((trainMean - yFit).^2));
trainMaxAbsNm = max(abs(trainMean - yFit));
cvModel = crossval(gpr, 'KFold', 5);
cvPrediction = kfoldPredict(cvModel);
cvRmseNm = sqrt(mean((cvPrediction - yFit).^2));
cvMaxAbsNm = max(abs(cvPrediction - yFit));

fig = figure('Color', 'w', 'Position', [100 100 980 620]);
fill([outputSpeedRpm; flipud(outputSpeedRpm)], ...
     [gpLower95Nm; flipud(gpUpper95Nm)], ...
     [0.80 0.88 1.00], 'EdgeColor', 'none', ...
     'FaceAlpha', 0.45, 'DisplayName', 'GP 95% interval');
hold on;
plot(xRpm, yNm, 'ko', 'MarkerSize', 4.5, 'MarkerFaceColor', 'k', ...
    'DisplayName', sprintf('Original MATLAB points (%d)', height(raw)));
plot(outputSpeedRpm, peakEnvelopeNm, 'b-', 'LineWidth', 2.4, ...
    'DisplayName', sprintf('Constrained GP envelope (%d points)', height(gpTable)));
xline(cornerRpm, ':', sprintf('Corner %.3f rpm', cornerRpm), ...
    'HandleVisibility', 'off');
yline(manufacturerPeakNm, ':', 'Peak torque 27 N m', ...
    'HandleVisibility', 'off');
grid on;
xlabel('Actuator output speed (rpm)');
ylabel('Actuator output torque (N m)');
title('DM-J4340P-2EC constrained Gaussian-process envelope');
legend('Location', 'southwest');
xlim([0 102]);
ylim([0 29]);
exportgraphics(fig, fullfile(outputDir, '03_DM_J4340P_GP_envelope.png'), ...
    'Resolution', 220);
savefig(fig, fullfile(outputDir, '03_DM_J4340P_GP_envelope.fig'));

save(fullfile(outputDir, 'DM_J4340P_gp_envelope.mat'), ...
    'gpTable', 'gpr', 'raw', 'cornerRpm', 'cornerRadS', ...
    'trainRmseNm', 'trainMaxAbsNm', 'cvRmseNm', 'cvMaxAbsNm');

summaryFile = fullfile(outputDir, 'DM_J4340P_gp_summary.txt');
fid = fopen(summaryFile, 'w');
cleanupObj = onCleanup(@() fclose(fid));
fprintf(fid, 'DM-J4340P-2EC constrained GP envelope\n');
fprintf(fid, 'Generated: %s\n', char(datetime('now')));
fprintf(fid, 'Input points: %d\n', height(raw));
fprintf(fid, 'Descending-branch GP training points: %d\n', numel(xFit));
fprintf(fid, 'Exported lookup points: %d\n', height(gpTable));
fprintf(fid, 'Kernel: Matern 3/2\n');
fprintf(fid, 'Fixed observation sigma: 0.03 N*m\n');
fprintf(fid, 'Corner speed: %.6f rpm (%.9f rad/s)\n', cornerRpm, cornerRadS);
fprintf(fid, 'Training RMSE: %.9f N*m\n', trainRmseNm);
fprintf(fid, 'Training max absolute error: %.9f N*m\n', trainMaxAbsNm);
fprintf(fid, '5-fold CV RMSE: %.9f N*m\n', cvRmseNm);
fprintf(fid, '5-fold CV max absolute error: %.9f N*m\n', cvMaxAbsNm);
fprintf(fid, 'Runtime curve: constrained GP mean (not confidence lower bound)\n');
fprintf(fid, 'Hard projections: [0,27] N*m, non-increasing, 100 rpm=0\n');

fprintf('Input points: %d\n', height(raw));
fprintf('GP branch points: %d\n', numel(xFit));
fprintf('Exported points: %d\n', height(gpTable));
fprintf('Training RMSE: %.6g N*m\n', trainRmseNm);
fprintf('5-fold CV RMSE: %.6g N*m\n', cvRmseNm);
fprintf('Generated: %s\n', outputCsv);
