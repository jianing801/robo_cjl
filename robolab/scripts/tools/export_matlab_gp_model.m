function export_matlab_gp_model(matPath, outputPath, modelId, segment, coordinateSystem, predictionDomain, measuredDomain)
% Export fitted MATLAB fitrgp models to the JSON schema consumed by the
% motor-dependent URDF predictor.  Supports independent Matern-5/2 GPs and
% fields that were constant in every SolidWorks sample.

if nargin ~= 7
    error(['Usage: export_matlab_gp_model(matPath, outputPath, modelId, ', ...
        'segment, coordinateSystem, predictionDomain, measuredDomain)']);
end
loaded = load(matPath);
if ~isfield(loaded, 'models') || ~isfield(loaded, 'properties')
    error('MAT file must contain models and properties.');
end

payload = struct();
payload.model_id = modelId;
payload.segment = segment;
payload.model = 'independent exact Gaussian processes, Matern 5/2 (MATLAB fitrgp)';
payload.coordinate_system = coordinateSystem;
payload.input_length_unit = 'm';
payload.prediction_domain_m = double(predictionDomain(:))';
payload.measured_domain_m = double(measuredDomain(:))';
payload.n_scans = numel(loaded.x) - 1;
payload.outputs = struct();

for k = 1:numel(loaded.properties)
    property = char(loaded.properties(k));
    source = loaded.models.(property);
    output = struct();
    output.label_cn = property;
    if isfield(loaded, 'labels')
        output.label_cn = char(loaded.labels(k));
    end
    output.unit = source.units;
    output.constant = logical(source.isConstant);
    if output.constant
        output.constant_value = double(source.yCenter);
        output.x_mean = 0;
        output.x_scale = 1;
        output.y_mean = double(source.yCenter);
        output.y_scale = 1;
        output.length_scale_standardized = 1;
        output.signal_std_standardized = 0;
        output.noise_std_standardized = 0;
        output.beta_standardized = 0;
        output.x_train_m = [];
        output.alpha_standardized = [];
    else
        model = source.model;
        kernelParameters = double(model.KernelInformation.KernelParameters(:));
        if numel(kernelParameters) ~= 2
            error('Expected Matern 5/2 [lengthScale, signalStd] parameters for %s.', property);
        end
        output.constant_value = [];
        output.x_mean = double(model.PredictorLocation(1));
        output.x_scale = double(model.PredictorScale(1));
        output.y_mean = double(source.yCenter);
        output.y_scale = double(source.yScale);
        output.length_scale_standardized = kernelParameters(1);
        output.signal_std_standardized = kernelParameters(2);
        output.noise_std_standardized = double(model.Sigma);
        output.beta_standardized = double(model.Beta(1));
        output.x_train_m = double(model.X(:,1))';
        output.alpha_standardized = double(model.Alpha(:))';
    end
    payload.outputs.(property) = output;
end

outDir = fileparts(outputPath);
if ~isempty(outDir) && ~isfolder(outDir), mkdir(outDir); end
jsonText = jsonencode(payload, 'PrettyPrint', true);
fid = fopen(outputPath, 'w', 'n', 'UTF-8');
if fid < 0, error('Unable to open output JSON: %s', outputPath); end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, jsonText, 'char');
fprintf('Exported %s to %s\n', modelId, outputPath);
end
