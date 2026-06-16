%% run_simulation.m
% Ship Battery 시뮬레이션 실행 스크립트
% - 10개 시나리오 × k_R 민감도 분석 지원
% - 결과를 구조체로 저장

%% ===== 설정 =====
mdl = 'ShipBattery_Model';
dataPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data');

% k_R 민감도 분석 범위 (mOhm/g → Ohm/g)
k_R_values = [0.05, 0.1, 0.2, 0.5, 1.0] * 1e-3;  % Ohm/g

% 실행할 시나리오 선택 (1~10, 또는 'all')
scenarios_to_run = [1, 2, 4];  % 1=무진동, 2=MIL-STD, 4=MSS+MIL-STD Head (Head only)

% 부하 전류 (음수 = 방전)
I_load = -2.9;  % A (1C)
SOC_init = 0.9;

%% ===== 데이터 로드 =====
d = load(fullfile(dataPath, 'vibration_data_all.mat'));
vd = d.vibration_data;

%% ===== 모델 로드 =====
modelFile = fullfile(fileparts(mfilename('fullpath')), '..', 'models', [mdl '.slx']);
load_system(modelFile);

% 부하 전류 설정
set_param([mdl '/LoadConst'], 'Value', num2str(I_load));
set_param([mdl '/Battery'], 'stateOfCharge', num2str(SOC_init));

%% ===== 시뮬레이션 루프 =====
results = struct();
total_runs = length(scenarios_to_run) * length(k_R_values);
run_count = 0;

for si = 1:length(scenarios_to_run)
    sc_idx = scenarios_to_run(si);
    sc = vd.scenarios(sc_idx);

    % 진동 데이터 → timeseries
    vib_ts = timeseries(sc.a_total, sc.t, 'Name', 'vibration');
    assignin('base', 'vib_ts', vib_ts);

    % 시뮬레이션 시간
    set_param(mdl, 'StopTime', num2str(sc.t(end)), 'MaxStep', '0.01');

    for ki = 1:length(k_R_values)
        k_R = k_R_values(ki);
        run_count = run_count + 1;

        % k_R 설정
        set_param([mdl '/kR_Gain'], 'Gain', num2str(k_R, '%.10e'));

        % 실행
        fprintf('[%d/%d] Case %d (%s) | k_R=%.2f mOhm/g ... ', ...
            run_count, total_runs, sc.case_id, sc.case_label, k_R*1e3);
        tic;
        simOut = sim(mdl);
        t_elapsed = toc;
        fprintf('%.1fs\n', t_elapsed);

        % 결과 저장
        V = simOut.V_terminal;
        I = simOut.I_battery;
        T = simOut.T_cell;

        r = struct();
        r.case_id = sc.case_id;
        r.case_label = sc.case_label;
        r.beta_label = sc.beta_label;
        r.k_R = k_R;
        r.t = squeeze(V.Time);
        r.V = squeeze(V.Data);
        r.I = squeeze(I.Data);
        r.T = squeeze(T.Data);
        r.a_max_g = sc.a_max_g;
        r.a_rms_g = sc.a_rms_g;

        fieldName = sprintf('case%d_kR%d', sc_idx, round(k_R*1e6));
        results.(fieldName) = r;
    end
end

%% ===== 결과 저장 =====
outFile = fullfile(dataPath, 'simulation_results.mat');
save(outFile, 'results', 'k_R_values', 'scenarios_to_run', 'I_load', 'SOC_init');
fprintf('\nResults saved to %s\n', outFile);

%% ===== 요약 출력 =====
fprintf('\n=== Simulation Summary ===\n');
fn = fieldnames(results);
for i = 1:length(fn)
    r = results.(fn{i});
    V_ripple = std(r.V - movmean(r.V, round(1.0/mean(diff(r.t))))) * 1000;
    fprintf('  %-25s k_R=%.2f mOhm/g | V: %.3f->%.3f | T: %.2f->%.2f°C | ripple=%.2f mV\n', ...
        [r.case_label '(' r.beta_label ')'], r.k_R*1e3, ...
        r.V(1), r.V(end), r.T(1)-273.15, r.T(end)-273.15, V_ripple);
end
