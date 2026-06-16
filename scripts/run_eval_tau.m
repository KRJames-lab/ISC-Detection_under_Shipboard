%% run_eval_tau.m
% 평가 전용 τ sweep 시뮬레이션
% 학습에 사용되지 않은 τ값으로 AI 모델 일반화 테스트용
% τ = [100, 1000] → Module only, 3시나리오 × 5 k_R = 30런

%% ===== 설정 =====
basePath = fullfile(fileparts(mfilename('fullpath')), '..');
dataPath = fullfile(basePath, 'data');

% 평가용 τ (학습 τ={50,300,3600}과 겹치지 않음)
tau_values = [100, 1000];

% ISC 파라미터 (기존과 동일)
R_init  = 500;    % Ohm
R_final = 5;      % Ohm
t_onset = 200;    % s

% 시뮬 기본 설정
k_R_values = [0.05, 0.1, 0.2, 0.5, 1.0] * 1e-3;  % Ohm/g
I_load  = -2.9;   % A (low-load standby, ~0.04C at module level)
SOC_init = 0.9;

% 시나리오: 3개 (무진동, MIL-STD, MSS-Head)
scenarios_to_run = [1, 2, 4];  % NoVib / MIL-STD / MSS+MIL-STD Head (Head only)

%% ===== 데이터 로드 =====
vd = load(fullfile(dataPath, 'vibration_data_all.mat'));
vd = vd.vibration_data;

%% ===== 모듈 모델 로드 (Module only) =====
mdl = 'ShipBattery_Module';
load_system(fullfile(basePath, 'models', [mdl '.slx']));

mi = 2;  % Module index (기존 naming 호환)

%% ===== Sweep 루프 =====
total = length(tau_values) * length(scenarios_to_run) * length(k_R_values);
results = struct();
run_count = 0;

% 부하/SOC 설정
set_param([mdl '/LoadConst'], 'Value', num2str(I_load));
set_param([mdl '/Battery'], 'stateOfCharge', num2str(SOC_init));

for ti = 1:length(tau_values)
    tau = tau_values(ti);
    fprintf('\n===== tau = %d s =====\n', tau);

    for si = 1:length(scenarios_to_run)
        sc_idx = scenarios_to_run(si);
        sc = vd.scenarios(sc_idx);

        % 시뮬 시간: R_ISC가 R_final(=5Ω)에 근접하도록 8τ만큼 진행
        % (8τ 후 R_ISC ≈ 5 + 495*exp(-8) ≈ 5.17 Ω; 모든 R_deadline∈{10,15,25,50}에 충분)
        % min: vibration trace가 8τ보다 길면 8τ까지만, 짧으면 trace 끝까지
        T_end = min(sc.t(end), t_onset + 8*tau);
        dt = 0.1;
        t_sim = (0:dt:T_end)';

        % 진동 데이터 (필요시 반복 확장)
        if T_end > sc.t(end)
            n_repeat = ceil(T_end / sc.t(end));
            a_ext = repmat(sc.a_total, n_repeat, 1);
            t_ext = (0:dt:(length(a_ext)-1)*dt)';
            a_ext = a_ext(1:length(t_sim));
            t_ext = t_sim;
        else
            t_ext = sc.t;
            a_ext = sc.a_total;
        end

        vib_ts = timeseries(a_ext, t_ext, 'Name', 'vibration');
        assignin('base', 'vib_ts', vib_ts);

        % ISC timeseries: R_ISC(t) = R_final + (R_init-R_final)*exp(-(t-t_onset)/tau)
        R_isc = ones(size(t_sim)) * 1e6;
        idx = t_sim >= t_onset;
        R_isc(idx) = R_final + (R_init - R_final) * exp(-(t_sim(idx) - t_onset) / tau);
        isc_ts = timeseries(R_isc, t_sim, 'Name', 'isc');
        assignin('base', 'isc_ts', isc_ts);

        % 시뮬 시간 설정
        set_param(mdl, 'StopTime', num2str(T_end), 'MaxStep', '0.1');

        for ki = 1:length(k_R_values)
            k_R = k_R_values(ki);
            run_count = run_count + 1;

            set_param([mdl '/kR_Gain'], 'Gain', num2str(k_R, '%.10e'));

            fprintf('[%d/%d] Module | case%d(%s) | tau=%ds | kR=%.2f mOhm/g ... ', ...
                run_count, total, sc_idx, sc.case_label, tau, k_R*1e3);

            tic;
            simOut = sim(mdl);
            elapsed = toc;
            fprintf('%.1fs\n', elapsed);

            % 결과 저장 (기존 naming 호환: m2_s{sc}_k{ki}_tau{tau})
            r = struct();
            r.model = 'Module_8S24P';
            r.case_id = sc_idx;
            r.case_label = sc.case_label;
            r.tau = tau;
            r.k_R = k_R;
            r.V = squeeze(simOut.V_terminal.Data);
            r.T = squeeze(simOut.T_cell.Data);
            r.t = squeeze(simOut.V_terminal.Time);
            r.V_end = r.V(end);
            r.T_end = r.T(end) - 273.15;

            fn = sprintf('m%d_s%02d_k%d_tau%d', mi, sc_idx, ki, tau);
            results.(fn) = r;
        end
    end
end

%% ===== 저장 =====
outFile = fullfile(dataPath, 'eval_tau_results.mat');
save(outFile, 'results', 'tau_values', 'k_R_values', 'scenarios_to_run', ...
     'R_init', 'R_final', 't_onset', '-v7.3');
fprintf('\nSaved: %s\n', outFile);
fprintf('Total runs: %d\n', run_count);
