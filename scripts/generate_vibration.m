%% generate_vibration.m
% 선박 진동 가속도 합성 — 분리 분석
%
% Case 1: No vibration (baseline)              — 1개
% Case 2: MIL-STD only (고주파만, 4~33Hz)      — 1개
% Case 3: MSS only (저주파만)                   — 4방향
% Case 4: MSS + MIL-STD (전체)                  — 4방향
% → 총 10개 시나리오
%
% 출력: data/vibration_data.mat
%
% 사용법:
%   >> cd 'D:\01_Projects\05_Ship_Battery\scripts'
%   >> generate_vibration

clear waveMotionRAO;
clearvars;
close all;

%% ========================================================================
%  USER SETTINGS
%  ========================================================================

Hs = 4.0;                      % 유의파고 (m)
Tz = 8.0;                      % 영교차 주기 (s)
U_ship = 5;                    % 선속 (m/s)

% Head sea (180°) only — 본 연구는 MSS-Head만 evaluation에 사용
% (Following 0.04g / Beam 0.16g / Head 0.18g 중 최대 가속도 케이스)
beta_list_deg = [180];
beta_labels = {'Head (180)'};

milstd_scale = 1.0;
k_R = 0.5;                     % [mΩ/g]

Fs = 100;
h = 1/Fs;
T_final = 29030;       % 200(t_onset) + 8*tau_max(3600) + T_transient(30) = 29030
T_transient = 30;
g_const = 9.81;

disp('============================================================');
disp('  선박 진동 가속도 합성 — 분리 분석');
disp('  Case 1,2: 각 1개 / Case 3,4: 각 4방향 = 총 10 시나리오');
disp('============================================================');
fprintf('  Hs=%.1fm, Tz=%.1fs, U=%.1f m/s\n', Hs, Tz, U_ship);
fprintf('  MIL-STD 스케일: %.1f, k_R: %.3f mOhm/g\n', milstd_scale, k_R);
fprintf('  샘플링: %d Hz, 유효 시간: %d s\n', Fs, T_final - T_transient);
disp('------------------------------------------------------------');

%% ========================================================================
%  시간 벡터
%  ========================================================================
t_vib = (0:h:T_final-T_transient-1)';
N = length(t_vib);

%% ========================================================================
%  PART 1: MIL-STD-810H 고주파 (1회)
%  ========================================================================
disp('[1/3] MIL-STD-810H 고주파 진동 합성...');

freq_milstd = 4:1:33;
disp_amplitude_m = zeros(size(freq_milstd));
for idx = 1:length(freq_milstd)
    f = freq_milstd(idx);
    if f <= 15
        disp_amplitude_m(idx) = 0.030 * 0.0254;
    elseif f <= 25
        disp_amplitude_m(idx) = 0.020 * 0.0254;
    else
        disp_amplitude_m(idx) = 0.010 * 0.0254;
    end
end

accel_amplitude_g = disp_amplitude_m .* (2*pi*freq_milstd).^2 / g_const;

rng(42);
random_phases = 2*pi*rand(size(freq_milstd));

a_MILSTD = zeros(N, 1);
for idx = 1:length(freq_milstd)
    a_MILSTD = a_MILSTD + milstd_scale * accel_amplitude_g(idx) * ...
        sin(2*pi*freq_milstd(idx)*t_vib + random_phases(idx));
end

fprintf('  MIL-STD: max=%.4f g, RMS=%.4f g\n', max(abs(a_MILSTD)), rms(a_MILSTD));

%% ========================================================================
%  PART 2: MSS 저주파 — 4방향
%  ========================================================================
disp('[2/3] MSS 저주파 시뮬레이션 (4방향)...');

T0 = Tz / 0.710;
w0 = 2*pi / T0;

load(which('supply.mat'), 'vessel');

if vessel.forceRAO.w(end) > 3.0
    w_index = find(vessel.forceRAO.w > 3.0, 1) - 1;
    vessel.forceRAO.w = vessel.forceRAO.w(1:w_index);
    for DOF = 1:length(vessel.forceRAO.amp)
        vessel.forceRAO.amp{DOF} = vessel.forceRAO.amp{DOF}(1:w_index, :, :);
        vessel.forceRAO.phase{DOF} = vessel.forceRAO.phase{DOF}(1:w_index, :, :);
    end
end

omegaMax = vessel.forceRAO.w(end);
[S_M, Omega, Amp, ~, ~, mu] = waveDirectionalSpectrum('JONSWAP', ...
    [Hs, w0, 3.3], 60, omegaMax, 0, 24);

h_mss = 0.02;
RAO_update_period = 0.1;
a_MSS_all = zeros(N, length(beta_list_deg));

for b = 1:length(beta_list_deg)
    beta_rad = deg2rad(beta_list_deg(b));
    fprintf('  [%d/4] %s...', b, beta_labels{b});

    clear waveMotionRAO;
    nextRAOtime = 0;
    t_mss = 0:h_mss:T_final+T_transient-1;
    simdata_mss = zeros(length(t_mss), 19);

    for i = 1:length(t_mss)
        psi = 0;
        if t_mss(i) >= nextRAOtime
            [eta_WF, nu_WF, nudot_WF, waveElevation] = waveMotionRAO( ...
                t_mss(i), S_M, Amp, Omega, mu, vessel, U_ship, psi, beta_rad, 60);
            nextRAOtime = nextRAOtime + RAO_update_period;
        end
        simdata_mss(i,:) = [eta_WF' nu_WF' nudot_WF' waveElevation];
    end

    startIdx = max(1, floor(T_transient / h_mss) + 1);
    t_mss_trim = t_mss(startIdx:end) - t_mss(startIdx);
    nudot_trim = simdata_mss(startIdx:end, 13:18);

    ax = nudot_trim(:,1) / g_const;
    ay = nudot_trim(:,2) / g_const;
    az = nudot_trim(:,3) / g_const;
    a_rss = sqrt(ax.^2 + ay.^2 + az.^2);

    a_MSS_all(:,b) = interp1(t_mss_trim, a_rss, t_vib, 'linear', 0);
    fprintf(' max=%.4f g\n', max(a_MSS_all(:,b)));
end

%% ========================================================================
%  PART 3: 10 시나리오 생성
%  ========================================================================
disp('[3/3] 10 시나리오 생성...');

k_R_ohm = k_R * 1e-3;
scenarios = struct();
idx = 0;

% --- Case 1: No vibration (1개) ---
idx = idx + 1;
a_total = zeros(N, 1);
scenarios(idx).case_id = 1;
scenarios(idx).case_label = 'No vibration';
scenarios(idx).beta_deg = NaN;
scenarios(idx).beta_label = 'N/A';
scenarios(idx).t = t_vib;
scenarios(idx).a_total = a_total;
scenarios(idx).deltaR = k_R_ohm * abs(a_total);
scenarios(idx).a_max_g = 0;
scenarios(idx).a_rms_g = 0;
scenarios(idx).deltaR_max_mOhm = 0;
scenarios(idx).deltaV_max_mV_1C = 0;

% --- Case 2: MIL-STD only (1개) ---
idx = idx + 1;
a_total = a_MILSTD;
scenarios(idx).case_id = 2;
scenarios(idx).case_label = 'MIL-STD only';
scenarios(idx).beta_deg = NaN;
scenarios(idx).beta_label = 'N/A';
scenarios(idx).t = t_vib;
scenarios(idx).a_total = a_total;
scenarios(idx).deltaR = k_R_ohm * abs(a_total);
scenarios(idx).a_max_g = max(abs(a_total));
scenarios(idx).a_rms_g = rms(a_total);
scenarios(idx).deltaR_max_mOhm = max(scenarios(idx).deltaR) * 1e3;
scenarios(idx).deltaV_max_mV_1C = max(scenarios(idx).deltaR) * 2.7 * 1e3;

% --- Case 3: MSS only (4방향) ---
for b = 1:length(beta_list_deg)
    idx = idx + 1;
    a_total = a_MSS_all(:,b);
    scenarios(idx).case_id = 3;
    scenarios(idx).case_label = 'MSS only';
    scenarios(idx).beta_deg = beta_list_deg(b);
    scenarios(idx).beta_label = beta_labels{b};
    scenarios(idx).t = t_vib;
    scenarios(idx).a_total = a_total;
    scenarios(idx).deltaR = k_R_ohm * abs(a_total);
    scenarios(idx).a_max_g = max(abs(a_total));
    scenarios(idx).a_rms_g = rms(a_total);
    scenarios(idx).deltaR_max_mOhm = max(scenarios(idx).deltaR) * 1e3;
    scenarios(idx).deltaV_max_mV_1C = max(scenarios(idx).deltaR) * 2.7 * 1e3;
end

% --- Case 4: MSS + MIL-STD (4방향) ---
for b = 1:length(beta_list_deg)
    idx = idx + 1;
    a_total = a_MSS_all(:,b) + a_MILSTD;
    scenarios(idx).case_id = 4;
    scenarios(idx).case_label = 'MSS + MIL-STD';
    scenarios(idx).beta_deg = beta_list_deg(b);
    scenarios(idx).beta_label = beta_labels{b};
    scenarios(idx).t = t_vib;
    scenarios(idx).a_total = a_total;
    scenarios(idx).deltaR = k_R_ohm * abs(a_total);
    scenarios(idx).a_max_g = max(abs(a_total));
    scenarios(idx).a_rms_g = rms(a_total);
    scenarios(idx).deltaR_max_mOhm = max(scenarios(idx).deltaR) * 1e3;
    scenarios(idx).deltaV_max_mV_1C = max(scenarios(idx).deltaR) * 2.7 * 1e3;
end

disp(['  완료: ', num2str(idx), '개 시나리오']);

%% ========================================================================
%  결과 요약 테이블
%  ========================================================================
disp(' ');
disp('============================================================');
disp('  결과 요약 (10 시나리오)');
disp('============================================================');
disp('  #   Case              방향              a_max(g)  a_RMS(g)  dR_max(mO)  dV@1C(mV)');
disp('  --- ----------------  ----------------  --------  --------  ----------  ---------');

for i = 1:length(scenarios)
    fprintf('  %2d  %-16s  %-16s  %8.4f  %8.4f  %10.4f  %9.4f\n', ...
        i, scenarios(i).case_label, scenarios(i).beta_label, ...
        scenarios(i).a_max_g, scenarios(i).a_rms_g, ...
        scenarios(i).deltaR_max_mOhm, scenarios(i).deltaV_max_mV_1C);
end
disp('============================================================');

%% ========================================================================
%  시각화 — 인덱스 맵
%  ========================================================================
% scenarios 구조:
%  1: Case1 (No vib)
%  2: Case2 (MIL-STD only)
%  3~6: Case3 (MSS only) × [0, 90, 135, 180]
%  7~10: Case4 (MSS+MIL) × [0, 90, 135, 180]

idx_novib = 1;
idx_milstd = 2;
idx_mss = 3;          % Case3: MSS Head only
idx_both = 4;         % Case4: MSS+MIL Head only
idx_beam_mss = 3;     % (legacy name) Head 1-direction representative
idx_beam_both = 4;    % (legacy name) Head 1-direction representative

case_colors = {[0.5 0.5 0.5], [1 0.3 0.3], [0.2 0.5 1], [0.1 0.1 0.1]};
dir_colors = {'b', 'r', [0 0.6 0], [0.8 0.4 0]};

% --- Figure 1: 4 Case 비교 (Beam sea 대표) ---
figure(1); clf;
beam_set = [idx_novib, idx_milstd, idx_beam_mss, idx_beam_both];
for c = 1:4
    subplot(4,1,c);
    plot(t_vib, scenarios(beam_set(c)).a_total, 'Color', case_colors{c}, 'LineWidth', 0.5);
    ylabel('a (g)');
    title(sprintf('Case %d: %s', c, scenarios(beam_set(c)).case_label));
    grid on; xlim([0 60]);
end
xlabel('Time (s)');
sgtitle('4 Case 비교 — Beam Sea (90°)');

% --- Figure 2: 2초 확대 (Beam sea) ---
figure(2); clf;
t_zoom = [30 32];
idx_z = t_vib >= t_zoom(1) & t_vib <= t_zoom(2);
for c = 1:4
    subplot(4,1,c);
    plot(t_vib(idx_z), scenarios(beam_set(c)).a_total(idx_z), ...
        'Color', case_colors{c}, 'LineWidth', 1);
    ylabel('a (g)');
    title(sprintf('Case %d: %s', c, scenarios(beam_set(c)).case_label));
    grid on;
end
xlabel('Time (s)');
sgtitle('2초 확대 — Beam Sea (90°)');

% --- Figure 3: FFT (4 Case, Beam sea) ---
figure(3); clf;
Nhalf = floor(N/2);
f_fft = (0:Nhalf-1)' * Fs / N;
for c = 1:4
    subplot(4,1,c);
    Y = fft(scenarios(beam_set(c)).a_total);
    P = abs(Y(1:Nhalf)) / N * 2;
    plot(f_fft, P, 'Color', case_colors{c}, 'LineWidth', 1);
    xlim([0 40]); ylabel('Amp (g)');
    title(sprintf('Case %d: %s', c, scenarios(beam_set(c)).case_label));
    grid on;
end
xlabel('Frequency (Hz)');
sgtitle('FFT — Beam Sea (90°)');

% --- Figure 4: deltaR (4 Case, Beam sea) ---
figure(4); clf;
for c = 1:4
    subplot(4,1,c);
    plot(t_vib, scenarios(beam_set(c)).deltaR*1e3, ...
        'Color', case_colors{c}, 'LineWidth', 0.5);
    ylabel('\DeltaR (m\Omega)');
    title(sprintf('Case %d: %s', c, scenarios(beam_set(c)).case_label));
    grid on; xlim([0 60]);
end
xlabel('Time (s)');
sgtitle(sprintf('\\DeltaR(t),  k_R = %.2f m\\Omega/g', k_R));

% --- Figure 5: MSS only — Head sea (1-direction only) ---
figure(5); clf;
n_dir = length(idx_mss);
for b = 1:n_dir
    subplot(n_dir,1,b);
    plot(t_vib, scenarios(idx_mss(b)).a_total, 'Color', dir_colors{b}, 'LineWidth', 1);
    ylabel('a (g)');
    title(sprintf('MSS only — %s', beta_labels{b}));
    grid on; xlim([0 120]);
end
xlabel('Time (s)');
sgtitle('Case 3 (MSS only) — Head sea');

% --- Figure 6: 막대 그래프 요약 ---
figure(6); clf;
bar_labels = {'No vib', 'MIL-STD', 'MSS(beam)', 'Both(beam)'};
bar_max = [scenarios(beam_set(1)).a_max_g, scenarios(beam_set(2)).a_max_g, ...
           scenarios(beam_set(3)).a_max_g, scenarios(beam_set(4)).a_max_g];
bar_rms = [scenarios(beam_set(1)).a_rms_g, scenarios(beam_set(2)).a_rms_g, ...
           scenarios(beam_set(3)).a_rms_g, scenarios(beam_set(4)).a_rms_g];
bar_dr  = [scenarios(beam_set(1)).deltaR_max_mOhm, scenarios(beam_set(2)).deltaR_max_mOhm, ...
           scenarios(beam_set(3)).deltaR_max_mOhm, scenarios(beam_set(4)).deltaR_max_mOhm];

subplot(1,3,1);
bar(bar_max, 'FaceColor', [0.3 0.5 0.8]);
set(gca, 'XTickLabel', bar_labels); ylabel('Max (g)'); title('최대 가속도'); grid on;
subplot(1,3,2);
bar(bar_rms, 'FaceColor', [0.3 0.7 0.4]);
set(gca, 'XTickLabel', bar_labels); ylabel('RMS (g)'); title('RMS 가속도'); grid on;
subplot(1,3,3);
bar(bar_dr, 'FaceColor', [0.8 0.3 0.3]);
set(gca, 'XTickLabel', bar_labels); ylabel('m\Omega'); title('Max \DeltaR'); grid on;
sgtitle('Beam Sea — Case별 비교');

% --- Figure 7: MIL-STD 프로파일 ---
figure(7); clf;
bar(freq_milstd, accel_amplitude_g * milstd_scale, 'FaceColor', [0.3 0.5 0.8]);
xlabel('Frequency (Hz)'); ylabel('Acceleration (g)');
title('MIL-STD-810H Table 528.1-I'); grid on;

disp(' ');
disp('Figure 1: 4 Case 시계열 (Beam sea, 60초)');
disp('Figure 2: 4 Case 확대 (2초)');
disp('Figure 3: 4 Case FFT');
disp('Figure 4: 4 Case deltaR');
disp('Figure 5: MSS only 방향별');
disp('Figure 6: 막대 그래프 요약');
disp('Figure 7: MIL-STD 프로파일');

%% ========================================================================
%  데이터 저장 — 시나리오별 개별 파일 + 전체 파일
%  ========================================================================
save_dir = '../data/vibration';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% --- 시나리오별 개별 저장 ---
disp(' ');
disp('개별 파일 저장...');
for i = 1:length(scenarios)
    s = scenarios(i);

    % 파일명 생성: case1_novib.mat, case2_milstd.mat, case3_mss_beam90.mat, ...
    switch s.case_id
        case 1
            fname = 'case1_novib';
        case 2
            fname = 'case2_milstd';
        case 3
            fname = sprintf('case3_mss_%ddeg', s.beta_deg);
        case 4
            fname = sprintf('case4_both_%ddeg', s.beta_deg);
    end

    vib.t = s.t;
    vib.a_total = s.a_total;
    vib.deltaR = s.deltaR;
    vib.case_id = s.case_id;
    vib.case_label = s.case_label;
    vib.beta_deg = s.beta_deg;
    vib.Fs = Fs;
    vib.params.Hs = Hs;
    vib.params.Tz = Tz;
    vib.params.U_ship = U_ship;
    vib.params.milstd_scale = milstd_scale;
    vib.params.k_R_mOhm_per_g = k_R;

    save(fullfile(save_dir, [fname '.mat']), 'vib');
    fprintf('  %s.mat\n', fname);
end

% --- 전체 데이터 저장 ---
vibration_data.scenarios = scenarios;
vibration_data.a_MILSTD = a_MILSTD;
vibration_data.a_MSS_all = a_MSS_all;
vibration_data.t = t_vib;
vibration_data.Fs = Fs;
vibration_data.N = N;
vibration_data.params.Hs = Hs;
vibration_data.params.Tz = Tz;
vibration_data.params.U_ship = U_ship;
vibration_data.params.milstd_scale = milstd_scale;
vibration_data.params.k_R_mOhm_per_g = k_R;
vibration_data.params.beta_list_deg = beta_list_deg;
vibration_data.milstd_profile.freq_Hz = freq_milstd;
vibration_data.milstd_profile.accel_g = accel_amplitude_g;
vibration_data.idx.novib = idx_novib;
vibration_data.idx.milstd = idx_milstd;
vibration_data.idx.mss = idx_mss;
vibration_data.idx.both = idx_both;

save('../data/vibration_data_all.mat', 'vibration_data');

disp(' ');
disp('저장 완료:');
disp('  data/vibration/case1_novib.mat');
disp('  data/vibration/case2_milstd.mat');
disp('  data/vibration/case3_mss_0deg.mat  ~ case3_mss_180deg.mat');
disp('  data/vibration/case4_both_0deg.mat ~ case4_both_180deg.mat');
disp('  data/vibration_data_all.mat (전체)');
disp(' ');
disp('Simscape 사용법:');
disp('  >> load(''data/vibration/case4_both_90deg.mat'')');
disp('  >> % vib.deltaR 을 Variable Resistor 입력으로 사용');
