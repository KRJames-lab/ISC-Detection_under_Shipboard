%% build_simscape_model.m
% Ship Battery Simscape Model Builder
% NCR18650PF + Variable Resistor (vibration-induced) + Thermal
%
% 회로 구성:
%   Battery(+) → VarR(+/-) → ISensor(+/-) → LoadSrc(+/-) → GND
%   Battery(-) → GND
%   Battery(H) → ConvHT → AmbientT(25°C)
%
% 신호 흐름:
%   Vibration a(t) → |a| → k_R gain → SPS → VarR.R
%   Load I(t) → SPS → LoadSrc.I
%   VSensor.V → PSS → V_terminal (workspace)
%   ISensor.I → PSS → I_battery (workspace)
%   TSensor.T → PSS → T_cell (workspace)
%
% 포트 매핑 (R2022b Simscape):
%   Battery (thermal):  LConn(1)=(+), RConn(1)=(-), RConn(2)=H
%   VarR:               LConn(1)=R(signal), LConn(2)=(+), RConn(1)=(-)
%   ISensor:            LConn(1)=(+), RConn(1)=I(signal), RConn(2)=(-)
%   VSensor:            LConn(1)=(+), RConn(1)=V(signal), RConn(2)=(-)
%   LoadSrc:            LConn(1)=(+), RConn(1)=I(signal), RConn(2)=(-)
%   TSensor:            LConn(1)=measure, RConn(1)=ref, RConn(2)=T(signal)

%% ===== 설정 파라미터 =====
mdl = 'ShipBattery_Model';

% --- NCR18650PF 셀 파라미터 ---
cell_SOC_vec  = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
cell_OCV_vec  = [2.50, 3.35, 3.50, 3.65, 3.85, 4.05, 4.20];   % V
cell_R0_vec   = [7.0, 6.2, 5.8, 5.6, 5.8, 6.0, 6.2] * 1e-3;  % Ohm
cell_R1_vec   = [20, 18, 15, 12, 15, 18, 20] * 1e-3;           % Ohm
cell_tau1_vec = [30, 25, 20, 18, 20, 25, 30];                   % s
cell_AH       = 2.9;    % Ah
cell_thermal_mass = 45;  % J/K (48g × ~1000 J/(kg·K))

% --- 시뮬레이션 파라미터 ---
k_R       = 0.5e-3;    % Ohm/g (진동 민감도)
I_load    = -2.9;      % A (음수 = 방전, Controlled CS 관례)
T_ambient = 298.15;    % K (25°C)
SOC_init  = 0.9;       % 초기 SOC
T_stop    = 600;       % s

%% ===== 모델 생성 =====
try close_system(mdl, 0); catch; end
new_system(mdl);
open_system(mdl);
set_param(mdl, 'Solver', 'ode23t', 'StopTime', num2str(T_stop));

%% ===== 블록 배치 =====
% --- 전기 회로 ---
add_block('batt_lib/Cells/Battery (Table-Based)', [mdl '/Battery'], ...
    'Position', [120, 80, 220, 170]);
add_block('fl_lib/Electrical/Electrical Elements/Variable Resistor', [mdl '/VarR'], ...
    'Position', [320, 80, 400, 140]);
add_block('fl_lib/Electrical/Electrical Sensors/Current Sensor', [mdl '/ISensor'], ...
    'Position', [500, 80, 580, 140]);
add_block('fl_lib/Electrical/Electrical Sources/Controlled Current Source', [mdl '/LoadSrc'], ...
    'Position', [700, 80, 780, 140]);
add_block('fl_lib/Electrical/Electrical Elements/Electrical Reference', [mdl '/GND'], ...
    'Position', [500, 250, 540, 280]);
add_block('nesl_utility/Solver Configuration', [mdl '/SolverCfg'], ...
    'Position', [320, 250, 420, 280]);
add_block('fl_lib/Electrical/Electrical Sensors/Voltage Sensor', [mdl '/VSensor'], ...
    'Position', [600, 200, 680, 260]);

% --- 열 회로 ---
add_block('fl_lib/Thermal/Thermal Elements/Convective Heat Transfer', [mdl '/ConvHT'], ...
    'Position', [200, 350, 280, 390]);
add_block('fl_lib/Thermal/Thermal Sources/Temperature Source', [mdl '/AmbientT'], ...
    'Position', [320, 350, 400, 390]);
add_block('fl_lib/Thermal/Thermal Sensors/Temperature Sensor', [mdl '/TSensor'], ...
    'Position', [200, 430, 280, 470]);
add_block('fl_lib/Thermal/Thermal Elements/Thermal Reference', [mdl '/ThermalRef'], ...
    'Position', [450, 380, 490, 410]);

% --- 진동 신호 체인 ---
add_block('simulink/Sources/Constant', [mdl '/VibConst'], ...
    'Position', [100, 400, 150, 430], 'Value', '0');
add_block('simulink/Math Operations/Abs', [mdl '/Abs'], ...
    'Position', [200, 403, 230, 427]);
add_block('simulink/Math Operations/Gain', [mdl '/kR_Gain'], ...
    'Position', [270, 403, 310, 427], 'Gain', num2str(k_R));
add_block('nesl_utility/Simulink-PS Converter', [mdl '/SPS_R'], ...
    'Position', [360, 403, 400, 427]);

% --- 부하 전류 신호 ---
add_block('simulink/Sources/Constant', [mdl '/LoadConst'], ...
    'Position', [100, 470, 150, 500], 'Value', num2str(I_load));
add_block('nesl_utility/Simulink-PS Converter', [mdl '/SPS_I'], ...
    'Position', [360, 473, 400, 497]);

% --- PS-Simulink 변환기 ---
add_block('nesl_utility/PS-Simulink Converter', [mdl '/PSS_V'], ...
    'Position', [560, 330, 600, 354]);
add_block('nesl_utility/PS-Simulink Converter', [mdl '/PSS_I'], ...
    'Position', [560, 380, 600, 404]);
add_block('nesl_utility/PS-Simulink Converter', [mdl '/PSS_T'], ...
    'Position', [360, 443, 400, 467]);

% --- 출력 (To Workspace) ---
add_block('simulink/Sinks/To Workspace', [mdl '/V_out'], ...
    'Position', [640, 333, 710, 357], 'VariableName', 'V_terminal', 'SaveFormat', 'Timeseries');
add_block('simulink/Sinks/To Workspace', [mdl '/I_out'], ...
    'Position', [640, 383, 710, 407], 'VariableName', 'I_battery', 'SaveFormat', 'Timeseries');
add_block('simulink/Sinks/To Workspace', [mdl '/T_out'], ...
    'Position', [440, 446, 510, 470], 'VariableName', 'T_cell', 'SaveFormat', 'Timeseries');

%% ===== 파라미터 설정 =====
% Battery
set_param([mdl '/Battery'], ...
    'SOC_vec', mat2str(cell_SOC_vec), ...
    'V0_vec', mat2str(cell_OCV_vec), ...
    'R0_vec', mat2str(cell_R0_vec), ...
    'AH', num2str(cell_AH), ...
    'prm_dyn', 'simscape.enum.tablebattery.prm_dyn.rc1', ...
    'R1_vec', mat2str(cell_R1_vec), ...
    'tau1_vec', mat2str(cell_tau1_vec), ...
    'thermal_port', 'simscape.enum.thermaleffects.model', ...
    'thermal_mass', num2str(cell_thermal_mass), ...
    'stateOfCharge', num2str(SOC_init));

% Convective HT: 18650 surface ≈ 3.67e-3 m², h ≈ 10 W/(m²·K)
set_param([mdl '/ConvHT'], 'area', '3.67e-3', 'heat_tr_coeff', '10');

% Ambient temperature
set_param([mdl '/AmbientT'], 'temperature', num2str(T_ambient));

%% ===== 블록 연결 =====
ph = struct();
blocks = {'Battery','VarR','ISensor','LoadSrc','GND','SolverCfg', ...
    'VSensor','ConvHT','AmbientT','TSensor','ThermalRef', ...
    'SPS_R','SPS_I','PSS_V','PSS_I','PSS_T'};
for i = 1:length(blocks)
    ph.(blocks{i}) = get_param([mdl '/' blocks{i}], 'PortHandles');
end

% --- 전기 회로 ---
add_line(mdl, ph.Battery.LConn(1), ph.VarR.LConn(2), 'autorouting','on');    % Bat(+) → VarR(+)
add_line(mdl, ph.VarR.RConn(1), ph.ISensor.LConn(1), 'autorouting','on');    % VarR(-) → IS(+)
add_line(mdl, ph.ISensor.RConn(2), ph.LoadSrc.LConn(1), 'autorouting','on'); % IS(-) → Load(+)
add_line(mdl, ph.LoadSrc.RConn(2), ph.GND.LConn(1), 'autorouting','on');     % Load(-) → GND
add_line(mdl, ph.Battery.RConn(1), ph.GND.LConn(1), 'autorouting','on');     % Bat(-) → GND
add_line(mdl, ph.SolverCfg.RConn(1), ph.GND.LConn(1), 'autorouting','on');   % Solver → GND

% --- 전압 센서 ---
add_line(mdl, ph.VSensor.LConn(1), ph.VarR.RConn(1), 'autorouting','on');    % VS(+) → 회로 노드
add_line(mdl, ph.VSensor.RConn(2), ph.GND.LConn(1), 'autorouting','on');     % VS(-) → GND

% --- 진동 신호 ---
add_line(mdl, 'VibConst/1', 'Abs/1', 'autorouting','on');
add_line(mdl, 'Abs/1', 'kR_Gain/1', 'autorouting','on');
add_line(mdl, 'kR_Gain/1', 'SPS_R/1', 'autorouting','on');
add_line(mdl, ph.SPS_R.RConn(1), ph.VarR.LConn(1), 'autorouting','on');      % → VarR.R signal

% --- 부하 전류 ---
add_line(mdl, 'LoadConst/1', 'SPS_I/1', 'autorouting','on');
add_line(mdl, ph.SPS_I.RConn(1), ph.LoadSrc.RConn(1), 'autorouting','on');   % → Load.I signal

% --- 출력 ---
add_line(mdl, ph.VSensor.RConn(1), ph.PSS_V.LConn(1), 'autorouting','on');   % V signal
add_line(mdl, 'PSS_V/1', 'V_out/1', 'autorouting','on');
add_line(mdl, ph.ISensor.RConn(1), ph.PSS_I.LConn(1), 'autorouting','on');   % I signal
add_line(mdl, 'PSS_I/1', 'I_out/1', 'autorouting','on');

% --- 열 회로 ---
add_line(mdl, ph.Battery.RConn(2), ph.ConvHT.LConn(1), 'autorouting','on');  % Bat.H → ConvHT
add_line(mdl, ph.ConvHT.RConn(1), ph.AmbientT.LConn(1), 'autorouting','on');% ConvHT → Ambient
add_line(mdl, ph.TSensor.LConn(1), ph.ConvHT.LConn(1), 'autorouting','on'); % TSensor → Bat.H node
add_line(mdl, ph.TSensor.RConn(1), ph.ThermalRef.LConn(1), 'autorouting','on'); % TSensor ref
add_line(mdl, ph.TSensor.RConn(2), ph.PSS_T.LConn(1), 'autorouting','on');  % T signal
add_line(mdl, 'PSS_T/1', 'T_out/1', 'autorouting','on');

%% ===== 저장 =====
modelPath = fullfile(fileparts(mfilename('fullpath')), '..', 'models', [mdl '.slx']);
save_system(mdl, modelPath);
disp(['Model saved: ' modelPath]);
disp('Build complete.');
