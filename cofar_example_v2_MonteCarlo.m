% ================================================================
% RFView® Monte Carlo Example
% ------------------------------------------------
% - Loads RFView® scenario (sample_scenario_002.mat)
% - Sets up Tx/Rx antenna parameters
% - Runs Monte Carlo simulations (default: 100)
% - Generates clutter matrix Hc for each run with unique seeds
% - Stores results in Hc_Cell
%
% RFView® © Information Systems Laboratories, Inc.®, 2005 — 2021
% ================================================================

clear all

%% -------------------- Load Scenario --------------------
load sample_scenario_002.mat SIM

% Add toolboxes
addpath(genpath('../toolboxes'));
addpath(genpath('../tools'));
addpath(genpath('../rfview_kernel'));
addpath(genpath('../array'));
addpath(genpath('../antenna_models'));
addpath(genpath('../rfview_mex'));

%% -------------------- Antenna & Simulation Setup --------------------
% Use planar array toolbox
SIM.FUNCTIONS.SIM_INIT.txarrayf = @tx_array_planar;
SIM.FUNCTIONS.SIM_INIT.rxarrayf = @rx_array_planar;

% Options
SIM.OPTIONS.clutter      = 1;
SIM.OPTIONS.discrete     = 1;
SIM.PARAMS.atod_os_fac   = 5;
SIM.TS.NumPulse          = 64;    % Changed number of pulses (default = 64)

% Monte Carlo runs
Monte = 100;

% Rx antenna parameters
SIM.ANTENNA.rx_ant_dh         = .015;
SIM.ANTENNA.rx_ant_dv         = .015;
SIM.ANTENNA.rx_ant_numelem_h  = 48;
SIM.ANTENNA.rx_ant_numelem_v  = 5;
SIM.ANTENNA.rx_ant_numchan_h  = 16;
SIM.ANTENNA.rx_ant_numchan_v  = 1;
SIM.ANTENNA.rx_fb_ratio       = 14;
SIM.ANTENNA.rx_pattern        = 'u';
SIM.ANTENNA.rx_element_type   = 'isotropic';
SIM.ANTENNA.rx_pol            = 'v';

% Tx antenna parameters
SIM.ANTENNA.tx_ant_dh         = .015;
SIM.ANTENNA.tx_ant_dv         = .015;
SIM.ANTENNA.tx_ant_numelem_h  = 48;
SIM.ANTENNA.tx_ant_numelem_v  = 5;
SIM.ANTENNA.tx_ant_numchan_h  = 1;
SIM.ANTENNA.tx_ant_numchan_v  = 1;
SIM.ANTENNA.tx_fb_ratio       = 0;
SIM.ANTENNA.tx_pattern        = 'h';
SIM.ANTENNA.tx_element_type   = 'isotropic';
SIM.ANTENNA.tx_pol            = 'v';

%% -------------------- Monte Carlo Loop --------------------
Hc_Cell = cell(1, Monte);   % Storage for clutter matrices

for m = 1:Monte
    % Loop over platform positions (currently only 1)
    for ii = 1   % 1:1:length(SIM.SCEN.PLATRX.lat)
        
        % Set seeds for reproducibility
        SIM.SEED.rand_seed  = m;
        SIM.SEED.randn_seed = m;
        
        % Set indices
        SIM.platform_index  = ii;
        SIM.target_index    = ii;
        
        % Run RFView with ICM
        SIM.OPTIONS.use_icm = 1;
        SIM = rfview_func1(SIM);
        
        % Extract noise dimensions
        [L,N,R] = size(SIM.nse);
        RangeGate = R/2 + 50;
        M = 4;              % Number of channels, 8 for 32 pulses
        ChannelNumber = 5;   % Channel selection
        p = M * N;             % Matrix Dimension
        K = 5 * p;          % Number of snapshots
        
        % Clutter impulse response
        Clutter_Data = SIM.xc_clut;
        
        % Build clutter matrix Hc
        if M == 1 && ChannelNumber <= L   % Single channel
            Hc = reshape(Clutter_Data(ChannelNumber, :, RangeGate-K/2:RangeGate+K/2-1), p, K);
        else                              % Multiple channels
            Hc = reshape(Clutter_Data(1:M, :, RangeGate-K/2:RangeGate+K/2-1), p, K);
        end
        
        % Store in cell array
        Hc_Cell{m} = Hc;
    end
end
