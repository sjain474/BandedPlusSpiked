% ================================================================
% RFView Example Script (with ICM on/off comparison)
% ------------------------------------------------
% - Loads RFView® scenario parameters
% - Loops over platform positions
% - Generates simulated radar data (IQ, clutter maps, etc.)
% - Runs with and without Internal Clutter Model (ICM)
% - Performs basic processing: range–Doppler maps, clutter covariance
% - Plots covariance structure and SVDs
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

% Simulation options
SIM.OPTIONS.clutter   = 1;
SIM.OPTIONS.discrete  = 1;
SIM.PARAMS.atod_os_fac = 5;
SIM.TS.PRF     = 1650;   % Changed PRF (default = 1100 Hz)
SIM.TS.NumPulse = 16;

% Rx antenna parameters
SIM.ANTENNA.rx_ant_dh = .015;
SIM.ANTENNA.rx_ant_dv = .015;
SIM.ANTENNA.rx_ant_numelem_h = 48;
SIM.ANTENNA.rx_ant_numelem_v = 5;
SIM.ANTENNA.rx_ant_numchan_h = 16;
SIM.ANTENNA.rx_ant_numchan_v = 1;
SIM.ANTENNA.rx_fb_ratio = 14;
SIM.ANTENNA.rx_pattern = 'u';
SIM.ANTENNA.rx_element_type = 'isotropic';
SIM.ANTENNA.rx_pol = 'v';

% Tx antenna parameters
SIM.ANTENNA.tx_ant_dh = .015;
SIM.ANTENNA.tx_ant_dv = .015;
SIM.ANTENNA.tx_ant_numelem_h = 48;
SIM.ANTENNA.tx_ant_numelem_v = 5;
SIM.ANTENNA.tx_ant_numchan_h = 1;
SIM.ANTENNA.tx_ant_numchan_v = 1;
SIM.ANTENNA.tx_fb_ratio = 0;
SIM.ANTENNA.tx_pattern = 'h';
SIM.ANTENNA.tx_element_type = 'isotropic';
SIM.ANTENNA.tx_pol = 'v';

%% -------------------- Simulation Loop --------------------
for ii = 1   % loop over platform positions (here only 1)
    SIM.platform_index = ii;
    SIM.target_index   = ii;
    
    % ----------- Run RFView without ICM -----------
    SIM.OPTIONS.use_icm = 0;
    SIM = rfview_func1(SIM);
    process_example   % beamform, match filter
    clr_lim = [-120 -60];
    plt_rng_dop
    title('without ICM');
    
    % ----------- Run RFView with ICM -----------
    SIM.OPTIONS.use_icm = 1;
    SIM = rfview_func1(SIM);
    process_example
    clr_lim = [-120 -60];
    figure
    plt_rng_dop
    title('with ICM');
    
    % ----------- Save key outputs -----------
    CH(ii,:)               = 20*log10(squeeze(abs(SIM.hc(1,1,:))));
    PathPowerHistory(:,:,ii) = SIM.CM.PathPower;
    TxPatHistory(:,:,ii)     = 20*log10(abs(reshape(SIM.tp,200,200)));
end

%% -------------------- Post-Processing --------------------
% Plot clutter map
figure
x = [0:size(SIM.CM.PathPower,2)-1]*SIM.MLSCATS.CellEW;
y = [0:size(SIM.CM.PathPower,1)-1]*SIM.MLSCATS.CellNS;
imagesc(x/1852,y/1852,flipud(SIM.CM.PathPower),[-300 -200]);
axis xy
xlabel('distance (nm)');
ylabel('distance (nm)');
title('RFView clutter map');
hc = colorbar;
ylabel(hc,'relative power (dB)');

% Range–Doppler map (from process_example)
figure
f = fvect(SIM.WAVEFORM.PRF,m);
imagesc(f,SIM.PARAMS.rbins/1852/2, ...
        20*log10(abs(tmp_conv(:,1:length(SIM.PARAMS.rbins)).'))+30,[-115 -80]);
axis xy
xlabel('Doppler frequency (Hz)');
ylabel('range (nm)');
hc = colorbar;
ylabel('Power (dBm)');

% Export scenario to KML
rfview2kml

%% -------------------- Covariance Matrix Analysis --------------------
ClutterData = SIM.xc_clut;
[L,N,R] = size(ClutterData);
RangeGate = R/2;
M = 16;
p = M*N;
K = 20*p;

% Clutter submatrix
Hc = reshape(ClutterData(1:M,:,RangeGate-K/2+1:RangeGate+K/2),p,K);
R_c = (Hc*Hc')/K;

% Noise covariance
sigma = 1.5e-7;
Noise_Cov = sigma^2*eye(p);

% Clutter+Noise covariance
Cov = R_c + Noise_Cov;

% Plot covariance matrix
figure
imagesc(10*log10(abs(Cov)));
colormap("jet")
colorbar
title('Clutter + Noise Covariance');

% Compare SVD spectra
figure
[~,D_c,~] = svd(R_c);
[~,D_n,~] = svd(Noise_Cov);
[~,D,~]   = svd(Cov);
semilogy(diag(D_c)); hold on
semilogy(diag(D_n));
semilogy(diag(D));
hold off
grid on
title('Eigenvalue Spectra');
legend('Clutter','Noise','Clutter+Noise');
