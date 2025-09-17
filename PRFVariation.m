clear; close all;

% ================================================================
% Radar Covariance Estimation & PRF Analysis
% ------------------------------------------------
% - Loads multiple PRF datasets (Cofar_PRF_xxxx.mat)
% - Estimates clutter+noise covariance using Banded+Spiked model
% - Runs Monte Carlo simulations across varying sample sizes
% - Computes SCNR performance vs samples, azimuth, and Doppler
% - Plots and saves figures in specified directory
% ================================================================

%% -------------------- Simulation Parameters --------------------
data_dir = '/MATLAB Drive/Revision Codes/';
file_tags = {'1100', '1650', '2200'};   % PRFs to analyze
Cov_Num = length(file_tags);

M = 4;                % Number of spatial channels
ChannelNumber = 5;    % Channel index
K_multiplier = 20;    % Scaling for snapshots per CPI
sigma_mat = [1.5e-7, 1.5e-7, 1.5e-7];  % Noise variances
alpha_mat = [1.5, 1.5, 1.5];           % Regularization scaling
p = 256;              % Dimension
Monte = 100;          % Monte Carlo simulations

% Sample sizes
Num_samples = floor(2 .^ linspace(7, 9, 7));
num_samples_len = length(Num_samples);

% Angle/Doppler grids
angle_rad = (-180:10:180) * pi / 180;
doppler = -0.5:0.05:0.5;
num_angles = length(angle_rad);
num_doppler = length(doppler);

PRF_Num_Values = length(file_tags);

% Storage
Avg_Rho_band     = zeros(num_samples_len, PRF_Num_Values);
Avg_Angle_band   = zeros(num_samples_len, PRF_Num_Values, num_angles);
Avg_Doppler_band = zeros(num_samples_len, PRF_Num_Values, num_doppler);
CovarianceStruct = struct();

%% -------------------- Weight Matrix Precomputation --------------------
w = createWeightMatrix(p);
s_m = storeIndexSets(p, p - 1);
W_struct = generateWeightMatrices(p);

tic; % start timer

% MP median normalization constants
MP_median = zeros(size(Num_samples));
for i = 1:length(Num_samples)
    MP_median(i) = compute_MP_median(p, Num_samples(i));
end

%% -------------------- Main Loop over PRFs --------------------
for file_idx = 1:length(file_tags)
    % === Load PRF data ===
    tag = file_tags{file_idx};
    file_path = fullfile(data_dir, ['Cofar_PRF_' tag '.mat']);
    data = load(file_path);
    SIM = data.SIM;

    % Extract clutter & noise
    Clutter_Data = SIM.xc_clut;
    Noise = SIM.nse;
    [L, N, R] = size(Noise);
    K = K_multiplier * N;
    RangeGate = round(R / 2) + 50;
    p = M * N;
    sigma = sigma_mat(file_idx);

    % Synthetic noise
    x_n = sigma * (randn(p, K) + 1i * randn(p, K)) / sqrt(2);

    % Clutter matrix
    if M == 1 && ChannelNumber <= L
        Hc = reshape(Clutter_Data(ChannelNumber, :, RangeGate - K/2 : RangeGate + K/2 - 1), p, K);
    else
        Hc = reshape(Clutter_Data(1:M, :, RangeGate - K/2 : RangeGate + K/2 - 1), p, K);
    end

    % Covariance matrix
    Cov = ((Hc * Hc') / (sigma^2 * K)) + eye(p);

    % Store covariance
    field_name = ['PRF_' tag];
    CovarianceStruct.(field_name) = Cov;
    R_cov = Cov;
    inv_R = pinv(R_cov);

    % Precompute clutter slices
    H_c_transpose_H_c = zeros(p,p,length(Num_samples));
    H_c_cells = cell(length(Num_samples),1);
    for index = 1:length(Num_samples)
        if mod(Num_samples(index),2)==1
            H_c_k = reshape(Clutter_Data(1:M,:,RangeGate-(Num_samples(index)-1)/2: ...
                         RangeGate+(Num_samples(index)-1)/2),p,(Num_samples(index)));
        else
            H_c_k = reshape(Clutter_Data(1:M,:,RangeGate-(Num_samples(index))/2: ...
                         RangeGate+(Num_samples(index))/2-1),p,(Num_samples(index)));
        end
        H_c_cells{index} = H_c_k;
        H_c_transpose_H_c(:,:,index) = H_c_k*H_c_k';
    end

    % Monte Carlo storage
    Rho_Total_band = zeros(Monte, length(Num_samples));
    Rho_Angle_band = zeros(Monte,length(Num_samples),length(angle_rad));
    Rho_Doppler_band = zeros(Monte,length(Num_samples),length(doppler));

    %% === Monte Carlo Simulations ===
    for monte = 1:Monte
        local_rho_band = zeros(1, length(Num_samples));
        local_angle_band = zeros(length(Num_samples),length(angle_rad));
        local_Doppler_band = zeros(length(Num_samples),length(doppler));
        local_estimate=zeros(1,length(Num_samples));

        for i = 1:length(Num_samples)
            n = Num_samples(i);
            noise = sigma*(randn(p,n)+1i*randn(p,n))/sqrt(2);
            S = (H_c_transpose_H_c(:,:,i)+noise*noise')/(sigma*sigma*n);
            X = (H_c_cells{i}+noise)'/sigma;

            [U,eig_S,V] = svd(S);
            eigen = diag(eig_S);
            if isnan(MP_median(i))
                local_estimate(i) = median(eigen);
            else
                local_estimate(i) = median(eigen)/MP_median(i);
            end
            hat_sigma_2 = local_estimate(i);

            % === Estimate covariance ===
            A_hat = struct();
            for l = 1:p-1
                A_hat(l).hat_A = zeros(p,p);
            end
            lambda = alpha_mat(file_idx) * sqrt(log(p) / n);
            hat_R = struct();
            for l = 1:p-1
                sum_mat = zeros(p,p);
                for l_prime = 1:p-1
                    sum_mat = sum_mat + (W_struct(l_prime).W) .* (A_hat(l_prime).hat_A);
                end
                hat_R(l).R = (S) - lambda * sum_mat;
                nu = get_nu(lambda, w, l, hat_R(l).R, s_m);
                for m = 1:l
                    idx = sub2ind([p,p], s_m(m).indices(:,1), s_m(m).indices(:,2));
                    A_hat(l).hat_A(idx) = (w(l,m)/(lambda*(abs(w(l,m))^2+max(nu,0))))*hat_R(l).R(idx);
                end
            end

            % === Banded+Spiked estimate ===
            sum_mat = zeros(p);
            for l = 1:p-1
                sum_mat = sum_mat + (W_struct(l).W) .* (A_hat(l).hat_A);
            end
            B_est = (S+hat_sigma_2*eye(p)) - lambda * sum_mat;
            inv_hat_R = pinv(B_est);

            % Compute SCNR
            [local_rho_band(i), local_angle_band(i,:), local_Doppler_band(i,:)] = ...
                SCNR_avg(inv_hat_R, R_cov, inv_R, M, N, ChannelNumber);

            fprintf("Num Samples=%d, Monte=%d, PRF=%s, alpha=%f\n", ...
                     n, monte, file_tags{file_idx}, alpha_mat(file_idx));
        end
        Rho_Total_band(monte,:) = local_rho_band;
        Rho_Angle_band(monte,:,:) = local_angle_band;
        Rho_Doppler_band(monte,:,:) = local_Doppler_band;
    end

    % === Monte Carlo averages ===
    Avg_Rho_band(:, file_idx)        = mean(Rho_Total_band, 1);
    Avg_Angle_band(:, file_idx, :)   = squeeze(mean(Rho_Angle_band, 1));
    Avg_Doppler_band(:, file_idx, :) = squeeze(mean(Rho_Doppler_band, 1));
end

%% -------------------- Plotting Section --------------------
% Figures:
%   Fig.4 - SCNR vs Number of Samples
%   Fig.5 - SCNR vs Azimuth (fixed index)
%   Fig.6 - SCNR vs Doppler (fixed index)

% --- SCNR vs Number of Samples ---
f4 = figure(4); hold on;
markers = {'o','s','d','x','*','^'};
for j = 1:length(file_tags)
    semilogx(Num_samples, 10*log10(Avg_Rho_band(:,j)), ...
        'LineWidth',2,'MarkerSize',8,'Marker',markers{j});
end
xlabel('Number of Samples','Interpreter','latex','FontSize',14);
ylabel('Normalized SCNR (dB)','Interpreter','latex','FontSize',14);
grid on;
prf_legend = arrayfun(@(x) sprintf('PRF=%s Hz', file_tags{x}),1:length(file_tags),'UniformOutput',false);
legend(prf_legend,'Interpreter','latex','FontSize',12,'Location','southeast');
set(gca,'TickLabelInterpreter','latex','FontSize',12);
save_current_fig(f4,'PRFVar','/MATLAB Drive/Revision Codes/Revised Figures/');

% --- SCNR vs Azimuth ---
f5 = figure(5); hold on;
sample_index_for_plot = 2;
for j = 1:length(file_tags)
    plot(angle_rad,10*log10(squeeze(Avg_Angle_band(sample_index_for_plot,j,:))), ...
        'LineWidth',2,'MarkerSize',8,'Marker',markers{j});
end
xlabel('Azimuth (rad)','Interpreter','latex','FontSize',14);
ylabel('Normalized SCNR (dB)','Interpreter','latex','FontSize',14);
grid on;
legend(prf_legend,'Interpreter','latex','FontSize',12,'Location','southeast');
xlim([-pi,pi]);
set(gca,'TickLabelInterpreter','latex','FontSize',12);
save_current_fig(f5,'PRFVar','/MATLAB Drive/Revision Codes/Revised Figures/');

% --- SCNR vs Doppler ---
f6 = figure(6); hold on;
for j = 1:length(file_tags)
    plot(doppler,10*log10(squeeze(Avg_Doppler_band(sample_index_for_plot,j,:))), ...
        'LineWidth',2,'MarkerSize',8,'Marker',markers{j});
end
xlabel('Normalized Doppler','Interpreter','latex','FontSize',14);
ylabel('Normalized SCNR (dB)','Interpreter','latex','FontSize',14);
grid on;
legend(prf_legend,'Interpreter','latex','FontSize',12,'Location','southeast');
set(gca,'TickLabelInterpreter','latex','FontSize',12);
save_current_fig(f6,'PRFVar','/MATLAB Drive/Revision Codes/Revised Figures/');

%% -------------------- Runtime --------------------
elapsedTime = toc;
fprintf('Total runtime: %.4f seconds\n', elapsedTime);

%% -------------------- Helper Functions --------------------
function w = createWeightMatrix(p)
% CREATEWEIGHTMATRIX Generate weight matrix for banded covariance estimator
    w = zeros(p,p);
    for l = 1:p
        for m = 1:l
            w(l,m) = sqrt(2*l)/(l-m+1);
        end
    end
end

function g = storeIndexSets(p,l)
% STOREINDEXSETS Generate index sets for off-diagonal band structure
    g = struct();
    for m = 1:l
        diff = p-m;
        indexSet = [];
        for j = 1:p
            k1 = j+diff; k2 = j-diff;
            if k1>=1 && k1<=p, indexSet=[indexSet; j,k1]; end
            if k2>=1 && k2<=p && k2~=k1, indexSet=[indexSet; j,k2]; end
        end
        g(m).indices=indexSet;
    end
end

function W_struct = generateWeightMatrices(p)
% GENERATEWEIGHTMATRICES Build structured weight matrices for estimator
    W_struct = struct();
    for l = 1:p-1
        W = zeros(p,p);
        for m = 1:l
            w_lm = sqrt(2*l)/(l-m+1);
            diff = p-m;
            for j = 1:p
                k1=j+diff; k2=j-diff;
                if k1>=1 && k1<=p, W(j,k1)=conj(w_lm); end
                if k2>=1 && k2<=p, W(j,k2)=w_lm; end
            end
        end
        W_struct(l).W=W;
    end
end

function nu_root=get_nu(lambda,w,l,R,s_m)
% GET_NU Root-finding for variational inequality parameter nu
    obv=@(nu) lambda^2-objective(nu,w,l,R,s_m);
    nu_root=fzero(obv,w(size(w,1),1)^2);
end

function obv=objective(nu,w,l,R,s_m)
% OBJECTIVE Helper for get_nu
    obv=0;
    for m=1:l
        obv=obv+(w(l,m)^2)* ...
            norm(R(sub2ind(size(R),s_m(m).indices(:,1),s_m(m).indices(:,2))),2)^2 ...
            /((w(l,m)^2+nu)^2);
    end
end

function [avg_rho,avg_Angle,avg_Doppler]=SCNR_avg(inv_hat_R,R,inv_R,M,N,ChannelNumber)
% SCNR_AVG Compute average normalized SCNR over angle and Doppler
    angle_rad=(-180:10:180)*pi/180;
    doppler=-0.5:0.05:0.5;
    rho_mat=zeros(length(doppler),length(angle_rad));
    parfor i=1:length(doppler)
        local_rho=zeros(1,length(angle_rad));
        for j=1:length(angle_rad)
            local_rho(j)=SCNR_calc(inv_hat_R,R,inv_R,M,N,angle_rad(j),doppler(i),ChannelNumber);
        end
        rho_mat(i,:)=local_rho;
    end
    avg_rho=sum(rho_mat,'all')/numel(rho_mat);
    avg_Doppler=transpose(sum(rho_mat,2))/length(angle_rad);
    avg_Angle=sum(rho_mat,1)/length(doppler);
end

function normalized_rho=SCNR_calc(inv_hat_R,R,inv_R,M,N,theta,nu,ChannelNumber)
% SCNR_CALC Compute normalized SCNR for given angle and Doppler
    lambda=1; d=lambda/2;
    a=angle_steering_vector(M,theta,lambda,d,ChannelNumber);
    d_vec=doppler_steering_vector(nu,N);
    w_st=kron(a,d_vec);
    numerator=abs(w_st'*inv_hat_R*w_st)^2;
    denominator=abs(w_st'*inv_R*w_st)*abs(w_st'*inv_hat_R*R*inv_hat_R*w_st);
    normalized_rho=numerator/denominator;
end

function a=angle_steering_vector(M,theta,lambda,d,ChannelNumber)
% ANGLE_STEERING_VECTOR Generate spatial steering vector
    if M==1
        phase_shifts=2*pi*d/lambda*sin(theta)*(ChannelNumber).';
    else
        phase_shifts=2*pi*d/lambda*sin(theta)*(0:M-1).';
    end
    a=exp(1j*phase_shifts);
end

function d_vec=doppler_steering_vector(nu,N)
% DOPPLER_STEERING_VECTOR Generate Doppler steering vector
    d_vec=exp(1j*2*pi*nu*(0:N-1)).';
end

function median_MP=compute_MP_median(M,N,sigma)
% COMPUTE_MP_MEDIAN Compute median of Marchenko–Pastur distribution
    if nargin<3, sigma=1; end
    gamma=M/N;
    lambda_min=sigma^2*(1-sqrt(gamma))^2;
    lambda_max=sigma^2*(1+sqrt(gamma))^2;
    mp_pdf=@(lambda) (1/(2*pi*gamma*lambda*sigma^2))* ...
        sqrt((lambda_max-lambda).*(lambda-lambda_min)) .* ...
        (lambda>=lambda_min & lambda<=lambda_max);
    cdf=@(lambda) integral(mp_pdf,lambda_min,lambda,'ArrayValued',true);
    median_MP=fzero(@(lambda) cdf(lambda)-0.5,(lambda_min+lambda_max)/2);
end

function save_current_fig(figHandle,prefix,outDir)
% SAVE_CURRENT_FIG Save figure with auto filename
    if ~exist(outDir,'dir'), mkdir(outDir); end
    ax=get(figHandle,'CurrentAxes');
    xlab=get(get(ax,'XLabel'),'String');
    ylab=get(get(ax,'YLabel'),'String');
    if iscell(xlab), xlab=strjoin(xlab,''); end
    if iscell(ylab), ylab=strjoin(ylab,''); end
    clean=@(s) regexprep(s,'\\|{|}|\$|\^|_|~|\s+','');
    xlab_clean=clean(xlab);
    ylab_clean=clean(ylab);
    tstamp=datestr(now,'yyyymmddHHMM');
    fname=sprintf('%s%svs%st%s.fig',prefix,ylab_clean,xlab_clean,tstamp);
    savefig(figHandle,fullfile(outDir,fname));
end
