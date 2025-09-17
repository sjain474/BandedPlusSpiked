clear all 
%% ================================================================
% Load RFView Dataset
% ------------------------------------------------
% - Cofar.mat contains radar simulation data (noise + clutter)
% - Extract noise cube, clutter response, and reshape for analysis
% ================================================================
Data=load("Cofar.mat");% SIM contains the necessary parameters for the dataset
Noise=Data.SIM.nse;
[L,N,R]=size(Noise);
RangeGate=R/2+50;
M=4;              % Number of channels used
ChannelNumber=5;  % Channel index
K=20*N;           % Number of snapshots (samples)
sigma=1.5e-7;     % Noise standard deviation

% Generate synthetic complex Gaussian noise
Sample_CovNoise=sigma*(randn(M*N,K)+1i*randn(M*N,K))/sqrt(2);
Mean_Noise=mean(Sample_CovNoise,2);
x_n=Sample_CovNoise;
NSE=(x_n*x_n')/K;
[Un,sigma_2,Vn]=svd(x_n);

% Clutter response convolved with waveform
Clutter_Data=Data.SIM.xc_clut;

% Single-channel vs multi-channel selection
if M==1 && ChannelNumber<=L % Single channel
    Hc=reshape(Clutter_Data(ChannelNumber,:,RangeGate-K/2:RangeGate+K/2-1),M*N,K);
end
if M~=1 % Multiple channels
    Hc=reshape(Clutter_Data(1:M,:,RangeGate-K/2:RangeGate+K/2-1),M*N,K);
end

% SVD of clutter-only and clutter+noise
[U,D,V]=svd(Hc);
[U1,D1,V1]=svd(Hc+x_n);

%% ================================================================
% Plot singular values of clutter, noise, and clutter+noise
% ================================================================
figure(1)
semilogy(10*log10(diag(D)),LineWidth=2,Color='blue',LineStyle='--');
hold on;
semilogy(10*log10(diag(sigma_2)),LineWidth=2,Color='red',LineStyle='-.');
hold on;
semilogy(10*log10(diag(D1)),LineWidth=2,Color=[0.4660 0.6740 0.1880]); 
hold off;
grid on;

set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);
xlim([1,M*N])
xlabel("Singular Value Index",Interpreter="latex");
ylabel("Singular Value (dB)",Interpreter="latex");
legend('Clutter','Noise','Clutter+Noise',Interpreter="latex");

%% ================================================================
% Image plot of Clutter+Noise covariance matrix
% ================================================================
Cov=((Hc*Hc')/(sigma^2*K))+eye(M*N);

figure(2)
imagesc(10 * log10(sigma^2 * abs(Cov)))
h = colorbar; 
colormap('jet')

h.Label.String = 'Covariance Elements Magnitude (dB)';
h.Label.Interpreter = 'latex';

set(gca, 'FontSize', 12, 'FontName', 'Times', 'TickLabelInterpreter', 'latex')
set(h, 'FontSize', 12, 'FontName', 'Times', 'TickLabelInterpreter', 'latex')

%% ================================================================
% Singular Value Distribution of Clutter+Noise covariance matrix
% ================================================================
figure(3)
semilogy(10*log10(sigma^2*svd(Cov)),LineWidth=2);
xlabel('Singular Value Index',Interpreter='latex');
ylabel('Scaled Singular Magnitude (dB)',Interpreter='latex');
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);

% Setup for STAP covariance analysis
p = M * N;
R = Cov;
inv_R = pinv(R);

%% ================================================================
% Build weight matrices and index sets for Banded+Spiked estimation
% ================================================================
w = createWeightMatrix(p);
s_m = storeIndexSets(p, p - 1);
W_struct = generateWeightMatrices(p);
bandwidth=30; % Band size for TABASCO

%% ================================================================
% Monte Carlo setup
% - Vary number of samples (2^7 to 2^9)
% - Evaluate SCNR vs alpha (regularization parameter)
% ================================================================
Monte = 100; % Monte Carlo runs
Num_samples = floor(2 .^ linspace(7, 9, 7));

H_c_transpose_H_c=zeros(p,p,length(Num_samples));
H_c_cells = cell(length(Num_samples),1);

% Extract clutter matrices for different sample sizes
for index=1:length(Num_samples)
    if mod(Num_samples(index),2)==1
        H_c_k=(reshape(Clutter_Data(1:M,:,RangeGate-(Num_samples(index)-1)/2:RangeGate+(Num_samples(index)-1)/2),p,(Num_samples(index))));
    end
    if mod(Num_samples(index),2)==0
        H_c_k=(reshape(Clutter_Data(1:M,:,RangeGate-(Num_samples(index))/2:RangeGate+(Num_samples(index))/2-1),p,(Num_samples(index))));
    end
    H_c_cells{index}=H_c_k;
    H_c_transpose_H_c(:,:,index)=H_c_k*H_c_k';
end

% Precompute Marchenko–Pastur medians
MP_median=zeros(size(Num_samples));
for i=1:length(Num_samples)
    MP_median(i)=compute_MP_median(p, Num_samples(i));
end

% Define scanning ranges
angle_rad = (-180:10:180) * pi / 180;  % Angles in radians
doppler = -0.5:0.05:0.5;               % Normalized Doppler frequencies
Num_alpha=[0.5,1,1.5,2,2.5,3];         % Regularization scaling

% Preallocate SCNR results
num_samples=length(Num_samples);
num_lambda=length(Num_alpha);
num_angles=length(angle_rad);
num_doppler=length(doppler);
Avg_Rho_band     = zeros(num_samples, num_lambda);             
Avg_Angle_band   = zeros(num_samples, num_lambda, num_angles); 
Avg_Doppler_band = zeros(num_samples, num_lambda, num_doppler);

%% ================================================================
% Monte Carlo loop: estimate SCNR for each alpha and sample size
% ================================================================
tic;
for j=1:length(Num_alpha)
    Rho_Total_band = zeros(Monte, length(Num_samples));
    Rho_Angle_band=zeros(Monte,length(Num_samples),length(angle_rad));
    Rho_Doppler_band=zeros(Monte,length(Num_samples),length(doppler));
    for monte = 1:Monte
        local_rho_band = zeros(1, length(Num_samples));  
        local_angle_band = zeros(length(Num_samples),length(angle_rad));  
        local_Doppler_band = zeros(length(Num_samples),length(doppler));  
        local_estimate=zeros(1,length(Num_samples));
        for i = 1:length(Num_samples)
            n = Num_samples(i);
            % Generate noisy sample covariance
            noise=sigma*(randn(p,n)+1i*randn(p,n))/sqrt(2);
            S=(H_c_transpose_H_c(:,:,i)+noise*noise')/(sigma*sigma*n);
            X=(H_c_cells{i}+noise)'/sigma;

            % Eigenvalue-based variance estimate
            [U,eig,V]=svd(S);
            eigen=diag(eig);
            if isnan(MP_median(i))
                local_estimate(i) = median(eigen);
            else
                local_estimate(i)=median(eigen)/MP_median(i);
            end
            hat_sigma_2=local_estimate(i);    

            % Initialize A_hat
            A_hat = struct();
            for l = 1:p - 1
                A_hat(l).hat_A = zeros(p, p);
            end

            % Regularized covariance estimation
            lambda = Num_alpha(j) * sqrt(log(p) / n);
            hat_R = struct();
            for l = 1:p - 1
                sum_mat = zeros(p, p);
                for l_prime = 1:p - 1
                    sum_mat = sum_mat + (W_struct(l_prime).W) .* (A_hat(l_prime).hat_A);
                end
                hat_R(l).R = (S) - lambda * sum_mat;
                nu = get_nu(lambda, w, l, hat_R(l).R, s_m);
                
                for m = 1:l
                    idx = sub2ind([p, p], s_m(m).indices(:, 1), s_m(m).indices(:, 2));
                    A_hat(l).hat_A(idx) = (w(l, m) / (lambda * (abs(w(l, m))^2 + max(nu, 0)))) * hat_R(l).R(idx);
                end
            end

            % Banded+Spiked estimator and inverse
            sum_mat = zeros(p);
            for l = 1:p - 1
                sum_mat = sum_mat + (W_struct(l).W) .* (A_hat(l).hat_A);
            end
            B_est = (S+hat_sigma_2*eye(p)) - lambda * sum_mat;
            inv_hat_R = pinv(B_est);

            % SCNR calculation
            [local_rho_band(i),local_angle_band(i,:), local_Doppler_band(i,:)]= SCNR_avg(inv_hat_R, R, inv_R, M, N,ChannelNumber);
            fprintf("Num Samples=%d,Monte=%d,Alpha Value=%f\n",n,monte,Num_alpha(j));
        end
        Rho_Total_band(monte, :) = local_rho_band;
        Rho_Angle_band(monte,:,:)=local_angle_band;
        Rho_Doppler_band(monte,:,:)=local_Doppler_band;
    end

    % Average across Monte Carlo runs
    Avg_Rho_band(:, j)           = mean(Rho_Total_band, 1);
    Avg_Angle_band(:, j, :)      = squeeze(mean(Rho_Angle_band, 1));
    Avg_Doppler_band(:, j, :)    = squeeze(mean(Rho_Doppler_band, 1));
end

%% ================================================================
% Plot 1: SCNR vs Number of Samples for all lambda
% ================================================================
f4 = figure(4); hold on;
markers = {'o','s','d','x','*','^'};  

for j = 1:length(Num_alpha)
    semilogx(Num_samples, 10*log10(Avg_Rho_band(:,j)), ...
        'LineWidth', 2, 'MarkerSize', 8, 'Marker', markers{j});
end

set(gca, 'XScale', 'log');
xlabel('Number of Samples', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Normalized SCNR (dB)', 'Interpreter', 'latex', 'FontSize', 14);
grid on;
lambda_legend = arrayfun(@(x) sprintf('$\\alpha = %.1f$', x), Num_alpha, 'UniformOutput', false);
legend(lambda_legend, 'Interpreter', 'latex', 'FontSize', 12, 'Location', 'southeast');
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);
save_current_fig(f4, 'LambdaVar', 'F:\Codes\Transient\Revision Codes\Revised Figures');

%% ================================================================
% Plot 2: SCNR vs Azimuth (at Num_samples index = 2)
% ================================================================
f5 = figure(5); hold on;
for j = 1:length(Num_alpha)
    plot(angle_rad, 10*log10(squeeze(Avg_Angle_band(2,j,:))), ...
        'LineWidth', 2, 'MarkerSize', 8, 'Marker', markers{j});
end
xlabel('Azimuth (rad)', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Normalized SCNR (dB)', 'Interpreter', 'latex', 'FontSize', 14);
grid on;
legend(lambda_legend, 'Interpreter', 'latex', 'FontSize', 12, 'Location', 'southeast');
xlim([-pi, pi]);
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);
save_current_fig(f5, 'LambdaVar', 'F:\Codes\Transient\Revision Codes\Revised Figures');

%% ================================================================
% Plot 3: SCNR vs Doppler (at Num_samples index = 2)
% ================================================================
f6 = figure(6); hold on;
for j = 1:length(Num_alpha)
    plot(doppler, 10*log10(squeeze(Avg_Doppler_band(2,j,:))), ...
        'LineWidth', 2, 'MarkerSize', 8, 'Marker', markers{j});
end
xlabel('Normalized Doppler', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Normalized SCNR (dB)', 'Interpreter', 'latex', 'FontSize', 14);
grid on;
legend(lambda_legend, 'Interpreter', 'latex', 'FontSize', 12, 'Location', 'southeast');
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);
save_current_fig(f6, 'LambdaVar', 'F:\Codes\Transient\Revision Codes\Revised Figures');

%% ================================================================
% Report elapsed runtime
% ================================================================
elapsedTime = toc;   
fprintf('Total runtime: %.4f seconds\n', elapsedTime);
%% ================================================================
% Helper functions 
% ================================================================

%%
function w = createWeightMatrix(p)
    % Initialize the p-by-p weight matrix with zeros
    w = zeros(p, p);
    
    % Loop over the rows (l) and columns (m)
    for l = 1:p
        for m = 1:l
            % Compute the weight value according to the formula
            w(l, m) = sqrt(2 * l) / (l - m + 1);
        end
    end
end
%%
function g_l = generateIndices(p, l)
    % Initialize an empty set to store the indices
    g_l = [];
    
    % Loop over each m from 1 to l
    for m = 1:l
        % Calculate the required difference for the current m
        diff = p - m;
        
        % Generate all valid (j, k) pairs with |j - k| = diff
        for j = 1:p
            k1 = j + diff;  % Potential (j, k) with k > j
            k2 = j - diff;  % Potential (j, k) with k < j
            
            % Check if (j, k1) is within valid matrix bounds
            if k1 >= 1 && k1 <= p
                g_l = [g_l; j, k1];  % Add the pair (j, k1) to the set
            end
            
            % Check if (j, k2) is within valid matrix bounds and different from k1
            if k2 >= 1 && k2 <= p && k1~=k2
                g_l = [g_l; j, k2];  % Add the pair (j, k2) to the set
            end
        end
    end
end
%%
function g = storeIndexSets(p, l)
    % Initialize an empty structure to store index sets
    g = struct();
    
    % Loop over each m from 1 to l
    for m = 1:l
        % Calculate the difference for the current m
        diff = p - m;
        
        % Initialize a list to hold the (j, k) pairs for current m
        indexSet = [];

        % Generate all valid (j, k) pairs where |j - k| = diff
        for j = 1:p
            k1 = j + diff;  % Potential pair (j, k) with k > j
            k2 = j - diff;  % Potential pair (j, k) with k < j
            
            % Add valid pairs to the index set
            if k1 >= 1 && k1 <= p
                indexSet = [indexSet; j, k1];  % Append (j, k1)
            end
            if k2 >= 1 && k2 <= p && k2 ~= k1
                indexSet = [indexSet; j, k2];  % Append (j, k2)
            end
        end

        % Store the index set in the structure
        g(m).indices = indexSet;
    end
end
%%
function W_struct = generateWeightMatrices(p)
    % Initialize the structure to hold the matrices
    W_struct = struct();

    % Loop over all 1 <= l <= p - 1
    for l = 1:p - 1
        % Initialize the p x p matrix with zeros for the current l
        W = zeros(p, p);
        
        % Loop over all valid m <= l
        for m = 1:l
            % Compute the scalar weight w_{lm}
            %w_lm = (sqrt(2 * l) / (l - m + 1))*exp(1i*2*(l-m)*pi/(p-1));
            w_lm = sqrt(2 * l) / (l - m + 1);
            % Compute the difference needed for this m
            diff = p - m;

            % Populate the appropriate entries in the matrix
            for j = 1:p
                k1 = j + diff;  % Potential index with k > j
                k2 = j - diff;  % Potential index with k < j

                % If within bounds, assign the weight w_{lm}
                if k1 >= 1 && k1 <= p
                    W(j, k1) = conj(w_lm);
                end
                if k2 >= 1 && k2 <= p
                    W(j, k2) = w_lm;
                end
            end
        end

        % Store the generated matrix in the structure
        W_struct(l).W = W;
    end
end
%%
function isToeplitz = checkToeplitz(A)
    [n, m] = size(A);
    
    isToeplitz = true;  % Assume it is Toeplitz unless proven otherwise
    
    % Loop through all diagonals starting from the first column
    for k = -(n-1):(m-1)
        % Extract the k-th diagonal
        diag_elements = diag(A, k);
        
        % Check if all elements in the diagonal are identical
        if ~all(diag_elements == diag_elements(1))
            isToeplitz = false;
            return;  % Exit as soon as a violation is found
        end
    end
end
%% 
function nu_root=get_nu(lambda,w,l,R,s_m)
obv=@(nu) lambda^2-objective(nu,w,l,R,s_m);
% Use fzero to find the root of the objective function
nu_root = fzero(obv,w(size(w,1),1)^2);  % Start with initial guess nu = 0.1
end
function obv=objective(nu,w,l,R,s_m)
obv=0;
for m=1:l
    obv=obv+(w(l,m)^2)*norm(R(sub2ind(size(R),s_m(m).indices(:,1),s_m(m).indices(:,2))),2)^2/((w(l,m)^2+nu)^2);
end
end
%%
function B = generateBandedPSDMatrix(p, bandwidth, seed)
    % Generate a random banded positive semi-definite (PSD) matrix
    % Inputs:
    %   p         - Size of the matrix (p x p)
    %   bandwidth - Half-bandwidth (controls the band size)
    %   seed      - Random seed for reproducibility
    % Output:
    %   B         - Banded PSD matrix (p x p)

%     % Fix the random seed for reproducibility
%     rng(seed);
% 
%     % Generate a random symmetric matrix
%     A = randn(p, p);
%     A = (A + A') / 2;  % Make A symmetric
% 
%     % Make the matrix positive semi-definite (PSD)
%     A = A * A';  % Now A is symmetric positive semi-definite
% 
%     % Apply the banded structure
%     B = zeros(p, p);  % Initialize the banded matrix
%     for i = 1:p
%         for j = max(1, i - bandwidth):min(p, i + bandwidth)
%             B(i, j) = A(i, j);  % Copy elements within the bandwidth
%         end
%     end
% 
%     % Ensure symmetry (optional, since B is derived from a symmetric matrix)
%     B = (B + B') / 2;
B=zeros(p,p);
for i=1:p
    for j=1:p
        if abs(i-j)<bandwidth
        B(i,j)=1-(1/bandwidth)*abs(i-j)*exp(-1i*pi*(i-j)/p);
        end
    end
end
end
function [X_centered,S] = generateSampleCovariance(B, n)
    % Generate a sample covariance matrix from the banded PSD matrix
    % Inputs:
    %   B    - Banded PSD matrix (p x p)
    %   n    - Number of samples
    %   seed - Random seed for reproducibility
    % Output:
    %   S    - Sample covariance matrix (p x p)

    % Get the dimension of the banded matrix
    p = size(B, 1);

    % Generate n samples from a multivariate normal distribution
    % with mean zero and covariance B
    X = mvnrnd(zeros(1, p), B, n);  % (n x p) sample matrix

    % Compute the sample covariance matrix S manually
    X_centered = X - mean(X, 1);  % Center the data (subtract column mean)
    S = (X_centered' * X_centered) / (n - 1);  % Compute covariance matrix
end
%%
function [avg_rho, avg_Angle, avg_Doppler] = SCNR_avg(inv_hat_R, R, inv_R, M, N,ChannelNumber)
% SCNR_avg - Compute the average normalized SCNR over angles and Doppler frequencies
%
% Input:
%   inv_hat_R - Inverse of estimated covariance matrix (M*N x M*N)
%   R         - True covariance matrix (M*N x M*N)
%   inv_R     - Inverse of true covariance matrix (M*N x M*N)
%   M         - Number of spatial channels (antenna elements)
%   N         - Number of pulses in the CPI
%
% Output:
%   avg_rho   - Average normalized SCNR

    % Define angle and Doppler ranges
    angle_rad = (-180:10:180) * pi / 180;  % Angles in radians
    doppler = -0.5:0.05:0.5;  % Normalized Doppler frequencies

    % Pre-allocate the matrix to store SCNR values
    rho_mat = zeros(length(doppler), length(angle_rad));

    % Parallel loop over Doppler frequencies and angles
    parfor i = 1:length(doppler)
        local_rho = zeros(1, length(angle_rad));  % Local variable for parallel loop

        for j = 1:length(angle_rad)
            % Compute SCNR for each angle and Doppler frequency
            local_rho(j) = SCNR_calc(inv_hat_R, R, inv_R, M, N, angle_rad(j), doppler(i),ChannelNumber);
        end

        % Store the local results in the matrix
        rho_mat(i, :) = local_rho;
    end
    
    % Compute the average SCNR
    avg_rho = sum(rho_mat, 'all') / numel(rho_mat);
    avg_Doppler=transpose(sum(rho_mat,2))/length(angle_rad);
    avg_Angle=sum(rho_mat,1)/length(doppler);
end

function normalized_rho = SCNR_calc(inv_hat_R, R, inv_R, M, N, theta, nu,ChannelNumber)
% SCNR_calc - Compute the normalized SCNR for STAP
%
% Input: 
%   M          - Number of spatial channels (antenna elements)
%   N          - Number of pulses in the CPI
%   inv_hat_R  - Inverse of estimated covariance matrix (M*N x M*N)
%   R          - True covariance matrix (M*N x M*N)
%   inv_R      - Inverse of true covariance matrix (M*N x M*N)
%   theta      - Angle of arrival (AoA) in radians
%   nu         - Normalized Doppler frequency [-0.5, 0.5]
%
% Output:
%   normalized_rho - Normalized SCNR

    % Wavelength and element spacing (assuming half-wavelength spacing)
    lambda = 1;  % Normalized wavelength
    d = lambda / 2;  % Element spacing

    % Generate the angle steering vector (M x 1)
    a = angle_steering_vector(M, theta, lambda, d,ChannelNumber);

    % Generate the Doppler steering vector (N x 1)
    d_vec = doppler_steering_vector(nu, N);

    % Compute the Kronecker product of the two steering vectors
    w_st = kron(a, d_vec);  % Space-time steering vector (M*N x 1)
    
    % Calculate the normalized SCNR
    numerator = abs(w_st' * inv_hat_R * w_st)^2;
    denominator = abs((w_st' * inv_R * w_st)) * abs(w_st' * inv_hat_R * R * inv_hat_R * w_st);
    
    normalized_rho = numerator / denominator;
end

function a = angle_steering_vector(M, theta, lambda, d,ChannelNumber)
    % angle_steering_vector - Compute the angle steering vector for ULA
    % Inputs:
    %   M      - Number of antenna elements (channels)
    %   theta  - Angle of arrival (AoA) in radians
    %   lambda - Wavelength of the signal
    %   d      - Spacing between elements (typically lambda/2)
    % Output:
    %   a - Angle steering vector (M x 1)

    % Compute phase shifts for each antenna element
    if M==1
    phase_shifts = 2 * pi * d / lambda * sin(theta) * (ChannelNumber).';
    end
    if M~=1
    phase_shifts = 2 * pi * d / lambda * sin(theta) * (0:M-1).';
    end
    a = exp(1j * phase_shifts);  % Complex-valued steering vector
end

function d_vec = doppler_steering_vector(nu, N)
    % doppler_steering_vector - Compute Doppler steering vector
    % Inputs:
    %   nu - Normalized Doppler frequency [-0.5, 0.5]
    %   N  - Number of pulses in the CPI
    % Output:
    %   d_vec - Doppler steering vector (N x 1)

    % Generate the Doppler steering vector
    d_vec = exp(1j * 2 * pi * nu * (0:N-1)).';
end
%% Computation of the Median of MP Distribution
function median_MP = compute_MP_median(M, N, sigma)
    % M: dimensionality (number of rows)
    % N: number of samples (number of columns)
    % sigma: variance of noise (optional, default is 1)
    
    if nargin < 3
        sigma = 1;  % Default variance is 1 if not provided
    end

    % Compute the aspect ratio gamma
    gamma = M / N;

    % Marchenko-Pastur bounds for the eigenvalues
    lambda_min = sigma^2 * (1 - sqrt(gamma))^2;
    lambda_max = sigma^2 * (1 + sqrt(gamma))^2;

    % Define the Marchenko-Pastur PDF
    mp_pdf = @(lambda) (1 / (2 * pi * gamma * lambda * sigma^2)) * ...
        sqrt((lambda_max - lambda) .* (lambda - lambda_min)) .* ...
        (lambda >= lambda_min & lambda <= lambda_max);

    % Define the CDF function by numerically integrating the PDF
    cdf = @(lambda) integral(mp_pdf, lambda_min, lambda, 'ArrayValued', true);

    % Find the median (CDF(lambda) = 0.5) by using fzero
    median_MP = fzero(@(lambda) cdf(lambda) - 0.5, (lambda_min + lambda_max) / 2);

    % Display the result
end
%% Stein Shrinkage Algorithm
function eigenvals = optshrink_impl(eigenvals,gamma,loss,sigma)

    sigma2 = sigma^2;
    lam_plus = (1+sqrt(gamma))^2;
    
    ell = @(lam) ((lam>=lam_plus).*((lam+1-gamma) + sqrt((lam+1-gamma).^2-4*lam))/(2.0));

    c = @(lam)((lam>=lam_plus) .* sqrt( (1-gamma./((ell(lam)-1).^2) ) ./ (1+gamma./(ell(lam)-1) )   ));
    s = @(lam)(sqrt(1-c(lam).^2 ) );

    impl_F_1 = @(ell,c,s)(max(1+(c.^2).*(ell-1),0));
    impl_F_2 = @(ell,c,s)(max(ell./((c.^2)+ell.*(s.^2)),0));
    impl_F_3 = @(ell,c,s)(max(1 + (ell-1).*((c.^2)./((ell.^2).*(s.^2) + c.^2 )),1));
    impl_F_4 = @(ell,c,s)((s.^2 + (ell.^2).*(c.^2))./((s.^2) + (ell.*(c.^2))));
    impl_F_6 = @(ell,c,s)(1 + ((ell-1).*(c.^2)) ./ (((c.^2)+ell.*(s.^2)).^2));
    impl_O_1 = @(ell,c,s)(ell); % this is just debiasing to pop eigen
    impl_O_2 = @(ell,c,s)(ell); % this is just debiasing to pop eigen
    impl_O_6 = @(ell,c,s)(1+((ell-1)./(c.^2 + ell.*(s.^2)))); 
    impl_N_1 = @(ell,c,s)(max(1+(ell-1).*(1-2*(s.^2)),1));
    impl_N_2 = @(ell,c,s)(max(ell./((2*ell-1).*(s.^2) + c.^2),1));
    impl_N_3 = @(ell,c,s)(max(ell./(c.^2+(ell.^2).*(s.^2)),1));
    impl_N_4 = @(ell,c,s)(max(ell .* (c.^2) + (s.^2) ./ ell,1));
    impl_N_6 = @(ell,c,s)(max((ell - ((ell-1).^2) .* ...
                           (c.^2) .* (s.^2))./(((c.^2) + ell.*(s.^2)).^2),1));
    impl_Stein =  @(ell,c,s)(ell./(c.^2 + ell.*(s.^2))); 
    impl_Ent =  @(ell,c,s)(ell.*(c.^2) + s.^2); 
    impl_Div =  @(ell,c,s)(sqrt(((ell.^2).*(c.^2)+ell.*(s.^2))./(c.^2+(s.^2).*ell))); 
    impl_Fre =  @(ell,c,s)((sqrt(ell).*(c.^2)+s.^2).^2); 
    impl_Aff =  @(ell,c,s)( ((1+c.^2).*ell + (s.^2)) ./ (1+(c.^2)+ell.*(s.^2))); 

    assert(sigma>0)
    assert(prod(size(sigma))==1)
    eigenvals = eigenvals / sigma2;
    I = (eigenvals > lam_plus);
    eigenvals(~I) = 1;
    str1 = [loss '_func = @(lam)(max(1,impl_' loss '(ell(lam),c(lam),s(lam))));'];
    str2 = ['eigenvals(I)= ' loss '_func(eigenvals(I));'];
    eval(str1);
    eval(str2);
    eigenvals = sigma2 * eigenvals;
end
%%
function save_current_fig(figHandle, prefix, outDir)
    % Ensure output folder exists
    if ~exist(outDir, 'dir'); mkdir(outDir); end

    ax   = get(figHandle,'CurrentAxes');
    xlab = get(get(ax,'XLabel'),'String');
    ylab = get(get(ax,'YLabel'),'String');

    % Flatten cell array labels if needed
    if iscell(xlab), xlab = strjoin(xlab, ''); end
    if iscell(ylab), ylab = strjoin(ylab, ''); end

    % Clean up spaces and LaTeX symbols
    clean = @(s) regexprep(s, '\\|{|}|\$|\^|_|~|\s+', '');
    xlab_clean = clean(xlab);
    ylab_clean = clean(ylab);

    tstamp = datestr(now,'yyyymmddHHMM');

    % Build filename
    fname = sprintf('%s%svs%st%s.fig', prefix, ylab_clean, xlab_clean, tstamp);

    % Save the .fig
    savefig(figHandle, fullfile(outDir, fname));
end


function filename = generate_filename(ylabel_text, xlabel_text)
    % Replaces underscores with spaces and removes special LaTeX formatting
    ylabel_text = strrep(ylabel_text, '_', ' ');
    xlabel_text = strrep(xlabel_text, '_', ' ');
    
    % Get current date
    current_date = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    
    % Create the filename with the desired format
    filename = sprintf('%s vs %s Cofar256Journal %s', ylabel_text, xlabel_text, current_date);
end