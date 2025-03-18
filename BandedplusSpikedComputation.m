clear all 
%% Loading RFView DataSet
Data=load("Cofar.mat");% SIM contains the necessay parameters for the dataset
Noise=Data.SIM.nse;
[L,N,R]=size(Noise);
RangeGate=R/2+50;
M=4;%Number of channels used
ChannelNumber=5;
K=20*N;
sigma=2.5e-7;
Sample_CovNoise=sigma*(randn(M*N,K)+1i*randn(M*N,K))/sqrt(2);% Noise generation
Mean_Noise=mean(Sample_CovNoise,2);
x_n=Sample_CovNoise;
NSE=(x_n*x_n')/K;
[Un,sigma_2,Vn]=svd(x_n);
Clutter_Data=Data.SIM.xc_clut;% clutter impulse response convolved with waveform
if M==1 && ChannelNumber<=L %Singe Channel
    Hc=reshape(Clutter_Data(ChannelNumber,:,RangeGate-K/2:RangeGate+K/2-1),M*N,K);
end
if M~=1 %Multiple Channels
    Hc=reshape(Clutter_Data(1:M,:,RangeGate-K/2:RangeGate+K/2-1),M*N,K);
end
[U,D,V]=svd(Hc);
[U1,D1,V1]=svd(Hc+x_n);
%%
figure(1)
semilogy(10*log10(diag(D)));
hold on;
semilogy(10*log10(diag(sigma_2)));
hold on;
semilogy(10*log10(diag(D1))); 
hold off;
grid on;
% Set tick labels with LaTeX fonts
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);
xlim([1,M*N])
xlabel("Singular Value Index",Interpreter="latex");
ylabel("Singular Value (dB)",Interpreter="latex");
legend('Clutter','Noise','Clutter+Noise',Interpreter="latex");
%% Image Plot of Clutter Plus Noise Covariance Matrix
Cov=((Hc*Hc')/(sigma^2*K))+eye(M*N);
figure(2)
imagesc(10 * log10(sigma^2 * abs(Cov)))
h = colorbar; % Handle for colorbar
colormap('jet')

% Label the colorbar with LaTeX
h.Label.String = 'Covariance Elements Magnitude (dB)';
h.Label.Interpreter = 'latex';

% Set the font properties for ticks and colorbar
set(gca, 'FontSize', 12, 'FontName', 'Times', 'TickLabelInterpreter', 'latex')
set(h, 'FontSize', 12, 'FontName', 'Times', 'TickLabelInterpreter', 'latex')
%% SVD Plot of Clutter Plus Noise Covariance Matrix
figure(3)
semilogy(10*log10(sigma^2*svd(Cov)));
% Samples=(Hc+Sample_CovNoise)/sigma;
xlabel('Singular Value Index',Interpreter='latex');
ylabel('Scaled Singular Magnitude (dB)',Interpreter='latex');
% Set tick labels with LaTeX fonts
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);
%[~,P]=size(Samples);
p = M * N;
R = Cov;
inv_R = pinv(R);
%%
% Create weight matrices and store index sets
w = createWeightMatrix(p);
s_m = storeIndexSets(p, p - 1);
W_struct = generateWeightMatrices(p);
bandwidth=30;%bandsize for TABASCO
%% Monte Carlo parameters
Monte = 100;
Num_samples = floor(2 .^ linspace(7, 9, 7));
MP_median=zeros(size(Num_samples));
for i=1:length(Num_samples)
MP_median(i)=compute_MP_median(p, Num_samples(i));
end
% Define angle and Doppler ranges
angle_rad = (-180:10:180) * pi / 180;  % Angles in radians
doppler = -0.5:0.05:0.5;  % Normalized Doppler frequencies
Rho_Total_band = zeros(Monte, length(Num_samples));
Rho_Total_Diagonal = zeros(Monte, length(Num_samples));
Rho_Total_Stein = zeros(Monte, length(Num_samples));
Rho_Total_TOBASCO = zeros(Monte, length(Num_samples));
Rho_Angle_band=zeros(Monte,length(Num_samples),length(angle_rad));
Rho_Angle_Diagonal=zeros(Monte,length(Num_samples),length(angle_rad));
Rho_Angle_Stein=zeros(Monte,length(Num_samples),length(angle_rad));
Rho_Angle_TOBASCO=zeros(Monte,length(Num_samples),length(angle_rad));
Rho_Doppler_band=zeros(Monte,length(Num_samples),length(doppler));
Rho_Doppler_Diagonal=zeros(Monte,length(Num_samples),length(doppler));
Rho_Doppler_Stein=zeros(Monte,length(Num_samples),length(doppler));
Rho_Doppler_TOBASCO=zeros(Monte,length(Num_samples),length(doppler));
%% Monte Carlo Simulation
parfor monte = 1:Monte
    local_rho_band = zeros(1, length(Num_samples));  % Store errors for each run
    local_rho_Diagonal = zeros(1, length(Num_samples));  % Store errors for each run
    local_rho_Stein = zeros(1, length(Num_samples));  % Store errors for each run
    local_rho_TOBASCO = zeros(1, length(Num_samples));  % Store errors for each run
    local_angle_band = zeros(length(Num_samples),length(angle_rad));  % Store errors for each run
    local_angle_Diagonal = zeros(length(Num_samples),length(angle_rad));  % Store errors for each run
    local_angle_Stein = zeros(length(Num_samples),length(angle_rad));  % Store errors for each run
    local_angle_TOBASCO = zeros(length(Num_samples),length(angle_rad));  % Store errors for each run
    local_Doppler_band = zeros(length(Num_samples),length(doppler));  % Store errors for each run
    local_Doppler_Diagonal = zeros(length(Num_samples),length(doppler));  % Store errors for each run
    local_Doppler_Stein = zeros(length(Num_samples),length(doppler));  % Store errors for each run
    local_Doppler_TOBASCO = zeros(length(Num_samples),length(doppler));  % Store errors for each run
    local_estimate=zeros(1,length(Num_samples));
    for i = 1:length(Num_samples)
        n = Num_samples(i);
        [X,S] = generateSampleCovariance(R, n);
        [U,eig,V]=svd(S);
        eigen=diag(eig);
        if isnan(MP_median(i))
            local_estimate(i) = median(eigen);
            else
                local_estimate(i)=median(eigen)/MP_median(i);
        end
        stein_shrink=optshrink_impl(eigen,p/n,'Stein',sqrt(local_estimate(i)));
        inv_hat_Stein=U*diag(1./stein_shrink)*V';% Stein Shrinkage
        hat_sigma_2=local_estimate(i);    
        % Initialize A_hat
        A_hat = struct();
        for l = 1:p - 1
            A_hat(l).hat_A = zeros(p, p);
        end

        % Estimate covariance matrix
        lambda = 2 * sqrt(log(p) / n);
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
        
        % Estimate B and its inverse
        sum_mat = zeros(p);
        for l = 1:p - 1
            sum_mat = sum_mat + (W_struct(l).W) .* (A_hat(l).hat_A);
        end
        B_est = (S+hat_sigma_2*eye(p)) - lambda * sum_mat;%Banded plus Spiked Matrix
        inv_hat_R = pinv(B_est);%Inverse Banded plus Spiked Matrix
        tobasco_Sigma=TOBASCO(X,S,p,n,bandwidth);%TABASCO
        inv_hat_TOBASCO=pinv(tobasco_Sigma);
        factor=median(eigen)/MP_median(i);
        % Compute SCNR for the current sample size
        [local_rho_band(i),local_angle_band(i,:), local_Doppler_band(i,:)]= SCNR_avg(inv_hat_R, R, inv_R, M, N,ChannelNumber);
        [local_rho_Diagonal(i),local_angle_Diagonal(i,:),local_Doppler_Diagonal(i,:)] = SCNR_avg(pinv(S+factor*eye(p)), R, inv_R, M, N,ChannelNumber);
        [local_rho_Stein(i),local_angle_Stein(i,:),local_Doppler_Stein(i,:)] = SCNR_avg(inv_hat_Stein, R, inv_R, M, N,ChannelNumber);
        [local_rho_TOBASCO(i),local_angle_TOBASCO(i,:),local_Doppler_TOBASCO(i,:)] = SCNR_avg(inv_hat_TOBASCO, R, inv_R, M, N,ChannelNumber);
    end
    Rho_Total_band(monte, :) = local_rho_band;
    Rho_Total_Diagonal(monte, :) = local_rho_Diagonal;
    Rho_Total_Stein(monte, :) = local_rho_Stein;
    Rho_Total_TOBASCO(monte, :) = local_rho_TOBASCO;
    Rho_Angle_band(monte,:,:)=local_angle_band;
    Rho_Angle_Diagonal(monte,:,:)=local_angle_Diagonal;
    Rho_Angle_Stein(monte,:,:)=local_angle_Stein;
    Rho_Angle_TOBASCO(monte,:,:)=local_angle_TOBASCO;
    Rho_Doppler_band(monte,:,:)=local_Doppler_band;
    Rho_Doppler_Diagonal(monte,:,:)=local_Doppler_Diagonal;
    Rho_Doppler_Stein(monte,:,:)=local_Doppler_Stein;
    Rho_Doppler_TOBASCO(monte,:,:)=local_Doppler_TOBASCO;
end

%% Compute the average error across all Monte Carlo runs
Avg_Rho_band = mean(Rho_Total_band, 1);
Avg_Rho_Diagonal = mean(Rho_Total_Diagonal, 1);
Avg_Rho_Stein=mean(Rho_Total_Stein, 1);
Avg_Rho_TOBASCO=mean(Rho_Total_TOBASCO, 1);
%% Angle: Compute the average error across all Monte Carlo runs
Avg_Angle_band = squeeze(mean(Rho_Angle_band, 1));
Avg_Angle_Diagonal = squeeze(mean(Rho_Angle_Diagonal, 1));
Avg_Angle_Stein=squeeze(mean(Rho_Angle_Stein, 1));
Avg_Angle_TOBASCO=squeeze(mean(Rho_Angle_TOBASCO, 1));
%% Doppler: Compute the average error across all Monte Carlo runs
Avg_Doppler_band = squeeze(mean(Rho_Doppler_band, 1));
Avg_Doppler_Diagonal = squeeze(mean(Rho_Doppler_Diagonal, 1));
Avg_Doppler_Stein=squeeze(mean(Rho_Doppler_Stein, 1));
Avg_Doppler_TOBASCO=squeeze(mean(Rho_Doppler_TOBASCO, 1));
%% Plot the results with LaTeX fonts
figure;
semilogx(Num_samples, 10*log10(Avg_Rho_band), 'LineWidth', 2, 'MarkerSize', 8, 'Marker', 'o');
hold on;
semilogx(Num_samples, 10*log10(Avg_Rho_Diagonal), 'LineWidth', 2, 'MarkerSize', 8, 'Marker', '*');
hold on;
semilogx(Num_samples, 10*log10(Avg_Rho_Stein), 'LineWidth', 2, 'MarkerSize', 8, 'Marker', 'x');
hold on;
semilogx(Num_samples, 10*log10(Avg_Rho_TOBASCO), 'LineWidth', 2, 'MarkerSize', 8, 'Marker', 'diamond');

set(gca, 'XScale', 'log');
xlabel('Number of Samples', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Normalized SCNR (dB)', 'Interpreter', 'latex', 'FontSize', 14);
%title('Normalized SCNR vs. Number of Samples', 'Interpreter', 'latex', 'FontSize', 16);
grid on;
legend({'Banded+Spiked', 'Diagonal Loading', 'Spiked', 'TABASCO'}, ...
       'Interpreter', 'latex', 'FontSize', 12, 'Location', 'southeast');
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);

%% Plot the Angle with LaTeX fonts
figure;
plot(angle_rad, 10*log10(Avg_Angle_band(2,:)), 'LineWidth', 2, 'MarkerSize', 8,'Marker','o');
hold on;
plot(angle_rad, 10*log10(Avg_Angle_Diagonal(2,:)), 'LineWidth', 2, 'MarkerSize', 8,'Marker','*');
hold on;
plot(angle_rad, 10*log10(Avg_Angle_Stein(2,:)), 'LineWidth', 2, 'MarkerSize', 8,'Marker','x');
hold on;
plot(angle_rad, 10*log10(Avg_Angle_TOBASCO(2,:)), 'LineWidth', 2, 'MarkerSize', 8,'Marker','diamond');

% Set axis labels and title with LaTeX interpreter
xlabel('Azimuth (rad)', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Normalized SCNR (dB)', 'Interpreter', 'latex', 'FontSize', 14);
%title('Normalized SCNR vs. Azimuth', 'Interpreter', 'latex', 'FontSize', 16);
grid on;

% Set legend with LaTeX interpreter and location in southwest
legend({'Banded+Spiked', 'Diagonal Loading', 'Spiked', 'TABASCO'}, ...
       'Interpreter', 'latex', 'FontSize', 12, 'Location', 'southeast');
xlim([-pi,pi]);
% Set tick labels with LaTeX fonts
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);
%% Plot the Doppler with LaTeX fonts
figure;
plot(doppler, 10*log10(Avg_Doppler_band(2,:)), 'LineWidth', 2, 'MarkerSize', 8,'Marker','o');
hold on;
plot(doppler, 10*log10(Avg_Doppler_Diagonal(2,:)), 'LineWidth', 2, 'MarkerSize', 8,'Marker','*');
hold on;
plot(doppler, 10*log10(Avg_Doppler_Stein(2,:)), 'LineWidth', 2, 'MarkerSize', 8,'Marker','x');
hold on;
plot(doppler, 10*log10(Avg_Doppler_TOBASCO(2,:)), 'LineWidth', 2, 'MarkerSize', 8,'Marker','diamond');

% Set axis labels and title with LaTeX interpreter
xlabel('Normalized Doppler', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Normalized SCNR (dB)', 'Interpreter', 'latex', 'FontSize', 14);
%title('Normalized SCNR vs. Doppler', 'Interpreter', 'latex', 'FontSize', 16);
grid on;

% Set legend with LaTeX interpreter and location in southwest
legend({'Banded+Spiked', 'Diagonal Loading', 'Spiked', 'TABASCO'}, ...
       'Interpreter', 'latex', 'FontSize', 12, 'Location', 'southeast');

% Set tick labels with LaTeX fonts
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);
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
function filename = generate_filename(ylabel_text, xlabel_text)
    % Replaces underscores with spaces and removes special LaTeX formatting
    ylabel_text = strrep(ylabel_text, '_', ' ');
    xlabel_text = strrep(xlabel_text, '_', ' ');
    
    % Get current date
    current_date = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    
    % Create the filename with the desired format
    filename = sprintf('%s vs %s Cofar256Journal %s', ylabel_text, xlabel_text, current_date);
end
