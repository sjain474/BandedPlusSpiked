function hat_Sigma=TOBASCO(X,S,p,n,bandwidth)
% Compute key metrics for the TOBASCO algorithm
kappa_hat = compute_kappa_hat(X);
[Delta_hat, D_Delta, hat_gamma] = compute_gamma(X);
eta_hat = trace(S) / p;

% Generate the sequences of weight and V matrices
W_seq = generate_weight_matrices(p, bandwidth);
V_seq = generate_v_matrices(p, bandwidth);

% Initialize placeholders for metrics
theta_hat_w = zeros(1, bandwidth);
gamma_hat_v = zeros(1, bandwidth);
gamma_hat_w = zeros(1, bandwidth);
hat_beta_0 = zeros(1, bandwidth);

% Loop over each bandwidth value and compute metrics
for j = 1:bandwidth
    theta_hat_w(j) = compute_theta_w(D_Delta, W_seq{j});
    gamma_hat_v(j) = compute_gamma_v(Delta_hat, D_Delta, V_seq{j}, n);
    gamma_hat_w(j) = compute_gamma_v(Delta_hat, D_Delta, W_seq{j}, n);
    hat_beta_0(j) = compute_beta_0(theta_hat_w(j), gamma_hat_v(j), ...
                                   gamma_hat_w(j), eta_hat, ...
                                   hat_gamma, kappa_hat, n, p);
end

% Find the optimal bandwidth k_0
[~, k_0] = min(hat_beta_0 .* (1 - gamma_hat_v));

% Estimate the covariance matrix using the optimal bandwidth
hat_Sigma = hat_beta_0(k_0) * (W_seq{k_0} .* S) + ...
            (1 - hat_beta_0(k_0)) * eta_hat * eye(p);
end

%%
function B = generateTaperedPSDMatrix(p, bandwidth, seed)
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
B=eye(p);
rho=0.2;
for i=1:p
    for j=1:p
        if i~=j
        B(i,j)=rho^(abs(i-j));
        end
    end
end
end
%%
function [X_centered, S] = generateSampleCovariance(B, n)
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
function W_seq = generate_weight_matrices(p, K)
    % Generates a sequence of weight matrices W(k) for k = 1,...,K
    % Inputs:
    %   p  - Dimension of the matrix (p x p)
    %   K  - Maximum bandwidth parameter
    % Output:
    %   W_seq - Cell array containing the sequence of weight matrices

    W_seq = cell(K, 1);  % Initialize a cell array to store matrices

    for k = 1:K
        W_seq{k} = create_weight_matrix(p, k);
    end
end

function W = create_weight_matrix(p, k)
    % Creates a single weight matrix W for a given bandwidth k
    W = zeros(p, p);  % Initialize the weight matrix

    for i = 1:p
        for j = 1:p
            dist = abs(i - j);  % Distance between indices
            if dist <= k / 2
                W(i, j) = 1;  % Set weight to 1
            elseif dist < k
                W(i, j) = 2 - 2 * dist / k;  % Linear decay
            else
                W(i, j) = 0;  % Set weight to 0
            end
        end
    end
end
%%
function V_seq = generate_v_matrices(p, K)
    % Generates a sequence of weight matrices W(k) for k = 1,...,K
    % Inputs:
    %   p  - Dimension of the matrix (p x p)
    %   K  - Maximum bandwidth parameter
    % Output:
    %   W_seq - Cell array containing the sequence of weight matrices

    V_seq = cell(K, 1);  % Initialize a cell array to store matrices

    for k = 1:K
        V_seq{k} = create_V_matrix(p, k);
    end
end

function V = create_V_matrix(p, k)
    % Creates a single weight matrix W for a given bandwidth k
    V = zeros(p, p);  % Initialize the weight matrix

    for i = 1:p
        for j = 1:p
            dist = abs(i - j);  % Distance between indices
            if dist <= k / 2
                V(i, j) = 1;  % Set weight to 1
            elseif dist < k
                V(i, j) = sqrt(2 - 2 * dist / k);  % Linear decay
            else
                V(i, j) = 0;  % Set weight to 0
            end
        end
    end
end
%%
function kappa_hat = compute_kappa_hat(X)
    % Compute the kappa_hat for real or complex data.
    % Inputs:
    %   X - Data matrix of size (n x p), where n is the number of samples and
    %       p is the number of features (dimensions).
    % Output:
    %   kappa_hat - Estimated kappa value (adjusted for real or complex data).

    [n, p] = size(X);  % Get the dimensions of the input data
    m4 = zeros(1, p);  % Initialize m_4(j) for all features
    m2 = zeros(1, p);  % Initialize m_2(j) for all features
    hat_K = zeros(1, p);  % Initialize hat_K(j) for all features

    % Check if the data is real or complex
    isComplex = ~isreal(X);

    % Compute m_4(j) and m_2(j) for each feature j
    for j = 1:p
        x_j = X(:, j);  % Extract the j-th feature/column
        mean_j = mean(x_j);  % Mean of the j-th feature

        % Compute m_4(j) and m_2(j)
        m4(j) = mean(abs(x_j - mean_j).^4);  % Fourth central moment
        m2(j) = mean(abs(x_j - mean_j).^2);  % Second central moment

        % Compute hat_k(j) with adjustment for complex data
        if isComplex
            hat_k_j = (m4(j) / (m2(j)^2)) - 2;  % Complex case
        else
            hat_k_j = (m4(j) / (m2(j)^2)) - 3;  % Real case
        end

        % Compute hat_K(j)
        hat_K(j) = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * hat_k_j + 6);
    end

    % Compute kappa_hat
    kappa_hat = max(-2 / (p + 2), (1 / (3 * p)) * sum(hat_K));
end
%%
function [Delta_hat,D_Delta,hat_gamma]=compute_gamma(X)
[n,p]=size(X);
Delta_hat=zeros(p,p);
for j=1:n
    Delta_hat=Delta_hat+(X(j,:)'*X(j,:))/norm(X(j,:))^2;
end
Delta_hat=(p/n)*Delta_hat;
[U,D_Delta,V]=svd(Delta_hat);
hat_gamma=(n/(n-1))*(((norm(Delta_hat,"fro"))^2)/p-p/n);
end
%%
function theta_w=compute_theta_w(D,W)
    % Compute theta_w given matrices D and W
    % Inputs:
    %   D - Diagonal matrix (p x p)
    %   W - Weight matrix W(k) (p x p)
    % Output:
    %   theta_w - Computed value of theta_w

p=size(W,1);
theta_w=trace((D*W)^2)/p^2;
end
%%
function gamma_hat_v=compute_gamma_v(Delta_hat,D_Delta,V,n)
p=size(V,1);
gamma_hat_v=(n/(n-1))*((norm(V.*Delta_hat,"fro")^2)/2-trace((D_Delta*V)^2)/(n*p));
end
function hat_beta_0=compute_beta_0(theta_hat_w,gamma_hat_v,gamma_hat_w,eta_hat,gamma_hat,kappa_hat,n,p)
t=n*(gamma_hat_v-1);
A=p*theta_hat_w/eta_hat^2-1+2*gamma_hat_v-2*gamma_hat/p;
hat_beta_0=t/(t+(n/(n-1))*(p*theta_hat_w/eta_hat^2+gamma_hat_w-2*gamma_hat/p)+kappa_hat*A);
end
