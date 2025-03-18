function S = generateSampleCovariance(B, n)
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
    X_centered = X;  % Center the data (subtract column mean)
    S = (X_centered' )*( X_centered) / (n);  % Compute covariance matrix
end