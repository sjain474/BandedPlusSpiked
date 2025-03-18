clear all;
clc;
%% Parameters
%p_values = [64,128, 256, 512];  % Different values of p to test
p=256;
c_values = logspace(log10(5), 2, 10); % Different values of c_value (log-spaced)
n_values=[64,128,256,512,1024];
c_values_dB = 10 * log10(c_values); % Convert to dB
numSpikes = floor(p/10);
bandwidth = 4;
Monte = 100; % Number of Monte Carlo simulations
epsilon_results = zeros(length(n_values), length(c_values));
Sigma = generateSpikedBandedHermitian(p, bandwidth, numSpikes);%Banded plus Spiked Matrix
lambda_max_Sigma = norm(Sigma, 2);
% Visualization
    figure;
    imagesc(10 * log10(abs(Sigma)));
    title(['Spiked Banded Hermitian Matrix for p = ', num2str(p)]);
    colorbar;
%% Parallel computation
parpool('local'); % Start parallel pool

for n_idx = 1:length(n_values)
    n = n_values(n_idx);
    
    

    
    %n = 1.5 * p; % Sample size
    local_lambda_max = zeros(1, Monte);
    local_trace = zeros(1, Monte);
    
    % Parallel Monte Carlo simulation
    parfor i = 1:Monte
        S = generateSampleCovariance(Sigma, n);
        local_trace(i) = trace(S);
        local_lambda_max(i) = norm(S, 2);
    end
    
    lambda_max_S = mean(local_lambda_max);
    trace_S = mean(local_trace);
    
    for c_idx = 1:length(c_values)
        c_value = c_values(c_idx);
        if n>=p
        epsilon_results(n_idx, c_idx) = sqrt(2*(p * (c_value + lambda_max_S)^2 + p * log10(lambda_max_S) + p - trace_S / c_value))/(c_value*sqrt(p));
        end
        if n<p
            epsilon_results(n_idx, c_idx) = 2*(p * (c_value + lambda_max_S) + p * log10(c_value) + p - trace_S / c_value)/(c_value*p);
        end
    end
end

delete(gcp('nocreate')); % Close parallel pool

%% Plot results
% Define different markers and line styles for better visualization
line_styles = {'-o', '--s', '-.^', ':d', '-x'}; 
marker_styles = {'o', 's', '^', 'd', 'x'};
figure;
hold on;
for n_idx = 1:length(n_values)
    plot(c_values_dB, (epsilon_results(n_idx, :)), line_styles{mod(n_idx-1, length(line_styles)) + 1}, ...
        'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', ['$K = ' num2str(n_values(n_idx)) '$'], ...
        'Marker', marker_styles{mod(n_idx-1, length(marker_styles)) + 1});
end

% Axis formatting
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12); % Set tick labels in LaTeX
xlabel('$c$ (dB)', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Relative Bound $\Big(\Big(\frac{2\epsilon}{p}\Big)^{\frac{1}{\gamma}}\cdot\frac{1}{c}\Big)$', 'Interpreter', 'latex', 'FontSize', 14);
%title('Variation of $\epsilon$ with $c$ for different $p$', 'Interpreter', 'latex', 'FontSize', 18);
legend('Interpreter', 'latex', 'FontSize', 14, 'Location', 'best');
grid on;

% Adjust x-limits to fit data range
xlim([min(c_values_dB) max(c_values_dB)]);
