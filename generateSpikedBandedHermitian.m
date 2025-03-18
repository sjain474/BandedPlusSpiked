function Sigma = generateSpikedBandedHermitian(p, bandwidth, numSpikes)
    % GENERATESPIKEDBANDEDHERMITIAN Creates a spiked covariance matrix
    % with a banded structure.
    %
    % n: Size of the matrix (nxn)
    % bandwidth: Max distance from diagonal with nonzero elements
    % numSpikes: Number of large eigenvalues (spikes)
    % spikeMagnitude: Magnitude of spike eigenvalues
    % Sigma: Output spiked banded Hermitian covariance matrix

    % Step 1: Create a random Hermitian banded matrix
B=zeros(p,p);
for i=1:p
    for j=1:p
        if abs(i-j)<bandwidth
        B(i,j)=(0.5)^(abs(i-j)/bandwidth)*exp(-1i*pi*(i-j)/(2*bandwidth));
        end
    end
end


    % Step 2: Compute Eigen decomposition
    [V, D] = eig(B);
    eigenvalues = diag(D);

    % Step 3: Introduce spikes in eigenvalues
    %sortedIndices = randperm(p); % Randomize which eigenvalues to spike
    eigenvalues(1:p-numSpikes)=ones(p-numSpikes,1);
    eigenvalues(p-numSpikes+1:p)=logspace(log10(1), log10(1e1), numSpikes);

    % Step 4: Reconstruct the spiked covariance matrix
    D_spiked = diag(eigenvalues);
    Sigma = V * D_spiked * V';

    % Ensure symmetry numerically
    Sigma = (Sigma + Sigma') / 2;

    fprintf('Generated a %dx%d spiked banded Hermitian covariance matrix.\n', p, p);
end
