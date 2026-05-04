classdef diffusion_mapping_tests < matlab.unittest.TestCase
    % Regression tests for matlab/analysis_code/diffusion_mapping.m
    % Covers issues #98 (complex eigenvalues) and #94 (sqrt(N) component cap).

    methods(Test)
        function test_real_eigenvalues_on_cosine_similarity(testCase)
            % Issue #98: a positive symmetric (cosine-similarity-like)
            % input must yield real, descending-sorted eigenvalues.
            rng(0);
            X = randn(20, 50);
            Xn = X ./ vecnorm(X, 2, 2);
            S = (Xn * Xn.' + 1) / 2;        % positive cosine similarity in [0, 1]
            S = (S + S.') / 2;
            S(1:size(S,1)+1:end) = 1;        % unit diagonal

            [embedding, eigval] = diffusion_mapping(S, 3, 0.5, 0);

            testCase.verifyTrue(isreal(eigval), ...
                'Eigenvalues must be real.');
            testCase.verifyTrue(isreal(embedding), ...
                'Embedding must be real.');
            testCase.verifyEqual(eigval, sort(eigval, 'descend'), ...
                'AbsTol', 1e-12, ...
                'Eigenvalues must be sorted in descending order.');
        end

        function test_n_components_above_sqrt_n_honored(testCase)
            % Issue #94: previously the function silently capped output at
            % floor(sqrt(N)) components. Verify a request for more than
            % sqrt(N) components is now honored.
            rng(1);
            n = 25;                          % sqrt(25) = 5
            X = randn(n, 60);
            Xn = X ./ vecnorm(X, 2, 2);
            S = (Xn * Xn.' + 1) / 2;
            S = (S + S.') / 2;
            S(1:n+1:end) = 1;

            requested = 10;                  % > sqrt(n)
            embedding = diffusion_mapping(S, requested, 0.5, 0);

            testCase.verifyEqual(size(embedding, 2), requested, ...
                'diffusion_mapping must honor n_components > sqrt(N).');
        end

        function test_zero_row_sum_rejected(testCase)
            % An all-zero row should raise rather than silently produce Inf.
            S = eye(5);
            S(3, :) = 0;
            S(:, 3) = 0;

            testCase.verifyError( ...
                @() diffusion_mapping(S, 2, 0.5, 0), ...
                'diffusion_mapping:zeroRowSum');
        end
    end
end
