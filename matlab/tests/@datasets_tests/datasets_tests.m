classdef datasets_tests < matlab.unittest.TestCase
% Tests all data loaders. 

    methods(Test)
        function test_conte69(testCase)
            testCase.assumeTrue(exist('gifti', 'file'), 'Gifti library not found.');

            [surf_lh, surf_rh] = load_conte69();
            [sphere_lh, sphere_rh] = load_conte69('sPhErEs');
          
            for surface = {surf_lh, surf_rh, sphere_lh, sphere_rh}
                testCase.verifyEqual(fieldnames(surface{1}), {'tri'; 'coord'});
                testCase.verifySize(surface{1}.tri, [64980, 3]);
                testCase.verifySize(surface{1}.coord, [3, 32492]);
            end
        end

        function test_gradient(testCase)
            for name = {'fc', 'mpc'}
                for n_gradient = 1:2
                    gradient = load_gradient(name{1}, n_gradient);
                    testCase.verifySize(gradient, [64984, 1]);
                end
            end
        end

        function test_group_fc(testCase)
            fc = { ...
                load_group_fc({'vosdewael', 'schaefer'}, 100:100:400, 'main'), ...
                load_group_fc({'vosdewael', 'schaefer'}, 100:100:400, 'holdout')
                };
            for fc_group = fc
                fields = fieldnames(fc_group{1});
                for parcellation = fields'
                    fc_matrix = fc_group{1}.(parcellation{1});
                    n_regions = str2double(parcellation{1}(end-2:end));
                    testCase.verifySize(fc_matrix, [n_regions, n_regions])
                end
            end
           
        end

        function test_group_mpc(testCase)
            mpc = load_group_mpc('vosdewael', 200);
            testCase.verifySize(mpc.vosdewael_200, [200, 200]);
        end

        function test_marker(testCase)
            for name = {'curvature', 'thickness', 't1wt2w'}
                [metric_lh, metric_rh] = load_marker(name{1});
                testCase.verifySize(metric_lh, [32492, 1]);
                testCase.verifySize(metric_rh, [32492, 1]);
            end
        end

        function test_mask(testCase)
            for name = {'midline', 'temporal'}
                [mask_lh, mask_rh] = load_mask(name{1});
                testCase.verifySize(mask_lh, [32492, 1]);
                testCase.verifySize(mask_rh, [32492, 1]);
            end

        end

        function test_parcellation(testCase)
            parcellations = load_parcellation( ...
                {'vosdewael', 'schaefer'}, ...
                100:100:400 ...
                );
            
            names = fieldnames(parcellations);
            expected_names = ["vosdewael_"; "schaefer_"] + (100:100:400);
            testCase.verifyEqual( ...
                sort(names), ...
                cellstr(sort(expected_names(:))) ...
                );

            structfun(@(x) testCase.verifySize(x, [64984, 1]), parcellations);
        end

    
    end
end