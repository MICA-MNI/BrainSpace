classdef utils_tests < matlab.unittest.TestCase

    methods(Test)
        function test_graph_is_connected(testCase)
            % Tests the graph_is_connected function against MathWorks'
            % conncomp.
            
            one_disconnected = ones(5);
            one_disconnected(:,5) = 0;
            one_disconnected(5,:) = 0;

            test_graphs = {
                magic(5) + magic(5)', ... % connected
                eye(5), ... % Not connected
                one_disconnected
            };

            for G = test_graphs
                testCase.assertEqual( ...
                    all(conncomp(graph(G{1}))==1), ...
                    graph_is_connected(G{1}) ...
                );
            end
        end
    end
end