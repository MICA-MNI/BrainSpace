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
                one_disconnected, ...
                graph(one_disconnected)
            };

            for G = test_graphs
                if isa(G{1}, 'graph')
                    mathworks_answer = all(conncomp(G{1})==1);
                else
                    mathworks_answer = all(conncomp(graph(G{1}))==1);
                end
                brainspace_answer = graph_is_connected(G{1});

                testCase.assertEqual( ...
                    mathworks_answer, ...
                    brainspace_answer ...
                );
            end
        end
    end
end