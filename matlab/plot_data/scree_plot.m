function h = scree_plot(lambdas)

h.figure = figure('Color','White');
h.axes = axes(); 
h.plot = plot(lambdas ./ sum(lambdas),'o--','Color','k');
xlabel('Component Number');
ylabel('Scaled Eigenvalues');
set(h.axes,'box','off','FontName','DroidSans','FontSize',14)

