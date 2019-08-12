function h = scree_plot(lambdas)

h.figure = figure('Color','White');
h.axes = axes(); 
h.plot = plot(lambdas ./ sum(lambdas),'o-');
xlabel('Component Number');
ylabel('Eigenvalues');
set(h.axes,'box','off','FontName','DroidSans','FontSize',14)
