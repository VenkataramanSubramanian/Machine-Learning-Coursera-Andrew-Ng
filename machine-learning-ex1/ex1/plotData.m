function plotData(x, y)
%   PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

figure(1); plot(x,y,'rx','MarkerSize',10);
xlabel('population')
ylabel('profit')

end
