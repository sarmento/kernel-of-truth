set terminal latex
set output 'loss_functions.tex'

set xrange[-1:2]
set yrange[-0.25:3]

set arrow from -1, 0 to  1.5, 0 linewidth 4
set arrow from -0.75, -2 to -0.75, 3 linewidth 4

set label "$\\hat{y}_{i}$" at first 1.4, first -0.25
set label "$\\hat{y}_{i} = y_{i}$" at first  -0.15, first -0.25
set label "$\\mathcal{L}_{L2}$" at first 0.6, first 2.8
set label "$\\mathcal{L}_{L1}$" at first 1.15, first 2.8

set key off
unset border

unset xtics
unset ytics

plot (2*x)*(2*x), abs(2*x)
