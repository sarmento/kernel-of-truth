set terminal latex
set output 'line_example.tex'

set style line 1 lt 2 lc rgb "red" lw 3
set style line 2 linewidth 6 lc rgb 'blue' pt 5


set xrange[-1:7]
set label "$x_{1}$"  at first  7, first -1

set key off
unset border

unset xtics
unset ytics

set arrow from -1, 0, 3 to  7.2, 0 linewidth 4
set arrow from 0, -2 to 0, 14.5 linewidth 4

set arrow from 1.75, 5 to 6.25, 5 nohead lt 2 lw 1
set arrow from 6.0, 4.5 to 6, 13.5 nohead lt 2 lw 1

set label "($x_{11}, y_1$)"  at first  0.5, first 6.5
set label "($x_{21}, y_2$)"  at first  4.5, first 14.5
set label "($x_{u1},  ?$)"  at first  2.5, first 10.5
set label "${\\alpha}_{1}$"  at first 3, first 5.75
set label "$y_0$"  at first -0.3, first 1.4



plot 1 + 2*x,  '-' w p ls 2, '-' w p ls 2, '-' w p ls 2
2 5
e
6 13
e
4 9
