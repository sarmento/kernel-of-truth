set terminal latex
set output 'line_example_many_points.tex'

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

## The error
set arrow from 4, 9 to 4, 10.7 lt 2 lw 1
set arrow from 4, 10.7 to 4, 9 lt 2 lw 1
set label "$r_i$"  at first 3.75, first 9.75

## The true y point
set label "$(x_i, y_i)$"  at first 3.7, first 12

## The point in the plane
set label "$(x_i, \\hat{y}_i)$"  at first 3.3, first 6
set arrow from 3.7, 6.5 to 4, 9 lt 2 lw 1

##set label "($x_{u1},  ?$)"  at first  2.5, first 10.5
##set label "${\\alpha}_{1}$"  at first 3, first 5.75
##set label "$y_0$"  at first -0.3, first 1.4



plot 1 + 2*x,  '-' w p ls 2, '-' w p ls 2, '-' w p ls 2, '-' w p ls 2, '-' w p ls 2, '-' w p ls 2, '-' w p ls 2
0.5 1.5
e
2 5.5
e
3 6.0
e
4 11
e
4.5 9
e
6 12.5
e
6.5 13.5
