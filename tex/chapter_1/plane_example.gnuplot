## This will allow underscripts
##set termopt enhanced

set terminal latex
set output 'plane_example.tex'

set view 60, 115

set xrange [-2 :  4]
set yrange [ 0 :  6]
set zrange [-4 : 16]

## axis
set arrow from -2, 0, 0 to 6, 0, 0
set arrow from  0, 0,-2 to 0, 0, 17
set arrow from  0, 0, 0 to 0, 7, 0

## the tangents
set arrow from 0, -1, -1 to 0, 7, 15 nohead linetype 4 lw 5
set arrow from -4, 0, 3 to 6, 0, -2 nohead linetype 4 lw 5

#set grid y
#set grid x
#set grid z

set zzeroaxis
set ztics axis

## This is totally weird but needed for aligning the xy plane with the axis / arrows
set xyplane -0.2

## Legends and etc
set border 0
set key off

## Arrows to points (x_2, x_1, y)
set arrow from -2, 2, 10 to -2, 2,  6 linetype 3 lw 3 
set arrow from  2, 6.5, 11 to  2, 6, 12 linetype 3 lw 3
set arrow from  5, 2, -1.5 to  4, 1, 1 linetype 3 lw 3
## The intercept
set arrow from 0, -1, 4 to 0, 0,  1 linetype 3 lw 3


set label "($x_1 = 2, x_2 = -2, y = 6$)"  at first  -2, first 0.5 , first 11
set label "($x_1 = 6, x_2 = 2, y = 12$)"  at first  2, first 6.55, first 10.5
set label "($x_1 = 1, x_2 = 4, y = 1$)"  at first  5, first 0.5 , first -4
## The intercept
set label "$y_{00}$"  at first  -2, first -2.5, first 0

## Labels on the axis
set label "$x_1$"  at first  0, first   7, first -1
set label "$x_2$"  at first  7, first   0 , first 2
set label "y"      at first  0.25, first 0, first 18.5

set isosample 7
splot  1 + 2 * y - 0.5 * x

