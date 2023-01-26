
PRR : main.c PRR.c matrix_vector.c mmio.c
	gcc main.c PRR.c matrix_vector.c mmio.c -fopenmp -lgsl -lm -o PRR 
