/**
 * Pade-Rayleigh-Ritz Algorithm librairie
 * 
 * @authors Lucas NETO, Loïk SIMON
 * 
*/

#ifndef PRR_H
#define PRR_H

double* PRR(double* A, int* I_A, int* J_A, int nz, int n, unsigned int m, double precision);
void 	projection(double* A, int* I_A, int* J_A, int nz, int n, int  m, 
					double* y0, double** Bm, double** Bm_1, double** Vm);
int 	est_precision_suffisante(float epsilon, double* A, int* I_A, int* J_A, int nz, 
									double* qi, double* lambda, int N, int m);

#endif