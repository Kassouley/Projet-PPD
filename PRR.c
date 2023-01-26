#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "matrix_vector.h"
#include "PRR.h"

/**
 * It computes the eigenvectors of a matrix A using the PRR algorithm
 * 
 * @param A the sparse matrix
 * @param I_A the row indices of the non-zero elements of A
 * @param J_A column index of the non-zero elements of the matrix A
 * @param nz number of non-zero elements in the matrix
 * @param n number of rows/columns of A
 * @param m size of the projection
 * @param precision the precision of the algorithm
 * 
 * @return the eigen vectors of the matrix A.
 */
double* PRR(double* A, int* I_A, int* J_A, int nz, int n, unsigned int m, double precision)
{
    int     i;                  // iterable variable
    int     est_precis;         // boolean that stop the while loop
	double* x       = NULL;     // x vector (nx1)
	double* y0      = NULL;     // y0 vector (nx1)
	double* Bm      = NULL;     // Bm matrix (mxm)
	double* Bm_1    = NULL;     // Bm-1 matrix (mxm)
	double* Vm      = NULL;     // Vm matrix (nxm)
	double* Em      = NULL;     // Inverse of Bm-1 (mxm)
	double* Fm      = NULL;     // Fm matrix (mxm)
	double* lambda  = NULL;     // eigen value of Fm 
 	double* ui      = NULL;     // eigen vector of Fm
	double* qi      = NULL;     // approximation of the eigen vectors of the matrix A
	int compteur 	= 0;

   
    x       = (double*) calloc(n, sizeof(double));
    y0      = (double*) malloc(n * sizeof(double));
    Bm      = (double*) calloc(m*m, sizeof(double));
    Bm_1    = (double*) calloc(m*m, sizeof(double));
    Vm      = (double*) calloc(m*n, sizeof(double));
    Em      = (double*) calloc(m*m, sizeof(double));
    Fm      = (double*) calloc(m*m, sizeof(double));
    lambda  = (double*) malloc(m * sizeof(double));
    ui      = (double*) malloc(m*m*sizeof(double));
    qi      = (double*) calloc(m*n, sizeof(double));
    
	
    /** Initialisation of x */
	#pragma omp parallel for private(i)
	for (i = 0; i < n; i++)
	{
		x[i]=0;
	}
	x[0]=1;

	do
	{
		// print_matrice("x", x, N, 1);
		normalisation(&y0, x, n);
		// print_matrice(y0, N, 1);
		projection(A, I_A, J_A, nz, n, m, y0, &Bm, &Bm_1, &Vm);
		// print_matrice("Bm", Bm, m, m);
		// print_matrice("Bm-1", Bm_1, m, m);
		// print_matrice("Vm", Vm, N, m);

		inverse(&Em, Bm_1, m);
		// print_matrice("Em", Em,m,m);

		produit_matrice_matrice(&Fm, Em, Bm, m, m);
		// print_matrice("Fm", Fm,m,m);

		calculer_vecteurs_propres(Fm,  m, &lambda, &ui);
		// print_matrice("Eigein vector", vecteur_propre_A,m,m);
		// print_matrice("Eigein value", lambda,m,1);
		
        produit_matrice_matrice(&qi, Vm, ui, n, m);
		// print_matrice("qi", qi,N,m);
		
		compteur++;
		printf("compteur : %d \t", compteur);
		est_precis = est_precision_suffisante(precision, A, I_A, J_A, nz, qi, lambda, n, m);
		if (!est_precis)
		{
			combinaison_lineaire(&x, qi, n, m);
			// print_matrice("x", x, N, 1);
		}
		// break;
	} while (!est_precis);

	free(x);
    free(y0);
    free(Bm);  
    free(Bm_1); 
    free(Vm);
    free(Em);
	free(Fm);
    free(lambda);
	free(ui);
	return qi;
}


/**
 * It computes the matrix Bm, Bm-1 and Vm
 * 
 * @param A the matrix
 * @param I_A the row indices of the matrix A
 * @param J_A the column index of the non-zero elements of A
 * @param nz number of non-zero elements in the matrix A
 * @param n number of rows of the matrix
 * @param m size of the projection
 * @param y0 a initial vector y0
 * @param Bm the matrix Bm
 * @param Bm_1 the matrix Bm-1
 * @param Vm the matrix Vm
 */
void projection(double* A, int* I_A, int* J_A, int nz, int n, int  m, 
						double* y0, double** Bm, double** Bm_1, double** Vm)
{
	int 	i, k = 1;
	double 	scalaire1 = 0, scalaire2 = 0;
	double* yk 	 = NULL;
	double* yk_1 = NULL;

	yk_1 = (double*)malloc(sizeof(double) * n);
	
	#pragma omp parallel for private(i)
	for (i = 0; i < n; i++)
	{
		(*Vm)[i*m] = y0[i];
	}
	(*Bm_1)[0] = produit_scalaire(y0,y0,n);
	
	#pragma omp parallel for private(i)
	for (i = 0; i < n; i++)
	{
		yk_1[i] = y0[i];
	}
	
	for (k = 1; k < m; k++)
	{
		yk = produit_matrice_vecteur(A, I_A, J_A, yk_1, n, nz, n);
		
		
		#pragma omp parallel for private(i)
		for (i = 0; i < n; i++)
		{
			(*Vm)[i*m+k] = yk[i];
		}
		
		scalaire1 = produit_scalaire(yk, yk_1, n);
		scalaire2 = produit_scalaire(yk, yk, n);
		
		
		#pragma omp parallel for schedule(dynamic) private(i)
		for (i = 0; i <= (2*k); i++)
		{
			if (2*k-i < m)
			{
				if (i-2 >= 0 && i-2 < m)
				{
					(*Bm)[(2*k-i)*m+(i-2)] = scalaire1;
				}
				if (i-1 >= 0 && i-1 < m)
				{
					(*Bm_1)[(2*k-i)*m+(i-1)] = scalaire1;
					(*Bm)[(2*k-i)*m+(i-1)] = scalaire2;
				}
				if (i < m)
				{
					(*Bm_1)[(2*k-i)*m+(i)] = scalaire2;
				}
			}
		}
		
		#pragma omp parallel for private(i)
		for (i = 0; i < n; i++)
		{
			yk_1[i] = yk[i];
		}
		free(yk);
		
	}
	yk = produit_matrice_vecteur(A, I_A, J_A, yk_1, n, nz, n);
	(*Bm)[(m-1)*m+(m-1)] = produit_scalaire(yk, yk_1, n);
	free(yk_1);
	free(yk);
}


/**
 * Boolean function that verify if the precision is accurate
 * 
 * @param epsilon the precision we want to reach
 * @param A the matrix
 * @param I_A the row indices of the non-zero elements of A
 * @param J_A the column index of the non-zero elements of A
 * @param nz number of non-zero elements in the matrix
 * @param qi the eigenvectors
 * @param lambda the eigenvalues
 * @param N the size of the matrix
 * @param m size of the projection
 * 
 * @return a boolean value.
 */
int est_precision_suffisante(float epsilon, double* A, int* I_A, int* J_A, int nz, 
												double* qi, double* lambda, int N, int m)
{	
	int 		i, j;
	double*	 	A_x_q 			= NULL;
	double*	 	q 		   		= (double*)malloc(sizeof(double) * N);
	double*	 	lambda_x_q 		= (double*)malloc(sizeof(double) * N);
	double*	 	difference 		= (double*)malloc(sizeof(double) * N);
	double		precision 		= 0;
	double 		precision_max 	= 0;

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < N; j++)
		{
			q[j] = qi[j*m+i];
		}
		A_x_q = produit_matrice_vecteur(A, I_A, J_A, q, N, nz, N);
		// print_matrice("A_x_q",A_x_q,N,1);

		for (j = 0; j < N; j++)
		{
			lambda_x_q[j] = lambda[i] * q[j];
		}
		// print_matrice("lambda_x_q",lambda_x_q,N,1);
		
		for (j = 0; j < N; j++)
		{
			difference[j] = A_x_q[j] -  lambda_x_q[j];
		}
		// print_matrice("difference",difference,N,1);

		precision = norme(difference, N);
		// printf("p : %f\n", precision);

		
		if (precision > precision_max)
		{
			precision_max = precision;
		}
		// break;
		free(A_x_q);
	}
	printf("pmax : %f\n", precision_max);
	free(lambda_x_q);
	free(difference);
	return (precision_max < epsilon);

}