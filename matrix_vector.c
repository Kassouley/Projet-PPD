#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <omp.h>
#include "matrix_vector.h"

/**
 * Return the linear combinaison of qi
 * 
 * @param x the output vector
 * @param qi the array of the qi values
 * @param N size of q(i)
 * @param m number of vector q(i)
 */
void combinaison_lineaire(double** x, double* qi, int N, int m)
{
	int i, j;
    //#pragma omp parallel for schedule(dynamic,N/2) private(i,j)
	for (i = 0; i < N; i++)
	{
		(*x)[i] = 0;
		for (j = 0; j < m; j++)
		{
			(*x)[i] += /*(double)(1/(double)m) */ qi[i*m+j];
		}
	}	
}


/**
 * It normalizes a vector
 * 
 * @param out the output vector
 * @param x the vector to normalize
 * @param taille_x the size of the vector x
 */
void normalisation(double** out, double* x, int taille_x)
{
	int     i;
	double  n = norme(x, taille_x);

	#pragma omp parallel for schedule(dynamic) private(i)	
	for (i = 0; i < taille_x; i++)
	{
		(*out)[i] = x[i] / n;
	}
}

/**
 * It computes the norm of a vector
 * 
 * @param v         the vector to compute the norm of
 * @param taille_v  the size of the vector
 * 
 * @return The norm of the vector v.
 */
double norme(double* v, int taille_v)
{
	int 	i;
	double 	res = 0;
	
	//#pragma omp parallel for schedule(dynamic) private(i) reduction(+:res) 
	for (i = 0; i < taille_v; i++)
	{
		res += v[i]*v[i];
	}
	return sqrt(res);
}

/**
 * It computes the scalar product of two vectors
 * 
 * @param x         the first vector
 * @param y         the vector to be multiplied
 * @param taille    the size of the vectors
 * 
 * @return The dot product of the two vectors.
 */
double produit_scalaire(double* x, double* y, int taille)
{
	int     i;
	double  res = 0;
	
	//#pragma omp parallel for schedule(dynamic) reduction(+:res) private(i)
	for (i = 0; i < taille; i++)
	{
		res += x[i] * y[i];
	}
	return res;
}


/**
 * It computes the product of a sparse matrix and a vector
 * 
 * @param A         the values of the matrix
 * @param I_A       the row indices of the non-zero elements of A
 * @param J_A       the column index of the non-zero elements of A
 * @param v         the vector to multiply
 * @param lignes_A  number of rows in the matrix A
 * @param nz        number of non-zero elements in the matrix
 * @param taille_v  the size of the vector v
 * 
 * @return The result of the product of the matrix A and the vector v.
 */
double* produit_matrice_vecteur(double* A, int* I_A, int* J_A, double* v, int lignes_A, int nz, int taille_v)
{
	int     i;
	double* res = (double*)calloc(lignes_A, sizeof(double));
	
    //#pragma omp parallel for schedule(dynamic) reduction(+:res[:lignes_A]) private(i)
    for (i = 0; i < nz; i++) 
	{
        res[I_A[i]] += A[i] * v[J_A[i]];
    }
	return res;
}


/**
 * It computes the product of two matrices
 * 
 * @param out the output matrix
 * @param A The first matrix
 * @param B the matrix to be multiplied
 * @param lignes_A number of rows in matrix A
 * @param colonnes_B number of columns of the second matrix
 */
void produit_matrice_matrice(double** out, double* A, double* B, int lignes_A, int colonnes_B)
{
	int i, j, k;
	double tmp;
	//#pragma omp parallel for schedule(dynamic) private(i, j, k, tmp)
	for (i = 0; i < lignes_A; i++)
	{
		for (j = 0; j < colonnes_B; j++)
		{
			tmp = 0;
            //#pragma omp parallel for schedule(dynamic) reduction(+:tmp) private(k)
			for (k = 0; k < colonnes_B; k++)
			{
				tmp += A[i*colonnes_B+k] * B[k*colonnes_B+j];
			}
			(*out)[i*colonnes_B+j] = tmp;
		}
	}
}


/**
 * It takes a matrix A and returns the inverse
 * 
 * @param out the output matrix
 * @param A The matrix to invert
 * @param n the size of the matrix
 */
void inverse(double** out, double * A, int n)
{
    int                 s;
	gsl_matrix_view     mat_A       = gsl_matrix_view_array(A, n, n);
    gsl_matrix*         Ainv        = gsl_matrix_alloc(n, n);
    gsl_permutation*    p           = gsl_permutation_alloc(n);

	gsl_linalg_LU_decomp(&mat_A.matrix, p, &s);
    gsl_linalg_LU_invert(&mat_A.matrix, p, Ainv);
    gsl_permutation_free(p);
	*out = gsl_matrix_ptr(Ainv, 0, 0);
}

/**
 * It computes the eigenvalues and eigenvectors of a real symmetric matrix
 * 
 * @param A         the matrix
 * @param n         the size of the matrix
 * @param lambda    the eigenvalues of A
 * @param ui        the eigenvectors of A
 * 
 * @return The eigenvalues and eigenvectors of the matrix A.
 */
void calculer_elements_propres(double* A, int n, double** lambda, double** ui)
{
	int                             i, j;
    gsl_matrix_view                 mat_A = gsl_matrix_view_array(A, n, n);
    gsl_vector_complex*             eval  = gsl_vector_complex_alloc (n);
    gsl_matrix_complex*             evec  = gsl_matrix_complex_alloc (n, n);
    gsl_eigen_nonsymmv_workspace*   w     = gsl_eigen_nonsymmv_alloc(n);

    gsl_eigen_nonsymmv(&mat_A.matrix, eval, evec, w);
    gsl_eigen_nonsymmv_free(w);
    gsl_eigen_nonsymmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);

    for (i = 0; i < n; i++)
	{
		(*lambda)[i] = GSL_REAL(gsl_vector_complex_get(eval,i));
		for (j = 0; j < n; j++)
		{
			(*ui)[i*n+j] = GSL_REAL(gsl_matrix_complex_get(evec, i, j));
		}
	}
	
    gsl_vector_complex_free(eval);
    gsl_matrix_complex_free(evec);
}

/**
 * It prints a matrix
 * 
 * @param name  the name of the matrix
 * @param Mat   the matrix to print
 * @param n     number of rows
 * @param m     number of columns
 */
void print_matrice(char* name, double* A, int n, int m)
{
	int i, j;

	printf("%s : \n",name);
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			printf("%15f ", A[i*m+j]);
		}
		printf("\n");
	}
	printf("___________________________\n\n");
}