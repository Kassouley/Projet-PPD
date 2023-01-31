#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "PRR.h"
#include "mmio.h"
#include "matrix_vector.h"

int debug_point = -1;

int main(int argc, char* argv[])
{	
	int 	i, j;
	double* A = NULL;           // the values of the matrix
    int*    I_A = NULL;         // the row indices of the non-zero elements of A
    int*    J_A = NULL;         // the column index of the non-zero elements of A
    int     n1, n2;             // size of the matrix A (n1 and n2 should be egal)
    int     nz;                 // number of non-zero elements in the matrix
	int 	m ;
	int		nb_threads = 1;
	double 	precision = 0.1;
	double*	A_eigen_vector = NULL;
	double t0; 
	double t0_cpu; 
	double t1; 
	double t1_cpu; 
	double temps_reel;
	double temps_CPU;
	FILE * output = NULL;

	/* Checking if the user has entered the command line argument `--help` or `-h` and if so, it prints
	out the help message and exits the program. */
	for (i = 1; i < argc; i++)
	{
		if (!memcmp(argv[i],"--help",7) || !memcmp(argv[i],"-h",3))
		{
			printf("Pade-Rayleigh-Ritz Algorithm :\n\n");
			printf("Programme calculant les vecteurs propres approchés d'une grand matrice carré creuse.\n\n");
			printf("Utilisation :\n\t./PRR [matrix_file.mtx] [output_file] [m] {epsilon} {nbThread} {debugPoint}\n\n");
			printf("[matrix_file.mtx]\tFichier .mtx vers la matrice carré creuse\n");
			printf("[output_file]\t\tFichier dans le quel stocker les vecteurs propres approchés\n");
			printf("[m]\t\t\tTaille du sous-espace généré\n");
			printf("{epsilon}\t\tPrecision des vecteurs approchés (Default : 0.1)\n");
			printf("{nbThread}\t\tNombre de threads (Default : 1)\n");
			printf("{debugPoint}\t\tPermet d'afficher les matrices de la n ème itération de l'algorithme (Default : -1, n'affiche pas)\n");
			printf("\n");
			exit(EXIT_SUCCESS);
		} 		
	}

    /* Checking the number of arguments passed to the program. */
	if (argc < 4)
	{
		printf("Pade-Rayleigh-Ritz Algorithm :\n\tErreur : la commande nécessite au moins 3 arguments\n\tUtilisez l'option -h pour plus d'information.\n");
		exit(EXIT_FAILURE);
	}
	if ( argc >= 5 )
	{
		precision = atof(argv[4]);
	}
	if ( argc >= 6 )
	{
		nb_threads = atoi(argv[5]);
		if (nb_threads < 1)
		{
			printf("Pade-Rayleigh-Ritz Algorithm :\n\tErreur : le nombre de thread doit être supérieur à 0\n\tUtilisez l'option -h pour plus d'information.\n");
			exit(EXIT_FAILURE);
		}
	}
	if ( argc >= 7 )
	{
		debug_point = atoi(argv[6]);
		if (debug_point < 0)
		{
			printf("Pade-Rayleigh-Ritz Algorithm :\n\tErreur : le debug point doit être supérieur ou égal à 0\n\tUtilisez l'option -h pour plus d'information.\n");
			exit(EXIT_FAILURE);
		}
	}

	omp_set_num_threads(nb_threads);
	m = atoi(argv[3]);
	
	/* Checking if the user has entered a value for m that is less than 2. If so, it prints out an error
	message and exits the program. */
	if ( m < 2 )
	{
		printf("Pade-Rayleigh-Ritz Algorithm :\n\tErreur : le 3e argument [m] doit être superieur à 1\n\tUtilisez l'option -h pour plus d'information.\n");
		exit(EXIT_FAILURE);
	}

	mm_read_unsymmetric_sparse(argv[1], &n1, &n2, &nz, &A, &I_A, &J_A);

    /* Checking if the matrix is square. */
	if ( n1 != n2 )
    {
        printf("Pade-Rayleigh-Ritz Algorithm :\n\tErreur : La matrice d'entrée doit être carré\n");
        exit(EXIT_FAILURE);
    }

	/* Checking if the value of m is greater than or equal to the size of the input matrix. If so, it
	prints out an error message and exits the program. */
	if ( m >= n1 )
	{
        printf("Pade-Rayleigh-Ritz Algorithm :\n\tErreur : La valeur de m ne peut pas être supérieure ou égale à la taille de la matrice en entrée\n");
        exit(EXIT_FAILURE);
    }
	
	/* Getting the time at the start of the program. */
	t0 = omp_get_wtime(); 
	t0_cpu = clock(); 

	/* Calling the PRR function and storing the result in the A_eigen_vector variable. */
	A_eigen_vector = PRR(A, I_A, J_A, nz, n1, m, precision);

	/* Calculating the time taken by the program to run. */
	t1 = omp_get_wtime(); 
	t1_cpu = clock(); 
	temps_reel= t1 - t0;
	temps_CPU=(t1_cpu-t0_cpu)/CLOCKS_PER_SEC;
    printf("Temps d'execution : %f secondes / %f clocks\n", temps_reel,temps_CPU);

    /* Writing the eigen vectors to a file. */
	printf("\nEcriture des vecteurs propres approchés de A dans \"%s\"... ",argv[2]);
    output = fopen(argv[2], "w+");
    for(i = 0; i < n1; i++)
    {
        for(j = 0; j < m; j++)
        {
            fprintf(output, "%15f", A_eigen_vector[i*m+j]);
        }
		fprintf(output, "%s", "\n");
    }
    printf("Terminé\n");

    /* Closing the file and freeing the memory. */
	fclose(output);
	free(A);
	free(I_A);
	free(J_A);

	return 0;
}
