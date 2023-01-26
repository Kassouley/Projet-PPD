#include <stdio.h>
#include <stdlib.h>
#include "PRR.h"
#include <string.h>
#include <time.h>
#include "mmio.h"
#include <omp.h>

int main(int argc, char* argv[])
{	
	int 	i, j;
	double* A = NULL;                  // the values of the matrix
    int*    I_A = NULL;         // the row indices of the non-zero elements of A
    int*    J_A = NULL;         // the column index of the non-zero elements of A
    int     n1, n2;             // size of the matrix A (n1 and n2 should be egal)
    int     nz;                 // number of non-zero elements in the matrix
	int 	m ;
	double 	precision = 0.1;
	double*	A_eigen_vector = NULL;
	FILE * output = NULL;
	omp_set_num_threads(atoi(argv[5]));

	for (i = 1; i < argc; i++)
	{
		if (!memcmp(argv[i],"--help",7) || !memcmp(argv[i],"-h",3))
		{
			printf("Pade-Rayleigh-Ritz Algorithm :\n\n");
			printf("Programme calculant les vecteurs propres approchés d'une grand matrice carré creuse.\n\n");
			printf("Utilisation :\n\t./PRR [matrix_file.mtx] [output_file] [m] {epsilon}\n\n");
			printf("[matrix_file.mtx]\tFichier .mtx vers la matrice carré creuse\n");
			printf("[output_file]\t\tFichier dans le quel stocker les vecteurs propres approchés\n");
			printf("[m]\t\t\tTaille du sous-espace généré\n");
			printf("{epsilon}\t\tPrecision des vecteurs approchés. (Default : 0.1)\n");
			printf("\n");
			exit(EXIT_SUCCESS);
		} 		
	}
    if (argc < 4)
	{
		printf("Pade-Rayleigh-Ritz Algorithm :\n\tErreur : la commande nécessite au moins 3 arguments\n\tUtilisez l'option -h pour plus d'information.\n");
		exit(EXIT_FAILURE);
	}
	if ( argc >= 5 )
	{
		precision = atof(argv[4]);
	}

	m = atoi(argv[3]);
	
	if ( m < 2)
	{
		printf("Pade-Rayleigh-Ritz Algorithm :\n\tErreur : le 3e argument [m] doit être superieur à 1\n\tUtilisez l'option -h pour plus d'information.\n");
		exit(EXIT_FAILURE);
	}

	mm_read_unsymmetric_sparse(argv[1], &n1, &n2, &nz, &A, &I_A, &J_A);

    if ( n1 != n2 )
    {
        printf("Pade-Rayleigh-Ritz Algorithm :\n\tErreur : La matrice d'entrée doit être carré\n");
        exit(EXIT_FAILURE);
    }
	if ( m >= n1 )
	{
        printf("Pade-Rayleigh-Ritz Algorithm :\n\tErreur : La valeur de m ne peut pas être supérieure ou égale à la taille de la matrice en entrée\n");
        exit(EXIT_FAILURE);
    }
	
	
    clock_t start, end;
    double cpu_time_used;

    start = clock(); // démarre le chronomètre


	A_eigen_vector = PRR(A, I_A, J_A, nz, n1, m, precision);

    end = clock(); // arrête le chronomètre
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC; // calcule le temps d'exécution en secondes
    printf("Temps d'execution : %f secondes\n", cpu_time_used);




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


	
    fclose(output);
	free(A);
	free(I_A);
	free(J_A);

	return 0;
}
