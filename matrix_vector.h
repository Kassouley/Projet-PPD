/**
 * Basic functions of matrix and vector operation
 * 
 * @authors Lucas NETO, Loïk SIMON
 * 
*/

#ifndef MATRIX_VECTOR_H
#define MATRIX_VECTOR_H

/* Vector manipulation functions */
void     combinaison_lineaire(double** x, double* qi, int N, int m);
void     normalisation(double** out, double* x, int taille_x);
double   norme(double* v, int taille);
double   produit_scalaire(double* x, double* y, int taille);

/* Matrix manipulation functions */
double* produit_matrice_vecteur(double* A, int* I_A, int* J_A, double* v, int lignes_A, int nz, int taille_v);
void    produit_matrice_matrice(double** out, double* A, double* B, int lignes_A, int colonnes_B);
void    inverse(double** out, double * A, int n);
void    calculer_elements_propres(double* A, int n, double** valeur_propre_A, double** vecteur_propre_A);
void    print_matrice(char* name, double* A, int n, int m);

#endif
