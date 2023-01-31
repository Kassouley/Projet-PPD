
# Projet de Programmation Parallèle et Distribuée

Implémentation en C de l'algorithme de calcul des éléments propres d'une matrice symétrique en utilisant la méthode de Padé-Rayleigh-Ritz


## Authors

- [@Lucas NETO](https://www.github.com/Kassouley)
- [@Loïk SIMON](https://www.github.com/KoliNomis)

## Requierment

Ce programme utilise la librairie GSL 2.7, il est donc nécessaire de l'installer préalablement :

```bash
sudo apt-get install libgsl-dev
```

## Installation

Installer le projet avec le makefile

```bash
  make
```
    
## Documentation

Utilisation :

```bash
./PRR [matrix_file.mtx] [output_file] [m] {epsilon} {nbThread} {debugPoint}
```

[matrix_file.mtx]       Fichier .mtx vers la matrice carré creuse

[output_file]           Fichier dans le quel stocker les vecteurs propres approchés

[m]                     Taille du sous-espace généré

{epsilon}               Precision des vecteurs approchés (Default : 0.1)

{nbThread}              Nombre de threads (Default : 1)

{debugPoint}            Permet d'afficher les matrices de la n ème itération de l'algorithme (Default : -1, n'affiche pas)

Attention, l'utilsation d'un paramètre optionnel nécessite les autres paramètres le précédant.

## Example

```bash
./PRR -h
```

```bash
./PRR "matrix/matrix.mtx" "eigenVector.txt" 10 
```

```bash
./PRR "matrix/matrix.mtx" "matrix/output.txt" 5 0.001 
```

```bash
./PRR "matrix/matrix400.mtx" "output.txt" 10 0.001 8
```