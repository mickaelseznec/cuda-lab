# TP Partie 0 : faire fonctionner votre GPU
## Quel est votre matériel ?

Pour découvrir quel matériel est installé sur votre machine, nous allons utiliser le programme *deviceQuery*. Il est fourni par Nvidia dans chaque distribution de CUDA. Voici les différentes étapes pour le compiler et l'utiliser.

```bash
$ cd 0-deviceQuery
$ mkdir build && cd build
$ cmake ..
$ make
$ ./deviceQuery
```

Servez-vous du résultat de *deviceQuery* pour répondre à ces quelques questions:
1. Combien y-a-t-il d'unités de calcul (ALU) sur votre GPU ?
2. On considère qu'une ALU peut fournir le résultat de 3 opérations flotantes par coup d'horloge. Combien d'opérations par secondes en FLOPS (Floation-Poing Operations per Second) peut effectuer votre GPU ?
3. Quelle est la bande passante théorique du GPU ?
4. Comparez la fréquence de fonctionnement du GPU avec celle du CPU (vous pouvez utiliser *lscpu* par exemple). Pourquoi est-il quand même intéressant d'utiliser un GPU ?
