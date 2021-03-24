/*
 *  common-2018.h
 * 
 *  Created on: 13 gen. 2018
 *  Author: ecesar
*/

#ifndef NN_H_
#define NN_H_

/* Conjunt de constants necessàries per l'execució del programa. Si feu qualsevol extensió
 * caldrà fer els canvis corresponents (per exemple, si s'incrementa el nombre de patrons d'entrada
 * cal incrementar el valor de NUMPAT).
*/

#define NUMPAT 1934	//Patrons en el training set
#define NUMRPAT 946	//Patrons per reconéixer
#define NUMOUT 10	//Neurones de sortida
#define NUMHID 117	//Neurones en la capa oculta
#define NUMIN  1024	//Neurones d'entrada. 32*32
#define BSIZE  50   //Patrons per batch

/*Arrays on s'emmagatzemaran els pesos calculats durant la fase d'entrenament per a ser utilitzats posteriorment
 * en la fase de reconéixament (en un cas real, aquests valors haurien de ser emmagatzemats en un arxiu després
 * de la fase d'entrenament pel seu us posterior)
 * La matriu Target ens permet inicialitzar els patrons de sortida que coneixem (del training set)
*/
extern float Target[NUMPAT][NUMOUT];
extern int   Validation[NUMRPAT];
extern float WeightIH[NUMHID][NUMIN];
extern float WeightHO[NUMOUT][NUMHID];

/* Capçaleres de les funcions implementades a common.c */
char **loadPatternSet( int np, char *fname, int trainS );
char *readImg( FILE *fd );
void printImg( char *Img, int x );

#endif /* NN_H_ */
