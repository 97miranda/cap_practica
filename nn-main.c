/*
 * nn.c
 *
 *  Created on: 5 jul. 2016
 *  Author: ecesar
 *
 *      Descripció:
 *      Xarxa neuronal simple de tres capes. La d'entrada que són els pixels d'una
 *      imatge (mirar descripció del format al comentari de readImg) de 32x32 (un total de 1024
 *      entrades). La capa oculta amb un nombre variable de neurones (amb l'exemple proporcionat 117
 *      funciona relativament bé, però si incrementem el nombre de patrons d'entrament caldrà variar-lo).
 *      Finalment, la capa de sortida (que ara té 10 neurones ja que l'entrenem per reconéixer 10
 *      patrons ['0'..'9']).
 *      El programa passa per una fase d'entrenament en la qual processa un conjunt de patrons (en
 *      l'exemple proporcionat són 1934 amb els dígits '0'..'9', escrits a mà). Un cop ha calculat 
 * 	    els pesos entre la capa d'entrada i l'oculta i entre
 *      aquesta i la de sortida, passa a la fase de reconèixament, on llegeix 946 patrons d'entrada
 *      (es proporcionen exemples per aquests patrons), i intenta reconèixer de quin dígit es tracta.
 *
 *  Darrera modificació: gener 2019. Ara l'aprenentatge fa servir la tècnica dels mini-batches
 */

/*******************************************************************************
*    Aquest programa és una adaptació del fet per  JOHN BULLINARIA  
*    ( http://www.cs.bham.ac.uk/~jxb/NN/nn.html):
*
*    nn.c   1.0                                       � JOHN BULLINARIA  2004  *
*******************************************************************************/



#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <limits.h>
#include <omp.h>

#include "common.c"


int total;
int seed=50;

int rando()
{
    seed = (214013*seed+2531011);
    return seed>>16;
}

float frando()
{
    return (rando()/65536.0f);
}

void freeTSet( int np, char **tset ){
	int i;
	#pragma omp parallel for num_threads(12)
	for( i = 0; i < np; i++ ) free( tset[i] );
	free(tset);
}

void trainN(){
	char **tSet;

   	float DeltaWeightIH[NUMHID][NUMIN], DeltaWeightHO[NUMOUT][NUMHID];
	float Error, BError, eta = 0.3, alpha = 0.5, smallwt = 0.22;
	int ranpat[NUMPAT];
 	float Hidden[NUMHID], Output[NUMOUT], DeltaO[NUMOUT], DeltaH[NUMHID];
 	float SumO, SumH, SumDOW;

	if( (tSet = loadPatternSet( NUMPAT, "optdigits.tra", 1 ) ) == NULL){
       		printf( "Loading Patterns: Error!!\n" );
		exit( -1 );
	}

	int i,j,epoch,p,nb,np,k;

	//omp_set_nested(1);
        //#pragma omp parallel num_threads(12)
        //{
		//#pragma omp for private(i,j)
		for( i = 0; i < NUMHID; i++ )
			for( j = 0; j < NUMIN; j++ ){
				WeightIH[i][j] = 2.0 * ( frando() + 0.01 ) * smallwt;
				DeltaWeightIH[i][j] = 0.0;
			}

		//#pragma omp for private(i,j)
		for( i = 0; i < NUMOUT; i++)
			for( j = 0; j < NUMHID; j++ ){
				WeightHO[i][j] = 2.0 * ( frando() + 0.01 ) * smallwt;
				DeltaWeightHO[i][j] = 0.0;
			}
//	}
//	printf("Inicio for epoch\n");
	#pragma omp parallel num_threads(12)
	{
    		for( epoch = 0 ; epoch < 1000000 ; epoch++ ) {    // iterate weight updates
			#pragma omp for private(p)
        		for( p = 0 ; p < NUMPAT ; p++ )   // randomize order of individuals
        	    		ranpat[p] = p;
		 	//#pragma omp single
        		for( p = 0 ; p < NUMPAT ; p++) {
        	        	int x = rando();
        	        	int np = (x*x)%NUMPAT;
        	        	int op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
        		}
			//#pragma omp master
        		Error = BError = 0.0;
			//#pragma omp barrier

       			printf("."); fflush(stdout);

       			for ( nb = 0; nb < NUMPAT/BSIZE; nb++) { // repeat for all batches
       				BError = 0.0;
        	       		for( np = nb*BSIZE ; np < (nb + 1)*BSIZE ; np++ ){//repeat for all the training patterns within the batch
        	       			int p = ranpat[np];
					#pragma omp for private(i,j,SumH)
        	       			for( j = 0 ; j < NUMHID ; j++ ) {    // compute hidden unit activations
        	       				SumH = 0.0;	
        	       				for( i = 0 ; i < NUMIN ; i++ ) 
							SumH += tSet[p][i] * WeightIH[j][i];
        	       				Hidden[j] = 1.0/(1.0 + exp( -SumH )) ;
        	       			}
					#pragma omp for private(j,k,SumO) reduction(+:BError)
        	       			for( k = 0 ; k < NUMOUT ; k++ ) {    // compute output unit activations and errors
        	       				SumO = 0.0;	
        	       				for( j = 0 ; j < NUMHID ; j++ ) 
							SumO += Hidden[j] * WeightHO[k][j] ;
        	       				Output[k] = 1.0/(1.0 + exp(-SumO)) ;   // Sigmoidal Outputs
        	       				BError += 0.5 * (Target[p][k] - Output[k]) * (Target[p][k] - Output[k]) ;   // SSE
        	       				DeltaO[k] = (Target[p][k] - Output[k]) * Output[k] * (1.0 - Output[k]) ;   // Sigmoidal Outputs, SSE
        	       			}
					#pragma omp for private(i,j,k,SumDOW)
        	       			for( j = 0 ; j < NUMHID ; j++ ) {     // update delta weights DeltaWeightIH
        	       				SumDOW = 0.0 ;	
        	       		        	for( k = 0 ; k < NUMOUT ; k++ ) 
							SumDOW += WeightHO[k][j] * DeltaO[k] ;
        	       				DeltaH[j] = SumDOW * Hidden[j] * (1.0 - Hidden[j]) ;

	               				for( i = 0 ; i < NUMIN ; i++ )
        	      					DeltaWeightIH[j][i] = eta * tSet[p][i] * DeltaH[j] + alpha * DeltaWeightIH[j][i];
        	       			}
					#pragma omp for private(j,k)
               				for( k = 0 ; k < NUMOUT ; k ++ )    // update delta weights DeltaWeightHO
                       				for( j = 0 ; j < NUMHID ; j++ )
                       					DeltaWeightHO[k][j] = eta * Hidden[j] * DeltaO[k] + alpha * DeltaWeightHO[k][j];
				}

				//#pragma omp single//master
               			Error += BError;
				//#pragma omp barrier

				#pragma omp for private(i,j)
               			for( j = 0 ; j < NUMHID ; j++ )     // update weights WeightIH
					for(  i = 0 ; i < NUMIN ; i++ )
						WeightIH[j][i] += DeltaWeightIH[j][i] ;

				#pragma omp for private(j,k)
               			for( k = 0 ; k < NUMOUT ; k ++ )    // update weights WeightHO
                      			for( j = 0 ; j < NUMHID ; j++ )
                       				WeightHO[k][j] += DeltaWeightHO[k][j] ;
       			}
			//#pragma omp single // master
			//{
       			Error = Error/((NUMPAT/BSIZE)*BSIZE);	//mean error for the last epoch 		
       			if( !(epoch%100) ) printf( "\nEpoch %-5d :   Error = %f \n", epoch, Error ) ;
			if( Error < 0.0004 ) {
                               	printf( "\nEpoch %-5d :   Error = %f \n", epoch, Error ) ;
				break;
                        }
			//}
			//#pragma omp barrier

			//if( Error < 0.0004 ) {
			//	break ;  // stop learning when 'near enough'
       			//}
		}//end for epoch
	}//end parallel
	freeTSet( NUMPAT, tSet );

	printf( "END TRAINING\n" );
}


void printRecognized( int p, float Output[] ){
	int imax = 0;
	int i,k;

	//#pragma omp parallel num_threads(12)
	//{
	 	#pragma omp for private(i)
		for( i = 1; i < NUMOUT; i++)
			if ( Output[i] > Output[imax] ) imax = i;
		printf( "El patró %d sembla un %c\t i és un %d", p, '0' + imax, Validation[p] );
		if( imax == Validation[p] ) total++;
		#pragma omp for private(k)
    		for( k = 0 ; k < NUMOUT ; k++ )
        		printf( "\t%f\t", Output[k] ) ;
   		printf( "\n" );
	//}
}


void runN(){
	char **rSet;
	char *fname[NUMRPAT];
	int i,j,p,k;

	if( (rSet = loadPatternSet( NUMRPAT, "optdigits.cv", 0 )) == NULL){
		printf( "Error!!\n" );
		exit( -1 );
	}

	float Hidden[NUMHID], Output[NUMOUT];

	#pragma omp parallel num_threads(12)
	{
	 	#pragma omp for private(p,i,j)
    		for( p = 0 ; p < NUMRPAT ; p++ ) {    // repeat for all the recognition patterns
        		for( j = 0 ; j < NUMHID ; j++ ) {    // compute hidden unit activations
            			float SumH = 0.0;
            			for( i = 0 ; i < NUMIN ; i++ ) 
					SumH += rSet[p][i] * WeightIH[j][i];
            			Hidden[j] = 1.0/(1.0 + exp( -SumH )) ;
        		}

        		for( k = 0 ; k < NUMOUT ; k++ ) {    // compute output unit activations
            			float SumO = 0.0;
            			for( j = 0 ; j < NUMHID ; j++ ) 
					SumO += Hidden[j] * WeightHO[k][j] ;
            			Output[k] = 1.0/(1.0 + exp( -SumO )) ;   // Sigmoidal Outputs
        		}
        		printRecognized( p, Output );
    		}
	}

	printf( "\nTotal encerts = %d\n", total );

	freeTSet( NUMRPAT, rSet );
}

int main() {
	clock_t start = clock();
	printf("Llama trainN--------------------------------------------------------------\n");
	trainN();
	printf("Termina trainN y empieza runN---------------------------------------------\n");
	runN();
	printf("Termina runN--------------------------------------------------------------\n");

	clock_t end = clock();
	printf( "\n\nGoodbye! (%f sec)\n\n", (end-start)/(1.0*CLOCKS_PER_SEC) ) ;

	return 1 ;
}

/*******************************************************************************/
