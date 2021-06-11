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
#include <cuda.h>

#include "common.h"


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
	for( i = 0; i < np; i++ ) free( tset[i] );
	free(tset);
}

__global__ void Kernel_SumH(int d_p, char* d_tSet, float* d_WeightIH, float* d_Hidden)
{
	__shared__ float d_SumH[NUMIN];
	
	unsigned int d_i = threadIdx.x;
	unsigned int d_j = blockIdx.x;
	unsigned int d_bsize = blockDim.x;

	d_SumH[d_i] = 0.0;

	int tSet_pos = (d_p * d_bsize) + d_i;
	int Weight_pos = (d_j * d_bsize) + d_i;

	d_SumH[d_i] = d_tSet[tSet_pos] * d_WeightIH[Weight_pos];

	for (unsigned int stride = d_bsize/2; stride >= 1; stride >>=1)
	{
		__syncthreads();
		if (d_i < stride)
			d_SumH[d_i] += d_SumH[d_i + stride];
	}

	if (d_i == 0){
		__syncthreads();
		d_Hidden[d_j] = 1.0/(1.0 + exp( -d_SumH[0] ));
	}
}

__global__ void Kernel_SumO(int d_p, float *d_Hidden, float *d_WeightHO, float *d_Output, /*float *d_BError,*/ float temp_BError, float *d_Target, float *d_DeltaO)
{
	__shared__ float d_SumO[NUMHID];
	__shared__ float d_BError[NUMOUT];

	unsigned int d_j = threadIdx.x;
        unsigned int d_k = blockIdx.x;
        unsigned int d_bsize = blockDim.x;
	int Weight_pos = (d_k * d_bsize) + d_j;

	d_SumO[d_j] = d_Hidden[d_j] * d_WeightHO[Weight_pos];

	for (unsigned int stride = d_bsize/2; stride >= 1; stride >>=1)
        {
                __syncthreads();
                if (d_j < stride)
                        d_SumO[d_j] += d_SumO[d_j + stride];
        }

        if (d_j == 0){
		__syncthreads();
		d_Output[d_k] = 1.0/(1.0 + exp(d_SumO[0]));
	}

	int Target_pos = (d_p * d_bsize) + d_k;

	d_BError[d_k] = 0.5 * (d_Target[Target_pos] - d_Output[d_k]) * (d_Target[Target_pos] - d_Output[d_k]);

	for (unsigned int stride = d_bsize/2; stride >= 1; stride >>=1)
        {
                __syncthreads();
                if (d_k < stride)
//                        d_BError[d_k] += d_BError[d_k + stride];
			temp_BError += d_BError[d_k + stride];
        }

	if (d_j == 0){
                __syncthreads();
                d_DeltaO[d_k] = (d_Target[Target_pos] - d_Output[d_k]) * d_Output[d_k] * (1.0 - d_Output[d_k]);
        }
}

__global__ void Kernel_SumDOW(float *d_WeightHO, float *d_DeltaO, float *d_DeltaH, float *d_Hidden)
{
	__shared__ float d_SumDOW[NUMOUT];

	unsigned int d_k = threadIdx.x;
	unsigned int d_j = blockIdx.x;
	unsigned int d_bsize = blockDim.x;

	d_SumDOW[d_k] = 0.0;

	int pos_Weight = (d_k * d_bsize) + d_j;
	d_SumDOW[d_k] = d_WeightHO[pos_Weight] * d_DeltaO[d_k];

	for (unsigned int stride = d_bsize/2; stride >= 1; stride >>=1)
        {
                __syncthreads();
                if (d_k < stride)
                        d_SumDOW[d_k] += d_SumDOW[d_k + stride];
        }

	if (d_k == 0)
	{
		__syncthreads();
		d_DeltaH[d_j] = d_SumDOW[0] * d_Hidden[d_j] * (1.0 - d_Hidden[d_j]);
	}
}

__global__ void Kernel_DeltaIH(int d_p, char *d_tSet, float *d_DeltaWeightIH, float *d_DeltaH, int d_eta, int d_alpha)
{
	unsigned int d_i = threadIdx.x;
        unsigned int d_j = blockIdx.x;
        unsigned int d_bsize = blockDim.x;

	int pos_Delta = (d_j * d_bsize) + d_i;
	int pos_tSet = (d_p * d_bsize) + d_i;

	__syncthreads();
	d_DeltaWeightIH[pos_Delta] = d_eta * d_tSet[pos_tSet] * d_DeltaH[d_j] + d_alpha * d_DeltaWeightIH[pos_Delta];
}

__global__ void Kernel_DeltaHO(float *d_DeltaWeightHO, float *d_DeltaO, float *d_Hidden, int d_eta, int d_alpha)
{
	unsigned int d_j = threadIdx.x;
        unsigned int d_k = blockIdx.x;
        unsigned int d_bsize = blockDim.x;

	int pos_DeltaHO = (d_k * d_bsize) + d_j;

	d_DeltaWeightHO[pos_DeltaHO] = d_eta * d_Hidden[d_j] * d_DeltaO[d_k] + d_alpha * d_DeltaWeightHO[pos_DeltaHO];
}

__global__ void Kernel_WeightIH(float *d_WeightIH, float *d_DeltaWeightIH)
{
	unsigned int d_j = threadIdx.x;
        unsigned int d_i = blockIdx.x;
        unsigned int d_bsize = blockDim.x;

        int pos_Weight = (d_j * d_bsize) + d_i;

	d_WeightIH[pos_Weight] += d_DeltaWeightIH[pos_Weight];
}

__global__ void Kernel_WeightHO(float *d_WeightHO, float *d_DeltaWeightHO)
{
	unsigned int d_j = threadIdx.x;
        unsigned int d_k = blockIdx.x;
        unsigned int d_bsize = blockDim.x;

        int pos_Weight = (d_k * d_bsize) + d_j;

	d_WeightHO[pos_Weight] += d_DeltaWeightHO[pos_Weight];
}


void trainN(){
	char **tSet;

   	float DeltaWeightIH[NUMHID][NUMIN], DeltaWeightHO[NUMOUT][NUMHID];
	float Error, BError, eta = 0.3, alpha = 0.5, smallwt = 0.22;
	int ranpat[NUMPAT];
 	float Hidden[NUMHID], Output[NUMOUT], DeltaO[NUMOUT], DeltaH[NUMHID];
	float SumO, SumDOW;//SumH;

	//variables bucle 1
	char *d_tSet;
	float *d_WeightIH, *d_Hidden;
	int size_tSet = NUMPAT * 1024 * sizeof(char);
	int size_WeightIH =  NUMHID * NUMIN * sizeof(float);
	int size_Hidden = NUMHID * sizeof(float);
	cudaMalloc((void **) &d_tSet, size_tSet);
	cudaMalloc((void **) &d_WeightIH, size_WeightIH);
	cudaMalloc((void **) &d_Hidden, size_Hidden);

	//variables bucle 2
	float *d_WeightHO, *d_Output, *d_Target, *d_DeltaO;
//	float *d_BError;
	float temp_BError;
	int size_WeightHO = NUMOUT * NUMHID * sizeof(float);
	int size_Output = NUMOUT * sizeof(float);
	int size_Target = NUMPAT * NUMOUT * sizeof(float);
	int size_DeltaO = NUMOUT * sizeof(float);
//	int size_BError = NUMOUT * sizeof(float);
	cudaMalloc((void **) &d_WeightHO, size_WeightHO);
	cudaMalloc((void **) &d_Output, size_Output);
	cudaMalloc((void **) &d_Target, size_Target);
	cudaMalloc((void **) &d_DeltaO, size_DeltaO);
//	cudaMalloc((void **) &d_BError, size_BError);
	cudaMemcpy(d_Target, Target, size_Target, cudaMemcpyHostToDevice);

	//variables
	float *d_DeltaH, *d_DeltaWeightIH, *d_DeltaWeightHO;
	int size_DeltaH = NUMHID * sizeof(float);
	int size_DeltaWeightIH = NUMHID * NUMIN * sizeof(float);
	int size_DeltaWeightHO = NUMOUT * NUMHID * sizeof(float);
	cudaMalloc((void **) &d_DeltaH, size_DeltaH);
	cudaMalloc((void **) &d_DeltaWeightIH, size_DeltaWeightIH);
	cudaMalloc((void **) &d_DeltaWeightHO, size_DeltaWeightHO);

//	for (int i = 0; i < NUMPAT; i++){
//		int pos_tSet = i * 1024;
//		cudaMemcpy(d_tSet, tSet, size_tSet, cudaMemcpyHostToDevice);
//		cudaMemcpy2D(d_tSet, pos_tSet, tSet, pos_tSet, 1024, NUMPAT, cudaMemcpyHostToDevice);
//	}

	if( (tSet = loadPatternSet( NUMPAT, "optdigits.tra", 1 ) ) == NULL){
       		printf( "Loading Patterns: Error!!\n" );
		exit( -1 );
	}

	for (int i = 0; i < NUMPAT; i++)
		cudaMemcpy(&d_tSet[i * NUMIN], tSet[i], NUMIN, cudaMemcpyHostToDevice);

	for( int i = 0; i < NUMHID; i++ )
		for( int j = 0; j < NUMIN; j++ ){
			WeightIH[i][j] = 2.0 * ( frando() + 0.01 ) * smallwt;
			DeltaWeightIH[i][j] = 0.0;
		}

	for( int i = 0; i < NUMOUT; i++)
		for( int j = 0; j < NUMHID; j++ ){
			WeightHO[i][j] = 2.0 * ( frando() + 0.01 ) * smallwt;
			DeltaWeightHO[i][j] = 0.0;
		}

    	for( int epoch = 0 ; epoch < 1000000 ; epoch++ ) {    // iterate weight updates
        	for( int p = 0 ; p < NUMPAT ; p++ )   // randomize order of individuals
        		ranpat[p] = p;
        	
		for( int p = 0 ; p < NUMPAT ; p++) {
                	int x = rando();
                	int np = (x*x)%NUMPAT;
                	int op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
        	}

        	Error = BError = 0.0;
		printf("."); fflush(stdout);

		for ( int nb = 0; nb < NUMPAT/BSIZE; nb++) { // repeat for all batches
			BError = 0.0;
      	       		for(int  np = nb*BSIZE ; np < (nb + 1)*BSIZE ; np++ ){//repeat for all the training patterns within the batch
       	       			int p = ranpat[np];
        	       			
				/*for(int j = 0 ; j < NUMHID ; j++ ) {    // compute hidden unit activations
        		       		SumH = 0.0;
        		       		for( int i = 0 ; i < NUMIN ; i++ ) 
						SumH += tSet[p][i] * WeightIH[j][i];
        	       			Hidden[j] = 1.0/(1.0 + exp( -SumH )) ;
        	       		}*/

//				cudaMalloc((char **) &d_tSet, size_tSet);
//				cudaMalloc((float **) &d_WeightIH, NUMHID * NUMIN * sizeof(float));
//				cudaMalloc((float **) &d_Hidden_out, NUMHID * sizeof(float));

//				cudaMemcpy(d_tSet, tSet, size_tSet, cudaMemcpyHostToDevice);
				cudaMemcpy(d_WeightIH, WeightIH, size_WeightIH /*NUMHID * NUMIN*/, cudaMemcpyHostToDevice);

				Kernel_SumH<<< NUMHID, NUMIN >>>(p, d_tSet, d_WeightIH, d_Hidden);

				cudaMemcpy(Hidden, d_Hidden, size_Hidden /*NUMHID*/, cudaMemcpyDeviceToHost);

        	 	      	for(int k = 0 ; k < NUMOUT ; k++ ) {    // compute output unit activations and errors
       					SumO = 0.0;	
       					for( int j = 0 ; j < NUMHID ; j++ ) 
						SumO += Hidden[j] * WeightHO[k][j] ;
       					Output[k] = 1.0/(1.0 + exp(-SumO)) ;   // Sigmoidal Outputs
       					BError += 0.5 * (Target[p][k] - Output[k]) * (Target[p][k] - Output[k]) ;   // SSE
       					DeltaO[k] = (Target[p][k] - Output[k]) * Output[k] * (1.0 - Output[k]) ;   // Sigmoidal Outputs, SSE
       	       			}	

/*				cudaMemcpy(d_WeightHO, WeightHO, size_WeightHO , cudaMemcpyHostToDevice);
				cudaMemcpy(d_Hidden, Hidden, size_Hidden, cudaMemcpyHostToDevice);
//				cudaMemcpy(d_BError, BError, size_BError, cudaMemcpyHostToDevice);

				printf("antes de ir a GPU");
				Kernel_SumO<<< NUMOUT, NUMHID >>>(p, d_Hidden, d_WeightHO, d_Output,*/ /*d_BError*//*temp_BError, d_Target, d_DeltaO);
				printf("vuelto de la GPU");

				int size_BError = sizeof(BError);
				cudaMemcpy(Output, d_Output, size_Output, cudaMemcpyDeviceToHost);
				cudaMemcpy(BError, temp_BError, size_BError, cudaMemcpyDeviceToHost);
				cudaMemcpy(DeltaO, d_DeltaO, size_DeltaO, cudaMemcpyDeviceToHost);
*/				
	        	       	for(int  j = 0 ; j < NUMHID ; j++ ) {     // update delta weights DeltaWeightIH
        		       		SumDOW = 0.0 ;	
        		       	       	for(int k = 0 ; k < NUMOUT ; k++ ) 
						SumDOW += WeightHO[k][j] * DeltaO[k] ;
       					DeltaH[j] = SumDOW * Hidden[j] * (1.0 - Hidden[j]) ;
       					for(int i = 0 ; i < NUMIN ; i++ )
       						DeltaWeightIH[j][i] = eta * tSet[p][i] * DeltaH[j] + alpha * DeltaWeightIH[j][i];
       				}


/*				cudaMemcpy(d_WeightHO, WeightHO, size_WeightHO, cudaMemcpyHostToDevice);
				cudaMemcpy(d_DeltaO, DeltaO, size_DeltaO, cudaMemcpyHostToDevice);
//				cudaMemcpy(d_Hidden, Hidden, size_Hidden, cudaMemcpyHostToDevice);

				Kernel_SumDOW<<< NUMHID, NUMOUT >>>(d_WeightHO, d_DeltaO, d_DeltaH, d_Hidden);

				cudaMemcpy(DeltaH, d_DeltaH, size_DeltaH, cudaMemcpyDeviceToHost);
//				cudaMemcpy(d_DeltaH, DeltaH, size_DeltaH, cudaMemcpyHostToDevice);
				cudaMemcpy(d_DeltaWeightIH, DeltaWeightIH, size_DeltaWeightIH, cudaMemcpyHostToDevice);

				Kernel_DeltaIH<<< NUMHID, NUMIN >>>(p, d_tSet, d_DeltaWeightIH, d_DeltaH, eta, alpha);

				cudaMemcpy(DeltaWeightIH, d_DeltaWeightIH, size_DeltaWeightIH, cudaMemcpyDeviceToHost);
*/
/*               			for( int k = 0 ; k < NUMOUT ; k ++ )    // update delta weights DeltaWeightHO
                       			for(int  j = 0 ; j < NUMHID ; j++ )
                       				DeltaWeightHO[k][j] = eta * Hidden[j] * DeltaO[k] + alpha * DeltaWeightHO[k][j];
*/
				cudaMemcpy(d_DeltaWeightHO, DeltaWeightHO, size_DeltaWeightHO, cudaMemcpyHostToDevice);
				cudaMemcpy(d_DeltaO, DeltaO, size_DeltaO, cudaMemcpyHostToDevice);

				Kernel_DeltaHO<<< NUMOUT, NUMHID >>>(d_DeltaWeightHO, d_DeltaO, d_Hidden, eta, alpha);

				cudaMemcpy(DeltaWeightHO, d_DeltaWeightHO, size_DeltaWeightHO, cudaMemcpyDeviceToHost);
			}

               		Error += BError;

               		for(int  j = 0 ; j < NUMHID ; j++ )     // update weights WeightIH
				for(int  i = 0 ; i < NUMIN ; i++ )
					WeightIH[j][i] += DeltaWeightIH[j][i] ;

               		for( int k = 0 ; k < NUMOUT ; k ++ )    // update weights WeightHO
                		for(int j = 0 ; j < NUMHID ; j++ )
                       			WeightHO[k][j] += DeltaWeightHO[k][j] ;

/*			cudaMemcpy(d_DeltaWeightIH, DeltaWeightIH, size_DeltaWeightIH, cudaMemcpyHostToDevice);
                	cudaMemcpy(d_WeightIH, WeightIH, size_WeightIH, cudaMemcpyHostToDevice);

                	Kernel_WeightIH<<< NUMHID, NUMIN >>>(d_WeightIH, d_DeltaWeightIH);

                	cudaMemcpy(WeightIH, d_WeightIH, size_WeightIH, cudaMemcpyDeviceToHost);
                	cudaMemcpy(d_DeltaWeightHO, DeltaWeightHO, size_DeltaWeightHO, cudaMemcpyHostToDevice);
                	cudaMemcpy(d_WeightHO, WeightHO, size_WeightHO, cudaMemcpyHostToDevice);

                	Kernel_WeightHO<<< NUMOUT, NUMHID >>>(d_WeightHO, d_DeltaWeightHO);

                	cudaMemcpy(WeightHO, d_WeightHO, size_WeightHO, cudaMemcpyDeviceToHost);
*/       		}

       		Error = Error/((NUMPAT/BSIZE)*BSIZE);	//mean error for the last epoch 		
       		if( !(epoch%100) ) printf( "\nEpoch %-5d :   Error = %f \n", epoch, Error ) ;
		if( Error < 0.0004 ) {
                       	printf( "\nEpoch %-5d :   Error = %f \n", epoch, Error ) ;
			break;
                }
	}

	freeTSet( NUMPAT, tSet );
	cudaFree(d_tSet);
	cudaFree(d_WeightIH);
	cudaFree(d_Hidden);
	cudaFree(d_WeightHO);
	cudaFree(d_Output);
	cudaFree(d_Target);
	cudaFree(d_DeltaO);
	cudaFree(d_DeltaH);
//	cudaFree(d_BError);

	printf( "END TRAINING\n" );
}


void printRecognized( int p, float Output[] ){
	int imax = 0;
	int i,k;

	for( i = 1; i < NUMOUT; i++)
		if ( Output[i] > Output[imax] ) imax = i;
	printf( "El patró %d sembla un %c\t i és un %d", p, '0' + imax, Validation[p] );
	if( imax == Validation[p] ) 
		total++;
	for( k = 0 ; k < NUMOUT ; k++ )
       		printf( "\t%f\t", Output[k] ) ;
	printf( "\n" );
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

	printf( "\nTotal encerts = %d\n", total );

	freeTSet( NUMRPAT, rSet );
}

int main() {
	clock_t start = clock();
	trainN();
	runN();

	clock_t end = clock();
	printf( "\n\nGoodbye! (%f sec)\n\n", (end-start)/(1.0*CLOCKS_PER_SEC) ) ;

	return 1 ;
}

/*******************************************************************************/
