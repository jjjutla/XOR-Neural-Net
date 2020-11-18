#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "libann.h"

int trIn[NUMTRAIN][NUMINPUT] = {0};
int trOut[NUMTRAIN] = {0};
double current_o[NUMTRAIN] = {0};



/*
 * Function to read training data.  Data from "in.csv" is stored in array
 * "trIn", and that from "out.csv" in array "trOut".
 *
 * Returns zero if success, or one if there is an error.
 */

int read_data(void) {

	int i=0;
	int j=0;

	FILE *input=fopen("in.csv","r");
	if(input==NULL){
		fprintf(stderr, "Error: no inputs found\n");
		return 1;
	}

	for (i=0; i<4; i++){
        for (j=0; j<2;j++){
            fscanf(input,"%d %*[,]" , &trIn[i][j]);
		}
	}

	i = 0;

	fclose (input);

	FILE *output=fopen("out.csv","r");
	if(output==NULL){
		fprintf(stderr,"Error: no outputs found\n");
		return 1;
	}

	for(j=0; j<4; j++){
		fscanf(output, "%d",&trOut[j]);
	}

	fclose (output);

	return 0;

}

/* Function to produce output of one neuron */
double neuron(const int num_in, const double input[num_in], const double weight[num_in], const double bias) {


	double zj=0, temp=0, res=0;
	for(int i=0; i< num_in; i++)
    {

        temp=input[i]*weight[i];
        res=res+temp+bias;

    }

    zj=1/(1+exp(-res+bias));
    return zj;

}

void learn(void) {
	int i, k, p;
	double DeltaWeightIH[NUMHIDDEN][NUMINPUT],
		   DeltaWeightHO[NUMOUTPUT][NUMHIDDEN],
		   DeltaBiasIH[NUMHIDDEN],
		   DeltaBiasHO[NUMOUTPUT];
	double WeightIH[NUMHIDDEN][NUMINPUT], WeightHO[NUMOUTPUT][NUMHIDDEN];
	double DeltaO[NUMOUTPUT+1], SumDOW[NUMHIDDEN+1], DeltaH[NUMHIDDEN+1];
	double biasIH[NUMHIDDEN], biasHO[NUMOUTPUT];
	double eta = 0.5, alpha = 0.9, smallwt = 0.5;
	int index[NUMTRAIN]={0};
	double error;

	init_network(DeltaWeightIH, DeltaWeightHO, DeltaBiasIH, DeltaBiasHO,
			WeightIH, WeightHO, biasIH, biasHO, smallwt);



    error =0;
    int z;

    for(i=0;i<1000;i++){
        for(z=0; z<4; z++){
            error = update_network(DeltaWeightIH, DeltaWeightHO,
			        DeltaBiasIH, DeltaBiasHO,
					WeightIH, WeightHO, biasIH, biasHO,
					DeltaO, DeltaH, SumDOW, eta, alpha, index[z]);
        }
        shuffle_index(index);
    }



	/* --------------- End Answer to Task 3 Here -------------- */

	printf("\n\nPat\t") ;   // print network outputs
	for( i = 1 ; i <= NUMINPUT ; i++ ) {
		fprintf(stdout, "Input%-1d\t", i) ;
	}
	for( k = 1 ; k <= NUMOUTPUT ; k++ ) {
		fprintf(stdout, "Target%-1d\tOutput%-1d\t", k, k) ;
	}
	for( p = 0 ; p < NUMTRAIN ; p++ ) {
		fprintf(stdout, "\n%d\t", p) ;
		for( i = 0 ; i < NUMINPUT ; i++ ) {
			fprintf(stdout, "%d\t", trIn[p][i]) ;
		}
		for( k = 1 ; k <= NUMOUTPUT ; k++ ) {
			fprintf(stdout, "%d\t%f\t", trOut[p], current_o[p]) ;
		}
	}
	printf("\n");
	return;
}

