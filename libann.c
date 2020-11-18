#include "libann.h"

void shuffle(int array[], int size) {
	for(int i = 0; i < size; i++) {
		int j = rand()%size;
		int t = array[i];
		array[i] = array[j];
		array[j] = t;
	}
}

void get_row(int n, int input[4][2], double output[2]) {
	for(int i = 0; i < 2; i++) {
		output[i] = input[n][i];
	}
	return;
}


double calc_error(const double target, const double predicted) {
	return 0.5 * (target-predicted) * (target-predicted);
}

/* Function to produce output of neurons in each layer */
void forward_prop(int num_in, int num_out, double input[num_in], double weight[num_out][num_in], double bias[num_out],double output[num_out]) {
	int i;
	/* Compute output of neurons in the same layer */
	for (i = 0; i < num_out; i++) {
		output[i] = neuron(num_in, input, weight[i], bias[i]);
	}
	return;
}

void init_network(double DeltaWeightIH[NUMHIDDEN][NUMINPUT],
		double DeltaWeightHO[NUMOUTPUT][NUMHIDDEN],
		double DeltaBiasIH[NUMHIDDEN], 
		double DeltaBiasHO[NUMOUTPUT],
		double WeightIH[NUMHIDDEN][NUMINPUT], 
		double WeightHO[NUMHIDDEN][NUMINPUT], 
		double biasIH[NUMHIDDEN], 
		double biasHO[NUMOUTPUT],
		const double smallwt)
{
	int i, j, k;
	for( j = 0 ; j < NUMHIDDEN ; j++ ) {    // initialize WeightIH and DeltaWeightIH
		for( i = 0 ; i < NUMINPUT ; i++ ) {
			DeltaWeightIH[j][i] = 0.0 ;
			WeightIH[j][i] = 2.0 * ( rando() - 0.5 ) * smallwt ;
		}
		DeltaBiasIH[j] = 0.0;
		biasIH[j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
	}

	for( k = 0 ; k < NUMOUTPUT ; k ++ ) {    // initialize WeightHO and DeltaWeightHO
		for( j = 0 ; j < NUMHIDDEN ; j++ ) {
			DeltaWeightHO[k][j] = 0.0 ;
			WeightHO[k][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
		}
		DeltaBiasHO[j] = 0.0;
		biasHO[k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
	}
	return;
}

void update_weights(double DeltaWeightIH[NUMHIDDEN][NUMINPUT],
		double DeltaWeightHO[NUMOUTPUT][NUMHIDDEN],
		double DeltaBiasIH[NUMHIDDEN], 
		double DeltaBiasHO[NUMOUTPUT],
		double WeightIH[NUMHIDDEN][NUMINPUT], 
		double WeightHO[NUMHIDDEN][NUMINPUT], 
		double biasIH[NUMHIDDEN], 
		double biasHO[NUMOUTPUT],
		const double DeltaO[NUMOUTPUT+1], 
		const double DeltaH[NUMHIDDEN+1],
		const double trIn_p[NUMINPUT],
		const double outputH[NUMHIDDEN],
		const double eta, const double alpha)
{
	int i, j, k;
	// update weights WeightIH
	for( j = 0 ; j < NUMHIDDEN ; j++ ) {     // update weights WeightIH
		DeltaBiasIH[j] = eta * DeltaH[j] + alpha * DeltaBiasIH[j] ;
		biasIH[j] += DeltaBiasIH[j] ;
		for( i = 0 ; i < NUMINPUT ; i++ ) {
			DeltaWeightIH[j][i] = eta * trIn_p[i] * DeltaH[j] + alpha * DeltaWeightIH[j][i];
			WeightIH[j][i] += DeltaWeightIH[j][i] ;
		}
	}

	// update weights WeightHO
	for( k = 0 ; k < NUMOUTPUT ; k ++ ) {    // update weights WeightHO
		DeltaBiasHO[k] = eta * DeltaO[k] + alpha * DeltaBiasHO[k] ;
		biasHO[k] += DeltaBiasHO[k] ;
		for( j = 0 ; j < NUMHIDDEN ; j++ ) {
			DeltaWeightHO[k][j] = eta * outputH[j] * DeltaO[k] + alpha * DeltaWeightHO[k][j] ;
			WeightHO[k][j] += DeltaWeightHO[k][j] ;
		}
	}

}

void back_prop(const double DeltaO[NUMOUTPUT+1], 
		double SumDOW[NUMHIDDEN+1], 
		double DeltaH[NUMHIDDEN+1],
		const double WeightHO[NUMOUTPUT][NUMHIDDEN],
		double outputH[NUMHIDDEN])
{
	int j, k;
	// 'back-propagate' errors to hidden layer
	for( j = 0 ; j < NUMHIDDEN ; j++ ) {
		SumDOW[j] = 0.0 ;
		for( k = 0 ; k < NUMOUTPUT ; k++ ) {
			SumDOW[j] += WeightHO[k][j] * DeltaO[k] ;
		}
		DeltaH[j] = SumDOW[j] * outputH[j] * (1.0 - outputH[j]) ;
	}
}

double update_network(double DeltaWeightIH[NUMHIDDEN][NUMINPUT],
		double DeltaWeightHO[NUMOUTPUT][NUMHIDDEN],
		double DeltaBiasIH[NUMHIDDEN], 
		double DeltaBiasHO[NUMOUTPUT],
		double WeightIH[NUMHIDDEN][NUMINPUT], 
		double WeightHO[NUMHIDDEN][NUMINPUT], 
		double biasIH[NUMHIDDEN], 
		double biasHO[NUMOUTPUT],
		double DeltaO[NUMOUTPUT+1], 
		double DeltaH[NUMHIDDEN+1],
		double SumDOW[NUMHIDDEN+1], 
		const double eta, const double alpha,
		int p)
{
	double trIn_p[NUMINPUT];
	double outputH[NUMHIDDEN];
	double outputO[NUMOUTPUT];
	double pred, error;

	get_row(p, trIn, trIn_p);

	forward_prop(NUMINPUT,  NUMHIDDEN,trIn_p , WeightIH, biasIH, outputH);
	forward_prop(NUMHIDDEN, NUMOUTPUT,outputH, WeightHO, biasHO, outputO);

	pred = outputO[0];

	current_o[p] = pred;
	error = calc_error((double)trOut[p], pred);
	DeltaO[0] = (trOut[p] - pred) * pred * (1.0 - pred) ;

	back_prop(DeltaO, SumDOW, DeltaH, WeightHO, outputH);

	update_weights(DeltaWeightIH, DeltaWeightHO, DeltaBiasIH, DeltaBiasHO,
			WeightIH, WeightHO, biasIH, biasHO, DeltaO, DeltaH,
			trIn_p, outputH, eta, alpha);

	return error;
}

void shuffle_index(int index[NUMTRAIN])
{
	int p;
	for( p = 0 ; p < NUMTRAIN ; p++ ) {     // randomize order of training patterns
		index[p] = p ;
	}
	shuffle(index, NUMTRAIN);

	return;
}


void record(void){
	FILE *fp;
	char *fname = "result.csv";
	fp = fopen( fname, "w" );
	int NumTrain = NUMTRAIN;
	for(int p = 0 ; p < NumTrain ; p++ ) {
		fprintf( fp, "%d,%d,%d,%f", trIn[p][0], trIn[p][1], trOut[p], current_o[p]);
		fprintf( fp,"\r\n");
	}
	fclose( fp );
	return;
}
