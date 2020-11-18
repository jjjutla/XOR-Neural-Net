#ifndef _LIBNN_H
#define _LIBNN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Global constants. */
#define NUMTRAIN  4
#define NUMINPUT  2
#define NUMHIDDEN 2
#define NUMOUTPUT 1
#define rando() ((double)rand()/((double)RAND_MAX+1))

/* Global variables. */
extern double current_o[NUMTRAIN];
extern int trIn[NUMTRAIN][NUMINPUT];
extern int trOut[NUMTRAIN];

/*  ---------------------- DO NOT MODIFY ABOVE THIS LINE! ---------------------- */

/* Functions to be implemented as part of the assignment. */
int read_data(void);
double neuron(const int num_in, const double input[num_in], const double weight[num_in], const double bias);
void learn(void);

/*  ---------------------- DO NOT MODIFY BELOW THIS LINE! ---------------------- */
void shuffle(int array[], int size);
void shuffle_index(int index[NUMTRAIN]);
void get_row(int n, int input[NUMTRAIN][NUMINPUT], double output[NUMINPUT]);
double calc_error(double target, double predicted);
void forward_prop(int num_in, int num_out, double input[num_in], double weight[num_out][num_in], double bias[num_out], double output[num_out]);
void error_init(double init_value);
void init_network(double DeltaWeightIH[NUMHIDDEN][NUMINPUT], double DeltaWeightHO[NUMOUTPUT][NUMHIDDEN], double DeltaBiasIH[NUMHIDDEN], double DeltaBiasHO[NUMOUTPUT],
		double WeightIH[NUMHIDDEN][NUMINPUT], double WeightHO[NUMHIDDEN][NUMINPUT], double biasIH[NUMHIDDEN], double biasHO[NUMOUTPUT],
		const double smallwt);
void update_weights(double DeltaWeightIH[NUMHIDDEN][NUMINPUT], double DeltaWeightHO[NUMOUTPUT][NUMHIDDEN], double DeltaBiasIH[NUMHIDDEN], double DeltaBiasHO[NUMOUTPUT],
		double WeightIH[NUMHIDDEN][NUMINPUT], double WeightHO[NUMHIDDEN][NUMINPUT], double biasIH[NUMHIDDEN], double biasHO[NUMOUTPUT],
		const double DeltaO[NUMOUTPUT+1], const double DeltaH[NUMHIDDEN+1], const double trIn_p[NUMINPUT], const double outputH[NUMHIDDEN], const double eta, const double alpha);
double update_network(double DeltaWeightIH[NUMHIDDEN][NUMINPUT], double DeltaWeightHO[NUMOUTPUT][NUMHIDDEN], double DeltaBiasIH[NUMHIDDEN], double DeltaBiasHO[NUMOUTPUT],
		double WeightIH[NUMHIDDEN][NUMINPUT], double WeightHO[NUMHIDDEN][NUMINPUT], double biasIH[NUMHIDDEN], double biasHO[NUMOUTPUT],
		double DeltaO[NUMOUTPUT+1], double DeltaH[NUMHIDDEN+1], double SumDOW[NUMHIDDEN+1], const double eta, const double alpha, int p);
void record();

#endif
