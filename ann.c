/* To compile and run:
   $ gcc -c template.c
   $ gcc template.o mymain.c -o mymain
   $ ./mymain 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "libann.h"

int main(void) {
	read_data();
	learn();
	record();
	return 0;
}
