# XOR-Neural-Net
The aim is to get the network to learn how an XOR logic gate works. To do this, the network takes two inputs and produces one output. The inputs can take the values one or zero (i.e., true or false), and the network should output one (i.e., true) if either of the two inputs are true, and zero(i.e., false) if (i) both of the inputs are false or (ii) both of the inputs are true. The neural network program should learn to predict the correct output, given each of these possible inputs. The program was written in C and can be complied using gcc following the instructions in the README. Feel free to modify the in and out csv files to add your own data.

to run the program you will have to compile to code in your terminal. The following lines is the commands needed to compile and run using gcc:


$ gcc libann.c neuralnet.c ann.c -o ann

$ ./ann


feel free to edit the in.csv to change the data to anything you want.
