CFLAGS = -g -O0

training: readMNIST.o neuralnet.o training.o  
	c++ -g -O0 training.o readMNIST.o -o training
	cp training project

training.o: training.cpp readMNIST.h Makefile
	c++ -g -O0 training.cpp -c 

neuralnet.o: neuralnet.cpp Makefile
	c++ neuralnet.cpp -c

readMNIST.o: readMNIST.cpp readMNIST.h Makefile
	c++ -g -O0 readMNIST.cpp -c



