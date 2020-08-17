
CXX = g++

INCLUDEDIR = /home/michael/repos/ADOL-C/ADOL-C/include/ /usr/local/include/lapack/ /usr/local/include/blas/
IFLAGS = $(foreach d, $(INCLUDEDIR), -I$d)

CFLAGS = -O3 -Wall --std=c++11

LIBDIR = /home/michael/adolc_base/lib64 /usr/local/lib/ .

LFLAGS = $(foreach d, $(LIBDIR), -L $d) -ladolc -ltensor \
	-llapacke -llapack -lrefblas -lgfortran -lcblas -lm




xmain : main.cpp libtensor.a
	$(CXX) $(CFLAGS) $(IFLAGS) -o $@ $< $(LFLAGS)

libtensor.a : index.o tensor.o
	ar rc $@ $^

index.o : index.cpp index.hpp
	$(CXX) $(CFLAGS) $(IFLAGS) -c -o $@ $<

tensor.o : tensor.cpp tensor.hpp index.cpp index.hpp
	$(CXX) $(CFLAGS) $(IFLAGS) -c -o $@ $<

clean :
	rm -f *.o *.a xmain 

