
CXX = g++

INCLUDEDIR = /home/michael/repos/ADOL-C/ADOL-C/include/
IFLAGS = $(foreach d, $(INCLUDEDIR), -I$d)

CFLAGS = -O3 -Wall --std=c++11

LIBDIR = /home/michael/adolc_base/lib64
LFLAGS = $(foreach d, $(LIBDIR), -L $d) -ladolc -lm
#LFLAGS = /home/michael/adolc_base/lib64/libadolc.so.2

xmain : main.cpp
	$(CXX) $(CFLAGS) $(IFLAGS) -o $@ $^ $(LFLAGS)

clean :
	rm -f *.o x*

