#!/usr/bin/bash

make clean

make

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/michael/adolc_base/lib64

./xmain
