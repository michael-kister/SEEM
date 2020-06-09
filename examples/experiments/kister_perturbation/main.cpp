// g++ -Wall -O -g -o run_code workspace.cpp
 
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <fstream>
#include <set>

// more important
#include <perturbation.h>
//#include <eis_second_order.h>
//#include <gpu_pf.h>
#include <matrices.h>

#include <kister_perturbation.h>

/*------------------------------------------------------
 * Cheat Sheet for Powers of 2
 *------------------------------------------------------
 * 1....2     6......64    11....2,048    16......65,536
 * 2....4     7.....128    12....4,096    17.....131,072
 * 3....8     8.....256    13....8,192    18.....262,144
 * 4...16     9.....512    14...16,384    19.....524,288
 * 5...32    10...1,024    15...32,768    20...1,048,576
 *-----------------------------------------------------*/

//#include <bfgs.h>

//#include <cma_es.h>

int main ()
{

    double theta[19];
    theta[0]  = 2.0   ;
    theta[1]  = 5.0   ;
    theta[2]  = 0.991 ;
    theta[3]  = 0.005 ;
    theta[4]  = 1.5   ;
    theta[5]  = 0.25  ;
    theta[6]  = 0.1   ;
    theta[7]  = 6.0   ;
    theta[8]  = 0.75  ;
    theta[9]  = 0.357 ;
    theta[10] = 0.3   ;
    theta[11] = 0.0196;
    theta[12] = 0.0   ;
    theta[13] = 0.9   ;
    theta[14] = 0.0025;
    theta[15] =-1.6094;
    theta[16] = 0.8   ;
    theta[17] = 0.0025;
    theta[18] = 0.0025;
    
    MPK_SGU ss_model(1);// = MPK_SGU();

    ss_model.print_theta();
    
    ss_model.load_parameters(theta, 2);

    double state[] = {0.0,   0.0, 0.0, -.007, 0.0, 0.0};
    double control[] = {0.0, 0.0, 0.0, 0.0, 0.0};

    ss_model.decision(state, control);
    
    printmat4(control, 5,1,1,1);
    return 0;
}
