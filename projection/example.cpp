// g++ -Wall -O -g -o run_code workspace.cpp
 
#include "smolyak.h"

#include <functional>

#include <iostream>

#include <cmath>

#define MAP2SMO(x,a,b) ((2.0*(x-a)/(b-a))-1.0)
#define MAP2FUN(x,a,b) (((b-a)*(x+1.0)/2.0)+a)

int main ()
{
    printf("\n\n");
    
    double pi = 3.141592653589793238462643;
    
    int dim = 3; // (ρ, φ, θ)

    int q = 5;
    
    int mu  = q - dim;
    
    Grid G = smolyak_grid(dim, mu);
    
    Approximation X(dim, mu);
    Approximation Y(dim, mu);
    Approximation Z(dim, mu);
    
    // X = ρ * sin(φ) * cos(θ)
    std::function<double(const std::vector<double>&)> fX = [](const std::vector<double>& S){
	return S[0] * sin(S[1]) * cos(S[2]);
    };

    // Y = ρ * sin(φ) * sin(θ)
    std::function<double(const std::vector<double>&)> fY = [](const std::vector<double>& S){
	return S[0] * sin(S[1]) * sin(S[2]);
    };

    // Z = ρ * cos(φ)
    std::function<double(const std::vector<double>&)> fZ = [](const std::vector<double>& S){
	return S[0] * cos(S[1]);
    };

    std::map<Point, double> graph_x;
    std::map<Point, double> graph_y;
    std::map<Point, double> graph_z;

    std::vector<double> vec;

    double LB[] = {0.0, -0.5*pi, -0.5*pi};
    double UB[] = {1.0,  0.5*pi,  0.5*pi};
    
    //double LB[] = {0.0, -0.5, -0.5};
    //double UB[] = {2.0,  1.5,  1.5};
    
    for (const auto& it : G) {

	vec = static_cast<std::vector<double>>(it);

	for (int i = 0; i < 3; i++)
	    vec[i] = MAP2FUN(vec[i], LB[i], UB[i]);
	
	graph_x.insert(std::pair<Point,double>(it, fX(vec)));
	graph_y.insert(std::pair<Point,double>(it, fY(vec)));
	graph_z.insert(std::pair<Point,double>(it, fZ(vec)));
	
    }
    
    X.set_coefficients(graph_x);
    Y.set_coefficients(graph_y);
    Z.set_coefficients(graph_z);
    
    std::vector<Approximation> dX(3, X);
    std::vector<Approximation> dY(3, Y);
    std::vector<Approximation> dZ(3, Z);

    for (int i = 0; i < 3; i++) {

	dX[i].differentiate(i);
	dY[i].differentiate(i);
	dZ[i].differentiate(i);

    }

    std::function<double(const std::vector<double>&)> J1 = [&](const std::vector<double>& S){
	
	double d1 = 2.0 / (UB[0] - LB[0]);
	double d2 = 2.0 / (UB[1] - LB[1]);
	double d3 = 2.0 / (UB[2] - LB[2]);
	
	double p1 = dX[0](S)*(dY[1](S)*dZ[2](S)-dY[2](S)*dZ[1](S));
	double p2 = dX[1](S)*(dY[0](S)*dZ[2](S)-dY[2](S)*dZ[0](S));
	double p3 = dX[2](S)*(dY[0](S)*dZ[1](S)-dY[1](S)*dZ[0](S));
	
	return (d1 * d2 * d3) * (p1 - p2 + p3);
    };

    std::function<double(const std::vector<double>&)> J2 = [&](const std::vector<double>& S){
	return S[0]*S[0]*sin(S[1]);
    };

    double j1;

    int i = 0;
    
    for (const auto& it : G) {

	if (++i % 10 == 0) {

	    vec = static_cast<std::vector<double>>(it);

	    j1 = J1(vec);
	
	    for (int j = 0; j < 3; j++)
		vec[j] = MAP2FUN(vec[j], LB[j], UB[j]);
	
	
	    printf("True = %10.7f  Approx. = %10.7f\t", J2(vec), j1);
	    it.print();
	}
    }
    
    printf("\n\n");
    
    return 0;
}

