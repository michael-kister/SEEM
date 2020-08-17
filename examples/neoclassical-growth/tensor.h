#ifndef TENSOR_H
#define TENSOR_H

// for abort
#include <stdlib.h>

// Index class
#include "index.h"

// linear algebra
#include <lapacke.h>
#include <cblas.h>


class Tensor {
public:

    // Members
    int dimension;
    intvec sizes;
    int length;
    std::vector<double> X;

    // Constructors
    Tensor(const intvec& s_);
    Tensor(const intvec& s_, const double* X_);
    Tensor(void);

    // Methods for working with indices
    Index& Increment(Index& I) const;
    int Address(const Index& I) const;

    // Change the dimensionality of the tensor
    void FlattenFull(void);
    void FlattenPartial(void);
    void Unflatten(void);

    // Re-ordering the access order for elements
    void Permute(const intvec& P);
    
    // Display tensor
    void print0(void) const;
    void print(void) const;

    // Subscripting operators
    double& operator[](const Index& I);
    double  operator[](const Index& I) const;

    // Binary operators
    Tensor& operator*=(double d);            // A = A * d
    Tensor& operator+=(const Tensor& B);     // A = A + B
    Tensor& operator-=(const Tensor& B);     // A = A - B
    Tensor& operator*=(Tensor B);            // A = A * B
    Tensor& operator|=(Tensor B);            // A = A \ B
    Tensor& operator/=(Tensor B);            // A = A / B
    Tensor& operator<<=(const Tensor& B);    // A = A ⊗ B

    // Unary operators
    Tensor& operator^=(const intvec& P);     // permute
    Tensor& operator~();                     // FlattenFull
    Tensor& operator++();                    // add dimension to beginning
    Tensor& operator++(int);                 // add dimension to end
    
};
Tensor operator+ (Tensor A, const Tensor& B);
Tensor operator- (Tensor A, const Tensor& B);
Tensor operator* (Tensor A, double d);
Tensor operator* (double d, Tensor A);
Tensor operator* (Tensor A, const Tensor& B);
Tensor operator| (Tensor A, const Tensor& B);
Tensor operator/ (Tensor A, const Tensor& B);
Tensor operator<<(Tensor A, const Tensor& B);
Tensor operator^ (Tensor A, const intvec& P);

#endif
