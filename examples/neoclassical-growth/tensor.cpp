#include "tensor.hpp"

/**
 *
 *------------------------------------------------------------------------------ 
 */
Index& Tensor::Increment
(Index& I) const
{
    for (int d = dimension-1; d >= 0; --d) {
	if (I[d] < sizes[d]-1) {
	    ++I[d];
	    break;
	} else {
	    I[d] = 0;
	    if (d == 0)
		I.status = 0;
	}
    }
    return I;
}

/**
 *
 *------------------------------------------------------------------------------ 
 */
int Tensor::Address
(const Index& I) const
{
    int address = 0;
    int factor;
    for (int i = 0; i < dimension; ++i) {
	factor = 1;
	for (int j = i+1; j < dimension; ++j) {
	    factor *= sizes[j];
	}
	address += factor * I[i];
    }
    return address;
}
    
/**
 *
 *------------------------------------------------------------------------------ 
 */
Tensor::Tensor
(const intvec& s_) :
    sizes(s_)
{
    dimension = sizes.size();
    length = !!dimension;
    for (int i = 0; i < dimension; ++i)
	length *= sizes[i];
    X = std::vector<double>(length, 0);
}

/**
 *
 *------------------------------------------------------------------------------ 
 */
Tensor::Tensor
(void) :
    Tensor(intvec())
{
}

/**
 *
 *------------------------------------------------------------------------------ 
 */
Tensor::Tensor
(const intvec& s_, const double* X_) :
    Tensor(s_)
{
    for (int i = 0; i < length; ++i)
	X[i] = X_[i];
}
    
/**
 *
 *------------------------------------------------------------------------------ 
 */
double& Tensor::operator[]
(const Index& I)
{
    return X[Address(I)];
}

/**
 *
 *------------------------------------------------------------------------------ 
 */
double Tensor::operator[]
(const Index& I) const
{
    return X[Address(I)];
}

/**
 *
 *------------------------------------------------------------------------------ 
 */
void Tensor::Permute
(const intvec& P)
{
    intvec sz(dimension,0);
    for (int i = 0; i < dimension; ++i)
	sz[i] = sizes[P[i]];
    Tensor T(sz);
    for (Index I(intvec(dimension,0)); I.status; Increment(I))
	T[I.Permute(P)] = this->operator[](I);
	
    std::swap(T, *this);
}
    
/**
 *
 *------------------------------------------------------------------------------ 
 */
void Tensor::FlattenFull
(void)
{
    // if the dimension is odd, add a vacuous dimension
    if (dimension % 2) {
	sizes.push_back(1);
	++dimension;
    }

    // create permutation argument, as well as sizes for
    // new 2D array. P = [0,2,4,6,1,3,5,7]
    intvec P(dimension,0);
    intvec s(2,1);
    for (int i = 0; i < dimension/2; ++i) {
	P[i]               = 2*i;
	P[i+(dimension/2)] = 2*i+1;
	    
	s[0] *= sizes[2*i];
	s[1] *= sizes[2*i+1];
    }

    // permute array
    this->Permute(P);

    // now relabel dimensions and sizes
    this->sizes = s;
    this->dimension = 2;
}

// only this one can be undone
/**
 *
 *------------------------------------------------------------------------------ 
 */
void Tensor::FlattenPartial
(void)
{
    if (dimension % 2) {
	sizes.push_back(1);
	++dimension;
    }
    intvec P(dimension,0); // P = [0,2,4,6,1,3,5,7]
    for (int i = 0; i < dimension/2; ++i) {
	P[i]               = 2*i;
	P[i+(dimension/2)] = 2*i+1;
    }
    this->Permute(P);
}

// this can only be done using partial
/**
 *
 *------------------------------------------------------------------------------ 
 */
void Tensor::Unflatten
(void)
{
    intvec P(dimension,0);
    for (int i = 0; i < dimension/2; ++i) {
	P[2*i+0] = i+0;
	P[2*i+1] = i+dimension/2;
    }
    this->Permute(P);
}
    
/**
 *
 *------------------------------------------------------------------------------ 
 */
void Tensor::print0
(void) const
{
    for (Index I(intvec(dimension,0)); I.status; Increment(I)) {
	printf("S");
	I.print0();
	printf(" = %f\n", X[Address(I)]);
    }
}


/**
 *
 *------------------------------------------------------------------------------ 
 */
Tensor& Tensor::operator+=
(const Tensor& B)
{
    for (int i = 0; i < length; ++i)
	X[i] += B.X[i];
    return *this;
}

/**
 *
 *------------------------------------------------------------------------------ 
 */
Tensor& Tensor::operator-=
(const Tensor& B)
{
    for (int i = 0; i < length; ++i)
	X[i] -= B.X[i];
    return *this;
}

/**
 *
 *------------------------------------------------------------------------------ 
 */
Tensor& Tensor::operator|=
(Tensor B)
{
    // Solves for C in the system of linear equations:
    //     A*C = B,
    // i.e. the output is equal to A\B.

    // The intention is only for 2D tensors (matrices); for
    // tensors of higher dimension, the output does not have
    // an interpretation that is specified here.

    int nrow_a = 1;
    int ncol_a = 1;
    int nrow_b = 1;
    int ncol_b = 1;
	
    while (dimension < B.dimension || dimension % 2) {
	sizes.push_back(1);
	++dimension;
    }
    while (dimension > B.dimension) {
	B.sizes.push_back(1);
	++B.dimension;
    }

    for (int i = 0; i < dimension/2; ++i) {
	nrow_a *= sizes[2*i];
	ncol_a *= sizes[2*i+1];
	    
	nrow_b *= B.sizes[2*i];
	ncol_b *= B.sizes[2*i+1];
    }
	
    // check for compatibility
    if (ncol_a != nrow_b || ncol_a != nrow_a) {
	printf("\nERROR: cannot solve tensors [");
	for (int i = 0; i < dimension; ++i)
	    printf("%d,", sizes[i]);
	printf("\b] \\ [");
	for (int i = 0; i < B.dimension; ++i)
	    printf("%d,", B.sizes[i]);
	printf("\b].\n\n");
	abort();
    }

    // prepare to enter matrix routine
    this->FlattenPartial();
    B.FlattenPartial();

    // lapack routine needs this
    int ipiv[nrow_a];
	
    // solve linear equations
    LAPACKE_dgesv(LAPACK_ROW_MAJOR, nrow_a, ncol_b,
		  &X[0], ncol_a, ipiv, &B.X[0], ncol_b);
	
    // undo changes to B
    B.Unflatten();
    std::swap(*this, B);
    return *this;
}

Tensor& Tensor::operator/=
(Tensor B)
{
    // Solves for C in the system of linear equations:
    //     A*C = B,
    // i.e. the output is equal to A\B.

    // The intention is only for 2D tensors (matrices); for
    // tensors of higher dimension, the output does not have
    // an interpretation that is specified here.

    int nrow_a = 1;
    int ncol_a = 1;
    int nrow_b = 1;
    int ncol_b = 1;
	
    while (dimension < B.dimension || dimension % 2) {
	sizes.push_back(1);
	++dimension;
    }
    while (dimension > B.dimension) {
	B.sizes.push_back(1);
	++B.dimension;
    }

    for (int i = 0; i < dimension/2; ++i) {
	nrow_a *= sizes[2*i];
	ncol_a *= sizes[2*i+1];
	    
	nrow_b *= B.sizes[2*i];
	ncol_b *= B.sizes[2*i+1];
    }
	
    // check for compatibility
    if (ncol_a != nrow_b || ncol_b != nrow_b) {
	printf("\nERROR: cannot solve tensors [");
	for (int i = 0; i < dimension; ++i)
	    printf("%d,", sizes[i]);
	printf("\b] / [");
	for (int i = 0; i < B.dimension; ++i)
	    printf("%d,", B.sizes[i]);
	printf("\b].\n\n");
	abort();
    }

    // prepare to enter matrix routine
    this->FlattenPartial();
    B.FlattenPartial();

    // lapack routine needs this
    int ipiv[ncol_b];
	
    // solve linear equations
    LAPACKE_dgesv(LAPACK_COL_MAJOR, nrow_b, nrow_a,
		  &B.X[0], ncol_b, ipiv, &X[0], ncol_a);
	
    this->Unflatten();
    return *this;
}

Tensor& Tensor::operator*=
(Tensor B)
{
    int lda = 1; // leading dimension of A
    int ldb = 1; // leading dimension of B
    int ldc = 1; // leading dimension of C
    int m   = 1; // # rows of A & C
    int n   = 1; // # cols of B & C
    int k   = 1; // # cols of A & rows of C

    while (dimension < B.dimension || dimension % 2) {
	sizes.push_back(1);
	++dimension;
    }
    while (dimension > B.dimension) {
	B.sizes.push_back(1);
	++B.dimension;
    }

    intvec c_sizes(dimension,0);
    for (int i = 0; i < dimension/2; ++i) {
	// check that dimensions are compatible
	if (sizes[2*i+1] != B.sizes[2*i]) {
	    printf("\nERROR: cannot multiply tensors [");
	    for (int j = 0; j < dimension; ++j)
		printf("%d,", sizes[j]);
	    printf("\b] × [");
	    for (int j = 0; j < B.dimension; ++j)
		printf("%d,", B.sizes[j]);
	    printf("\b].\n\n");
	    abort();
	}

	// store sizes for the output tensor
	c_sizes[2*i+0] =   sizes[2*i+0];
	c_sizes[2*i+1] = B.sizes[2*i+1];

	// store some sizes
	lda *= sizes[2*i+1];
	ldb *= B.sizes[2*i+1];
	m *= sizes[2*i+0];
	n *= B.sizes[2*i+1];
	k *= sizes[2*i+1];
    }
    ldc = ldb;
	
    // create and output tensor
    Tensor C(c_sizes);

    // flatten tensors in preparation
    this->FlattenPartial();
    B.FlattenPartial();
    C.FlattenPartial();

    // perform matrix multiplication
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
		1.0, &X[0], lda, &B.X[0], ldb, 0.0, &C.X[0], ldc);

    // finish things out
    C.Unflatten();
    std::swap(*this, C);
    return *this;
}

Tensor& Tensor::operator^=
(const intvec& P)
{
    this->Permute(P);
    return *this;
}

Tensor& Tensor::operator~
()
{
    this->FlattenFull();
    return *this;
}

Tensor& Tensor::operator*=
(double d)
{
    for (int i = 0; i < length; ++i)
	X[i] *= d;
    return *this;
}

Tensor& Tensor::operator<<=
(const Tensor& B)
{
    // we're going to make something with double the dimensionality
    int d = dimension > B.dimension ? dimension : B.dimension;
    intvec s(2*d,0);
    for (int i = 0; i < d; ++i) {
	s[i]   = i <   dimension ?   sizes[i] : 1;
	s[i+d] = i < B.dimension ? B.sizes[i] : 1;
    }
    Tensor T(s);

    // we're looking through and performing the Kronecker calculations
    Index I(intvec(2*d,0));
    for (Index J(intvec(dimension,0)); J.status; Increment(J)) {
	for (Index K(intvec(B.dimension,0)); K.status; B.Increment(K)) {
	    T[I] = X[Address(J)] * B[K];
	    T.Increment(I);
	}
    }

    // finally, we want the tensor itself back
    std::swap(*this, T);
    return *this;
}

// post-fix
Tensor& Tensor::operator++
(int)
{
    sizes.push_back(1);
    ++dimension;
    return *this;
}

// pre-fix
Tensor& Tensor::operator++
()
{
    sizes.insert(sizes.begin(),1);
    ++dimension;
    return *this;
}

Tensor operator<<
(Tensor A, const Tensor& B)
{
    A <<= B;
    return A;
}

Tensor operator+
(Tensor A, const Tensor& B)
{
    A += B;
    return A;
}

Tensor operator-
(Tensor A, const Tensor& B)
{
    A -= B;
    return A;
}

Tensor operator*
(Tensor A, double d)
{
    A *= d;
    return A;
}

Tensor operator*
(double d, Tensor A)
{
    A *= d;
    return A;
}

Tensor operator*
(Tensor A, const Tensor& B)
{
    A *= B;
    return A;
}

Tensor operator|
(Tensor A, const Tensor& B)
{
    A |= B;
    return A;
}

Tensor operator/
(Tensor A, const Tensor& B)
{
    A /= B;
    return A;
}

Tensor operator^
(Tensor A, const intvec& P)
{
    A ^= P;
    return A;
}

void Tensor::print
(void) const
{
    auto SmartPrint =
	[](double x) -> void {
	    if (x <= -10000 || x >= 10000) {
		printf(" % 9.2f  ", x);
	    } else if (x <= -1000 || x >= 1000) {
		printf(" % 9.3f  ", x);
	    } else if (x <= -100 || x >= 100) {
		printf(" % 9.4f  ", x);
	    } else if (x <= -10 || x >= 10) {
		printf(" % 9.5f  ", x);
	    } else {
		printf(" % 9.6f  ", x);
	    }
	};

    Tensor T(*this);
    while (T.dimension < 4) {
	T.sizes.push_back(1);
	++T.dimension;
    }
    T.FlattenPartial();

    int d0 = dimension > 0 ? sizes[0] : 1;
    int d1 = dimension > 1 ? sizes[1] : 1;
    int d2 = dimension > 2 ? sizes[2] : 1;
    int d3 = dimension > 3 ? sizes[3] : 1;

    // the indexing is based on the true scenario,
    // while the access goes to a twisted version.
    Index I(intvec(T.dimension,0));

    printf("┌");
    for (int k = 0; k < d1; ++k) {
	for (int l = 0; l < d3; ++l) {
	    printf("────────────");
	}
	printf("\b┬");
    }
    printf("\b┐\n");
    for (int i = 0; i < d0; ++i) {
	if (i > 0) {
	    printf("├");
	    for (int k = 0; k < d1; ++k) {
		for (int l = 0; l < d3; ++l) {
		    printf("────────────");
		}
		printf("\b┼");
	    }
	    printf("\b┤\n");
	}
	for (int j = 0; j < d2; ++j) {
	    for (int k = 0; k < d1; ++k) {
		printf("│");
		for (int l = 0; l < d3; ++l) {
		    SmartPrint(T[I]);
		    T.Increment(I);
		}
		printf("\b");
	    }
	    printf("│\n");
	}
    }
    printf("└");
    for (int k = 0; k < d1; ++k) {
	for (int l = 0; l < d3; ++l)
	    printf("────────────");
	printf("\b┴");
    }
    printf("\b┘\n");
}

