#ifndef __SMOLYAK__
#define __SMOLYAK__

/**
 * This is a header file for the classes pertaining to Smolyak's algorithm. The ultimate goal
 * is an approximation class, which can take an array and produce a double. In order to use
 * projection methods, we will later form a vector of approximations, which yield the policy
 * functions.
 */
#include <cmath>
#include <cstdio>

#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <initializer_list>


#include "miscellaneous.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

int mfun(short int i);

double nfun(short int i);

int binomial_coefficient(int n, int k);

//double chebyshev(int n, double x);

double chebyshev_type_1_recursive(int n, double x);

inline void chebyshev_type_1(int n, double x, double* y);

inline void chebyshev_type_2(int n, double x, double* y);


enum ArgType { UNKNOWN, NATURAL, TILDE, SMOLYAK };

template<ArgType T>
struct ArgVec: public std::vector<double>
{
    ArgVec(int n, double d): std::vector<double>(n, d) {}

    ArgVec(void) {}

};

template struct ArgVec<UNKNOWN>;
template struct ArgVec<NATURAL>;
template struct ArgVec<TILDE>;
template struct ArgVec<SMOLYAK>;


/**
 * This is a class that that encapsulates functionality of rationals?
 */
struct Node
{
    int num, den;
    
    short int ind;

    Node(int n, int d, short int i): num(n), den(d), ind(i) {}
    
    Node(int n, int d): num(n), den(d) {}
    
    double extremum(void) const;
};

typedef std::set<Node> NodeSet;

NodeSet generate_nodes(int i);

NodeSet generate_nodes_preset(int m);

/**
 * Just a vector of nodes
 */
struct Point: public std::vector<Node>
{
    Point(const Node& nod): std::vector<Node>(1, nod) {}

    Point(std::initializer_list<Node> init_list);
    
    Point& operator<<=(const Node& rhs);

    void print(void) const;
    
    //operator std::vector<double>() const;
    operator ArgVec<SMOLYAK>() const;
};

/**
 * This is a class to reference which order polynomial in each of the dimensions we are
 * examining. We make it its own class because we want to 
 */
struct SmoIndex: public std::vector<short int>
{
    SmoIndex() {}

    SmoIndex(short int i): std::vector<short int>(1, i) {}
    
    int sum(void) const;
};

struct MonIndex: public std::vector<int>
{
    MonIndex() {}

    MonIndex(int i): std::vector<int>(1, i) {}
    
    MonIndex(int i, int j): std::vector<int>(i, j) {}
    
    MonIndex& operator<<=(int rhs);

    //double operator()(const std::vector<double>& X, const std::vector<int>& type) const;
    //double operator()(const std::vector<double>& X, int type) const;
    
    double operator()(const ArgVec<SMOLYAK>& X, const std::vector<int>& type) const;
    double operator()(const ArgVec<SMOLYAK>& X, int type) const;
    
    double operator()(const Point& P, const std::vector<int>& type) const;
    double operator()(const Point& P, int type) const;
};

/**
 * This is a class to build upon the Chebyshev basis, in order to retain the information
 * about the various inceptions/motivations. If we construct a grid containing these, then
 * we want to know the MonIndex (or vector of integers), something else, and something else.
 */
struct Monomial: public MonIndex
{
    int num_occur;

    std::vector<int>       cheb_type;
    std::vector<int>       coeff_smolyak;
    std::vector<double>    coeff_interpolation;
    std::vector<SmoIndex>  index_smolyak;
    
    double coeff_total;

    Monomial(): num_occur(0), coeff_total(0.0) {}

    Monomial(const Node& nd);
    
    Monomial& operator<<=(const Node& rhs);

    Monomial& operator+=(const Monomial& rhs);

    //double operator()(const std::vector<double>& X) const;
    double operator()(const ArgVec<SMOLYAK>& X) const;

    double operator()(const Point& P) const;

    void differentiate(int i);
    
    void set_coefficients(const std::map<Point, double>& graph);
    
    void set_coefficients(const std::map<Point, double>* graph);
    
    void print(void) const;
    
    // functions for obtaining approximation outside of OOP
    void print_receipt(double* coefficients) const;

    void print_receipt(int* indices, double* coefficients) const;
};

/**
 * This is a class to hold the set of points which make up the sparse grid. This will be useful
 * when we need to know the locations at which to obtain the policy function values.
 */
struct Grid: public std::set<Point>
{
    Grid() {}
    
    Grid(const SmoIndex& I);

    Grid(const MonIndex& I);

    Grid& operator*=(const NodeSet& rhs);

    Grid& operator+=(const Grid& rhs);

    void print(void) const;

    Point element(int);
};



/**
 * This is a class to build up the list of monomials used to approximate the function;
 * a vector of indices is mapped to a particular chebyshev combination. That combination
 * may have resulted from several different grids, so carries a vector for each of them.
 *
 * Note: this extra information is carried by the Monomial, not by the derived class. At
 * what stage should this information be augmented? 
 */
struct FunctionGrid: public std::map<MonIndex, Monomial>
{
    FunctionGrid() {}
    
    FunctionGrid(const SmoIndex& ind, int mu);

    FunctionGrid(int d, int m, const char* filename);

    FunctionGrid& operator*=(const NodeSet& rhs);

    FunctionGrid& operator+=(const FunctionGrid& rhs);

    void differentiate(int i);

    double get_constant(void);
    
    void print(void) const;

    // functions for obtaining approximation outside of OOP
    void print_receipt(double* coefficients) const;

    void print_receipt(int** indices, double* coefficients) const;

    int save(const char* filename);
};




void sparse_grid_recursion(Grid& G, SmoIndex& ind, int dim, int mu);

void sparse_grid_recursion(FunctionGrid& G, SmoIndex& ind, int dim, int mu);

Grid smolyak_grid(int dim, int mu);

FunctionGrid smolyak_function_grid(int dim, int mu);




/**
 * In general, rationals are compared as would be expected. It should be noted, however, that
 * integers are not expected to be less than zero in the context of these projection methods,
 * however this is accounted for by offloading any negatives into the numerator.
 *
 * Additionally, if we have 0/0, then this is considered equivalent to 1/2, because it yields
 * the same results in terms of node location. It is also important to note that when comparing
 * degrees of polynomials, we no longer deal in terms of rationals, but in vectors of integers,
 * which have already been extracted from the rationals.
 */
bool operator< (const Node& lhs, const Node& rhs);

bool operator< (const Point& lhs, const Point& rhs);

bool operator< (const MonIndex& lhs, const MonIndex& rhs);

Point operator<<(Point lhs, const Node& rhs);

MonIndex operator<<(MonIndex lhs, int rhs);

Monomial operator<<(Monomial lhs, const Node& rhs);






/**
 * This is the main class; it will primarily act as a functor, whose coefficients must be
 * set before it can be used. (Maybe add in a switch to ensure that?)
 */
class Approximation
{
private:
    
    int dim, mu;

    short int coeff_status;

    FunctionGrid fun_grid;

public:

    Approximation(int d, int m);

    Approximation(int d, int m, const char* filename);
    
    void set_coefficients(const std::map<Point, double>& graph);

    void set_coefficients(const std::map<Point, double>* graph);

    //double operator()(const std::vector<double>& X) const;
    double operator()(const ArgVec<SMOLYAK>& X) const;

    double operator()(const Point& P) const;

    void differentiate(int i);

    Approximation& operator%=(int i);
    
    void print(void);

    double get_constant(void);

    // functions for obtaining approximation outside of OOP
    int receipt_length(void) const;
    
    void print_receipt(double* coefficients) const;

    void print_receipt(int** indices, double* coefficients) const;

    // methods for saving/loading previously calculated approximations
    int save(const char* filename);

};

Approximation operator%(Approximation lhs, int i);


#endif
