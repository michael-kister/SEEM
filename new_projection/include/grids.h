
#ifndef __PROJECTION_LEVEL_3__
#define __PROJECTION_LEVEL_3__

/** 
 * This is a class to hold the set of points which make up the sparse grid. This
 * will be useful when we need to know the locations at which to obtain the
 * policy function values.
 * -----------------------------------------------------------------------------
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
 * This is a class to build up the list of monomials used to approximate the
 * function; a vector of indices is mapped to a particular Chebyshev basis. That
 * basis may have resulted from several different grids, so carries a vector for
 * each of them.
 *
 * Note: this extra information is carried by the Monomial, not by the derived
 * class. At what stage should this information be augmented? 
 * -----------------------------------------------------------------------------
 */

// we need some way of accessing the elements? Although I'm not sure why you
// couldn't just do a set...

struct FunctionGrid: public std::map<MonIndex, Monomial>
{
    FunctionGrid() {}
    
    FunctionGrid(const SmoIndex& ind, int mu);

    FunctionGrid& operator*=(const NodeSet& rhs);

    FunctionGrid& operator+=(const FunctionGrid& rhs);

    void differentiate(int i);

    double get_constant(void);
    
    void print(void) const;
};




void sparse_grid_recursion(Grid& G, SmoIndex& ind, int dim, int mu);

void sparse_grid_recursion(FunctionGrid& G, SmoIndex& ind, int dim, int mu);

Grid smolyak_grid(int dim, int mu);

FunctionGrid smolyak_function_grid(int dim, int mu);





#endif
