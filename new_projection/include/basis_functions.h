
#ifndef __PROJECTION_LEVEL_2__
#define __PROJECTION_LEVEL_2__


/** 
 * This class stores the level of the Smolyak sparse grid in a given dimension.
 * This is necessary, since the calculation of coefficients (for a function that
 * interpolates) exists as a sum of meshes, where you must know properties of
 * each mesh.
 * -----------------------------------------------------------------------------
 */
struct SmoIndex: public std::vector<short int>
{
    SmoIndex() {}

    SmoIndex(short int i): std::vector<short int>(1, i) {}

    int sum(void) const;
};


/** 
 * This holds a list of the orders of Chebyshev polynomials.
 *
 * This functor inherits from a vector of integers which denote the order of the
 * Chebyshev polynomial. As part of its identity as a functor, it does expect
 * any inputs to operator() to be within [-1,1]. This does get betrayed
 * sometimes, namely when we integrate over functions at edge points of the
 * Smolyak grid, but this overstepping should not be extreme.
 * -----------------------------------------------------------------------------
 */
struct MonIndex: public std::vector<int>
{
    MonIndex() {}

    MonIndex(int i): std::vector<int>(1, i) {}
    
    MonIndex& operator<<=(int rhs);

    double operator()(const ArgVec<SMOLYAK>& X, const std::vector<int>& type) const;
    
    double operator()(const ArgVec<SMOLYAK>& X, int type) const;
    
    double operator()(const ArgVec<SMOLYAK>& X) const;
    
    //double operator()(const std::vector<double>& X, const std::vector<int>& type) const;
    //double operator()(const std::vector<double>& X, int type) const;    
    //double operator()(const Point& P, const std::vector<int>& type) const;
    //double operator()(const Point& P, int type) const;
};




/**
 * This is a class to be a fully-fledge contributor to an approximation, in that
 * it takes basis functions and scales them according to Smolyak's equation.
 *
 * This is essentially how we later gain efficiency when calculating function
 * interpolations, since one Monomial will replace the repetition in Smolyak's
 * algorithm.
 *
 * However, this means that when we calculate the coefficient, we must calculate
 * it for every sparse grid layer, and then add together the coefficients.
 * -----------------------------------------------------------------------------
 */
struct Monomial: public MonIndex
{
    // the number of times this specific monomial has arisen in our layering
    // of sparse grids
    int num_occur;

    // cheb_type could probably be boolean, since it's only type 1 or 2.
    std::vector<int>       cheb_type;

    // these are the values that, when calculating the interpolating function,
    // comes from Smolyak's expression
    std::vector<int>       coeff_smolyak;
    //std::vector<double>    coeff_interpolation;

    // we need to know what types of meshes are being layered, in order to
    // correctly calculate coefficients for interpolation
    std::vector<SmoIndex>  index_smolyak;

    // the total scaling of our basis function
    double coeff_total;

    Monomial(): num_occur(0), coeff_total(0.0) {}

    Monomial(const Node& nd);
    
    Monomial& operator<<=(const Node& rhs);

    double operator()(const ArgVec<SMOLYAK>& X) const;

    void differentiate(int i);
    
    void set_coefficients(const std::map<Point, double>& graph);
    
    void print(void) const;

    //Monomial& operator+=(const Monomial& rhs);
    //double operator()(const Point& P) const;
};



#endif
