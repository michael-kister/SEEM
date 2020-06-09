
#include "projection_level_2.h"

/*
double MonIndex::operator()(const Point& P, const std::vector<int>& type) const
{
    double y = 1.0;
    for (int i = 0; i < this->size(); i++) {
	if (type[i] == 1)
	    chebyshev_type_1((*this)[i], P[i].extremum(), &y);
	else
	    chebyshev_type_2((*this)[i], P[i].extremum(), &y);
    }
    return y;
}

double MonIndex::operator()(const Point& P, int type) const
{
    double y = 1.0;
    for (int i = 0; i < this->size(); i++) {
	if (type == 1)
	    chebyshev_type_1((*this)[i], P[i].extremum(), &y);
	else
	    chebyshev_type_2((*this)[i], P[i].extremum(), &y);
    }
    return y;
}

double Monomial::operator()(const Point& P) const
{
    return coeff_total * MonIndex::operator()(P, cheb_type);
}
*/
/** 
 * This appears to be a way to add together monomials... I'm not sure if that is
 * ever necessary?
 * -----------------------------------------------------------------------------
Monomial& Monomial::operator+=(const Monomial& rhs)
{
    for (int i = 0; i < rhs.num_occur; i++) {
	
	this->coeff_smolyak.emplace_back(rhs.coeff_smolyak[i]);
	
	this->index_smolyak.emplace_back(rhs.index_smolyak[i]);
    }
    
    this->num_occur += rhs.num_occur;
    
    return *this;
}
 */



/** 
 * We use the sum to determine whether a sparse mesh is viable for our purposes.
 * -----------------------------------------------------------------------------
 */
int SmoIndex::sum(void) const
{
    short int out = 0;
    for (auto& n : *this)
	out += n;
    return (int)out;
}











/** 
 * Use bitshift operator to insert elements into our vector of monomial indices.
 * -----------------------------------------------------------------------------
 */
MonIndex& MonIndex::operator<<=(int rhs)
{
    this->emplace_back(rhs);
    return *this;
}

/** 
 * Since this is a functor, its main duty is to evaluate Chebyshev polynomials,
 * which is done here. The type of the chebyshev polynomial is given as a vector
 * -----------------------------------------------------------------------------
 */
double MonIndex::operator()(const ArgVec<SMOLYAK>& X, const std::vector<int>& type) const
{
    double y = 1.0;
    int n = this->size();
    for (int i = n; i--;)
	if (type[i] == 1)
	    y *= chebyshev_type_1((*this)[i], X[i]);
	else
	    y *= chebyshev_type_2((*this)[i], X[i]);
    return y;
}

/** 
 * Since this is a functor, its main duty is to evaluate Chebyshev polynomials,
 * which is done here. The type of the chebyshev polynomial is given as a single
 * number.
 * -----------------------------------------------------------------------------
 */
double MonIndex::operator()(const ArgVec<SMOLYAK>& X, int type) const
{
    double y = 1.0;
    int n = this->size();
    for (int i = n; i--;)
	if (type == 1)
	    y *= chebyshev_type_1((*this)[i], X[i]);
	else
	    y *= chebyshev_type_2((*this)[i], X[i]);
    return y;
}


/** 
 * Since this is a functor, its main duty is to evaluate Chebyshev polynomials,
 * which is done here. The type of the chebyshev polynomial is assumed to be 1.
 * -----------------------------------------------------------------------------
 */
double MonIndex::operator()(const ArgVec<SMOLYAK>& X) const
{
    double y = 1.0;
    int n = this->size();
    for (int i = n; i--;)
	y *= chebyshev_type_1((*this)[i], X[i]);
    return y;
}


MonIndex operator<<(MonIndex lhs, int rhs)
{
    lhs <<= rhs;
    return lhs;
}













/** 
 * A Monomial is constructed using a Node. The num(erator) member of the Node
 * serves to tell us what 
 * -----------------------------------------------------------------------------
 */
Monomial::Monomial(const Node& node):
    MonIndex(node.num), num_occur(1), coeff_total(0.0)
{
    // here it doesn't really matter that we use emplace_back
    coeff_interpolation.emplace_back(0.0);

    // here it actually matters that we use emplace_back
    index_smolyak.emplace_back(SmoIndex(node.ind));
}


/** 
 * I don't understand the second step here...
 * -----------------------------------------------------------------------------
 */
Monomial& Monomial::operator<<=(const Node& rhs)
{
    MonIndex::operator<<=(rhs.num);
    
    index_smolyak.back().emplace_back(rhs.ind);
    
    return *this;
}


/** 
 * When calling a Monomial instance as a functor, we refer to the base class
 * operator, and then scale this by the total coefficient (which is the sum of
 * all the coefficients from different layers of sparse grids).
 * -----------------------------------------------------------------------------
 */
double Monomial::operator()(const ArgVec<SMOLYAK>& X) const
{
    return coeff_total * MonIndex::operator()(X, cheb_type);
}

/** 
 * This is a very incomplete implementation of differentiating Chebyshev
 * polynomials. Basically, if it's a Chebyshev polynomial of the first kind, and
 * the order is greater than zero, then we use the simple relation:
 *
 * T_{n}(x)' = n U_{n-1}(x).
 * -----------------------------------------------------------------------------
 */
void Monomial::differentiate(int i)
{
    if (cheb_type[i] == 1) {
	if ((*this)[i] > 0) {
	    // then we can differentiate
	    cheb_type[i]++;
	    coeff_total *= (double)(*this)[i];
	    (*this)[i]--;
	} else {
	    // we don't know how to differentiate
	    fprintf(stderr, "You asked to differentiate variable [%d], but I cannot,\n", i);
	    fprintf(stderr, "since the polynomial level is only [%d].\n", (*this)[i]);
	}
    } else {
	// we don't know how to differentiate
	fprintf(stderr, "You asked to differentiate variable [%d], but I cannot,\n", i);
	fprintf(stderr, "since it is already of Chebyshev form 2.\n");
    }
}


/** @brief Calculates projection of target function onto one basis function.
 *         
 * Obviously the most complicated method; for details beyond simply how this is
 * implemented, see Malin et al. 2011. Note that because a single instantiation
 * of Monomial often includes references to the given basis function from more
 * than one layer, we must calculate the sum of coefficients.
 * The calculation of coefficients for a single layer is broken into 3 parts,
 * corresponding to parts of Eq. 25 in Malin.
 *
 * @param graph A mapping from points (at nodes) to values achieved by the
 *        targeted function.
 * -----------------------------------------------------------------------------
 */
void Monomial::set_coefficients(const std::map<Point, double>& graph)
{
    int dim = this->size();

    // not necessary for setting coefficients, but now is the time to do this.
    for (int i = 0; i < dim; i++)
	cheb_type.emplace_back(1);

    // our coefficient for each layer (re-use this variable)
    double theta; 

    // a little lambda
    auto cmi = [](int m, int i) -> double {
	return (i == 0 || i == m) ? 2.0 : 1.0;
    };

    // a bigger lambda
    auto CMI = [&cmi](const SmoIndex& I, const Point& P) -> double {
	double out = 1.0;
	for (int i = 0; i < I.size(); i++)
	    out *= cmi(mfun(I[i]), P[i].num);
	return out;
    };

    // grid that gets resized for each layer
    Grid pts;

    // running sum of contributions from different layers
    coeff_total = 0.0;
    
    // loop over all the layers
    for (int i = 0; i < num_occur; i++) {

	// reset theta to start fresh
	theta = 0.0;

	// define the points that are relevant to this layer
	pts = Grid(index_smolyak[i]);
	
	// loop over the points in the grid
	for (const auto& pt : pts)

	    // (1)
	    // mapping of target function times mapping of basis function,
	    // scaled by a term that depends on whether the point is at the
	    // edge of the sparse grid
	    theta += graph.find(pt)->second * MonIndex::operator()(pt) / CMI(index_smolyak[i],pt);

	// (2)
	// scale by with 2^d
	theta *= (double)(1 << dim);
	
	// scale by two more terms for each dimension
	for (int j = 0; j < dim; j++)

	    // (3)
	    // scaling by number of points and whether this basis function
	    // is at the edge of the set determined by this layer.
	    theta /= (nfun(index_smolyak[i][j]) * cmi(mfun(index_smolyak[i][j]),(*this)[j]));

	// upgrade the total coefficient
	coeff_total += theta * coeff_smolyak[i];
    }
}

Monomial operator<<(Monomial lhs, const Node& rhs)
{
    lhs <<= rhs;
    return lhs;
}



void Monomial::print(void) const
{
    int i = 0;
    printf("T[");
    for (const auto& it : *this)
	if (this->cheb_type[i++] == 1)
	    printf("%2d,", it);
	else
	    printf("*%1d,", it);
    printf("\b](x) : % .3e   <", coeff_total);
    for (const auto& it : index_smolyak) {
	for (const auto& jt : it)
	    printf("%d,", jt);
	printf("\b>, <");
    }
    printf("\b\b\b\n");
}
