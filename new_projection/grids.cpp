
#include "projection_level_3.h"

/** 
 * 
 * -----------------------------------------------------------------------------
 */
Grid::Grid(const SmoIndex& I)
{
    NodeSet tmp = generate_nodes(I[0]);

    for (const auto& it : tmp)
	this->insert(Point(it));
	
    for (int i = 1; i < I.size(); i++)
	*this *= generate_nodes(I[i]);
}


/*
Grid::Grid(const MonIndex& I)
{
    NodeSet tmp = generate_nodes(I[0]);

    for (const auto& it : tmp)
	this->insert(Point(it));
	
    for (int i = 1; i < I.size(); i++)
	*this *= generate_nodes(I[i]);
}
*/

/** 
 * Adding an additional dimension to a grid, by overloading the '*' operator.
 * -----------------------------------------------------------------------------
 */
Grid& Grid::operator*=(const NodeSet& rhs)
{
    Grid out;
    
    // looping over elements of a set of points
    for (const auto& pt : *this)
	
	// looping elements in a set of rationals
	for (const auto& nd : rhs)
	    
	    // insert the augmented point
	    out.insert(pt << nd);
    
    this->swap(out);
    return *this;
}

/** 
 * Adding grids together -- I'm not sure if this is necessary?
 * -----------------------------------------------------------------------------
 */
Grid& Grid::operator+=(const Grid& rhs)
{
    for (const auto& it : rhs)
	
	this->insert(it);
    
    return *this;
}


/** 
 * Accessing an element. See if maybe this could return by reference...
 * -----------------------------------------------------------------------------
 */
Point Grid::element(int index)
{
    auto it = this->begin();
    for (int i = 0; i < index; i++)
	++it;
    return *it;
}


/** 
 * Display elements of a grid.
 * -----------------------------------------------------------------------------
 */
void Grid::print(void) const
{
    printf("\n");
    for (const auto& it : *this)
	it.print();
    printf("\n");
}






FunctionGrid::FunctionGrid(const SmoIndex& ind, int mu)
{
    int dim = ind.size();

    NodeSet tmp = generate_nodes(ind[0]);

    for (const auto& nd : tmp)
	this->insert(std::pair<MonIndex, Monomial>(MonIndex(nd.num), Monomial(nd)));

    for (int i = 1; i < dim; i++)
	*this *= generate_nodes(ind[i]);

    // set the smolyak coefficients
    int i_mag = ind.sum();
    int c_smk = pow(-1, dim+mu-i_mag) * binomial_coefficient(dim-1, dim+mu-i_mag);

    //printf("(%u,%u) ==> % d\n", ind[0], ind[1], c_smk);
    
    // set all the Smolyak coefficients
    for (auto& it : *this)
	it.second.coeff_smolyak.push_back(c_smk);
}

FunctionGrid& FunctionGrid::operator*=(const NodeSet& rhs)
{
    FunctionGrid out;

    // loop over the pairs in a map 
    for (const auto& kv : *this) {
	// loop over elements in set
	for (const auto& nd : rhs) {
	    // insert(<<) int to MonIndex, and insert(<<) a Node to the Monomial 
	    out.insert(std::pair<MonIndex, Monomial>(kv.first<<nd.num, kv.second<<nd));
	}
    }

    this->swap(out);
    return *this;
}

FunctionGrid& FunctionGrid::operator+=(const FunctionGrid& rhs)
{
    for (const auto& kv : rhs)
	if (this->count(kv.first) > 0)
	    this->at(kv.first) += kv.second;
	else
	    this->insert(kv);
    return *this;
}

void FunctionGrid::differentiate(int i)
{
    /*
    for (auto& it : *this)
	if (it.first[i] > 0)
	    it.second.differentiate(i);
	else
	    this->erase(it);
    */

    for (auto it = this->begin(), next_it = it; it != this->cend(); it = next_it) {
	
	++next_it;
	
	if (it->first[i] > 0)
	    it->second.differentiate(i);
	else
	    this->erase(it);
	
    }
}

double FunctionGrid::get_constant(void)
{
    // this is a function that returns the constant in the approximation
    MonIndex i_const = this->begin()->first;

    for (int i = 0; i < i_const.size(); i++)

	i_const[i] = 0;
    
    return this->at(i_const).coeff_total;
}



void FunctionGrid::print(void) const
{
    printf("\n");
    for (const auto& it : *this)
	it.second.print();
    printf("\n");
}

void sparse_grid_recursion(Grid& G, SmoIndex& ind, int dim, int mu)
{
    int i = 0;
    
    do { // since we want it to happen at least once

	ind.push_back(i);
	
	if (ind.size() < dim)
	    
	    sparse_grid_recursion(G, ind, dim, mu);
	
	else
	    
	    if (ind.sum() > mu)
		
		G += Grid(ind); 

	ind.pop_back();
	
    } while (++i <= mu + dim - ind.sum());
}

Grid smolyak_grid(int dim, int mu)
{
    SmoIndex ind;

    Grid out;

    for (int i = 0; i <= mu + dim; i++) {
	
	ind.push_back((char)i);
	
	sparse_grid_recursion(out, ind, dim, mu);

 	ind.pop_back();
    }

    return out;
}










// they have to be separate, or do templates...

FunctionGrid smolyak_function_grid(int dim, int mu)
{
    SmoIndex ind;

    FunctionGrid out;

    for (int i = 0; i <= mu + dim; i++) {
	
	ind.push_back((char)i);
	
	sparse_grid_recursion(out, ind, dim, mu);

 	ind.pop_back();
    }

    return out;
}

void sparse_grid_recursion(FunctionGrid& G, SmoIndex& ind, int dim, int mu)
{
    int i = 0;
    
    do { // since we want it to happen at least once

	ind.push_back(i);
 
	if (ind.size() < dim)
	    
	    sparse_grid_recursion(G, ind, dim, mu);
	
	else
	    
	    if (ind.sum() > mu)

		G += FunctionGrid(ind, mu);

	ind.pop_back();
	
    } while (++i <= mu + dim - ind.sum());
}

