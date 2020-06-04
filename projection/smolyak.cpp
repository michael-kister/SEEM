
#include "smolyak.h"

/***********************
 *                     *
 *  UTILITY FUNCTIONS  *
 *                     *
 ***********************/

int mfun(short int i)
{
    return i ? (int)(1 << i) : 0;
}

double nfun(short int i)
{
    return i ? (double)(1 << i) : 0.5;
}

int binomial_coefficient(int n, int k)
{
    return factorial(n) / (factorial(k) * factorial(n-k));
}

double chebyshev_type_1_recursive(int n, double x)
{
    if (n == 0)
	return 1.0;
    else if (n == 1)
	return x;
    else
	return 2.0*x*chebyshev_type_1_recursive(n-1, x) - chebyshev_type_1_recursive(n-2, x);
}

// consider factoring these...
inline void chebyshev_type_1(int n, double x, double* y)
{
    if (n == 0)
	;
    else if (n == 1)
	*y *= x;
    else if (n == 2)
	*y *= 2.0*x*x - 1.0;
    else if (n == 3)
	*y *= x*(4.0*x*x - 3.0);
    else if (n == 4)
	*y *= 8.0*x*x*x*x - 8.0*x*x + 1.0;
    else if (n == 5)
	*y *= x*(16.0*x*x*x*x - 20.0*x*x + 5.0);
    else if (n == 6)
	*y *= 32.0*x*x*x*x*x*x - 48.0*x*x*x*x + 18.0*x*x - 1.0;
    else if (n == 7)
	*y *= x*(64.0*x*x*x*x*x*x - 112.0*x*x*x*x + 56.0*x*x - 7.0);
    else if (n == 8)
	*y *= 128.0*x*x*x*x*x*x*x*x - 256.0*x*x*x*x*x*x + 160.0*x*x*x*x - 32.0*x*x + 1.0;
    else if (n == 9)
	*y *= x*(256.0*x*x*x*x*x*x*x*x - 576.0*x*x*x*x*x*x + 432.0*x*x*x*x - 120.0*x*x + 9.0);
    else if (n == 10)
	*y *= 512.0*x*x*x*x*x*x*x*x*x*x - 1280.0*x*x*x*x*x*x*x*x + 1120.0*x*x*x*x*x*x - 400.0*x*x*x*x + 50.0*x*x - 1.0;
    else if (n == 11)
	*y *= x*(1024.0*x*x*x*x*x*x*x*x*x*x - 2816.0*x*x*x*x*x*x*x*x + 2816.0*x*x*x*x*x*x - 1232.0*x*x*x*x + 220.0*x*x - 11.0);
    else if (n == 12)
	*y *= 2048.0*x*x*x*x*x*x*x*x*x*x*x*x - 6144.0*x*x*x*x*x*x*x*x*x*x + 6912.0*x*x*x*x*x*x*x*x - 3584.0*x*x*x*x*x*x + 840.0*x*x*x*x - 72.0*x*x + 1.0;
    else if (n == 13)
	*y *= x*(4096.0*x*x*x*x*x*x*x*x*x*x*x*x - 13312.0*x*x*x*x*x*x*x*x*x*x + 16640.0*x*x*x*x*x*x*x*x - 9984.0*x*x*x*x*x*x + 2912.0*x*x*x*x - 364.0*x*x + 13.0);
    else if (n == 14)
	*y *= 8192.0*x*x*x*x*x*x*x*x*x*x*x*x*x*x - 28672.0*x*x*x*x*x*x*x*x*x*x*x*x + 39424.0*x*x*x*x*x*x*x*x*x*x - 26880.0*x*x*x*x*x*x*x*x + 9408.0*x*x*x*x*x*x - 1568.0*x*x*x*x + 98.0*x*x - 1.0;
    else if (n == 15)
	*y *= x*(16384.0*x*x*x*x*x*x*x*x*x*x*x*x*x*x - 61440.0*x*x*x*x*x*x*x*x*x*x*x*x + 92160.0*x*x*x*x*x*x*x*x*x*x - 70400.0*x*x*x*x*x*x*x*x + 28800.0*x*x*x*x*x*x - 6048.0*x*x*x*x + 560.0*x*x - 15.0);
    else if (n == 16)
	*y *= 32768.0*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x - 131072.0*x*x*x*x*x*x*x*x*x*x*x*x*x*x + 212992.0*x*x*x*x*x*x*x*x*x*x*x*x - 180224.0*x*x*x*x*x*x*x*x*x*x + 84480.0*x*x*x*x*x*x*x*x - 21504.0*x*x*x*x*x*x + 2688.0*x*x*x*x - 128.0*x*x + 1.0;
    else
	*y = chebyshev_type_1_recursive(n, x);
}

inline void chebyshev_type_2(int n, double x, double* y)
{
    if (n == 0)
	;
    else if (n == 1)
	*y *= 2.0*x;
    else if (n == 2)
	*y *= (-1.0 + 2.0*x)*(1.0 + 2.0*x);
    else if (n == 3)
	*y *= 4.0*x*(-1.0 + 2.0*x*x);
    else if (n == 4)
	*y *= (-1.0 - 2.0*x + 4.0*x*x)*(-1.0 + 2.0*x + 4.0*x*x);
    else if (n == 5)
	*y *= 2.0*x*(-1.0 + 2.0*x)*(1.0 + 2.0*x)*(-3.0 + 4.0*x*x);
    else if (n == 6)
	*y *= (1.0 - 4.0*x - 4.0*x*x + 8.0*x*x*x)*(-1.0 - 4.0*x + 4.0*x*x + 8.0*x*x*x);
    else if (n == 7)
	*y *= 8.0*x*(-1.0 + 2.0*x*x)*(1.0 - 8.0*x*x + 8.0*x*x*x*x);
    else if (n == 8)
	*y *= (-1.0 + 2.0*x)*(1.0 + 2.0*x)*(-1.0 - 6.0*x + 8.0*x*x*x)*(1.0 - 6.0*x + 8.0*x*x*x);
    else
	*y = 0.0;
}

/***********************
 *                     *
 *  NODULES AND NODES  *
 *                     *
 ***********************/

double Node::extremum(void) const
{
    int p = den ? num : 1;
    int q = den ? den : 2;
    return den ? cos((double)p * PI / q) : 0.0;
}

NodeSet generate_nodes(short int i)
{
    NodeSet out;
    int m = (int)(i ? 1 << i : i);
    for (int k = 0; k <= m; k++)
	out.insert(Node(k,m,i));
    return out;
}

NodeSet generate_nodes(int m)
{
    NodeSet out;
    for (int k = 0; k <= m; k++)
	out.insert(Node(k,m));
    return out;
}

/*******************************
 *                             *
 *  SIMPLE (POINT) GRID CLASS  *
 *                             *
 *******************************/

Point& Point::operator<<=(const Node& rhs)
{
    this->push_back(rhs);
    return *this;
}

Point::Point(std::initializer_list<Node> init_list)
{
    for (const auto& i : init_list)
	this->emplace_back(i);
}
    


Point::operator ArgVec<SMOLYAK>() const
{
    ArgVec<SMOLYAK> vec;
    for (const auto& it : *this)
	vec.push_back(it.extremum());
    return vec;
}

Grid::Grid(const SmoIndex& I)
{
    NodeSet tmp = generate_nodes(I[0]);

    for (const auto& it : tmp)
	this->insert(Point(it));
	
    for (int i = 1; i < I.size(); i++)
	*this *= generate_nodes(I[i]);
}

Grid::Grid(const MonIndex& I)
{
    printf("Mon\n");
    
    NodeSet tmp = generate_nodes(I[0]);

    for (const auto& it : tmp)
	this->insert(Point(it));
	
    for (int i = 1; i < I.size(); i++)
	*this *= generate_nodes(I[i]);
}

Grid& Grid::operator*=(const NodeSet& rhs)
{
    Grid  out;
    
    // looping over elements of a set of points
    for (const auto& pt : *this)
	// looping elements in a set of rationals
	for (const auto& nd : rhs)
	    // insert the augmented point
	    out.insert(pt << nd);
    
    this->swap(out);
    return *this;
}

Grid& Grid::operator+=(const Grid& rhs)
{
    for (const auto& it : rhs)
	this->insert(it);
    return *this;
}

Point Grid::element(int index)
{
    auto it = this->begin();
    for (int i = 0; i < index; i++)
	++it;
    return *it;
}

/*****************
 *               *
 *  INDEX CLASS  *
 *               *
 *****************/

int SmoIndex::sum(void) const
{
    short int out = 0;
    for (auto& n : *this)
	out += n;
    return (int)out;
}

MonIndex& MonIndex::operator<<=(int rhs)
{
    this->push_back(rhs);
    return *this;
}

double MonIndex::operator()(const ArgVec<SMOLYAK>& X, const std::vector<int>& type) const
{
    double y = 1.0;
    int n = this->size();
    for (int i = n; i--;)
	if (type[i] == 1)
	    chebyshev_type_1((*this)[i], X[i], &y);
	else
	    chebyshev_type_2((*this)[i], X[i], &y);
    return y;
}

double MonIndex::operator()(const ArgVec<SMOLYAK>& X, int type) const
{
    double y = 1.0;
    int n = this->size();
    for (int i = n; i--;)
	if (type == 1)
	    chebyshev_type_1((*this)[i], X[i], &y);
	else
	    chebyshev_type_2((*this)[i], X[i], &y);
    return y;
}

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

/********************
 *                  *
 *  MONOMIAL CLASS  *
 *                  *
 ********************/

Monomial::Monomial(const Node& nd):
    MonIndex(nd.num), num_occur(1), coeff_total(0.0)
{
    coeff_interpolation.push_back(0.0);

    index_smolyak.push_back(SmoIndex(nd.ind));
}

Monomial& Monomial::operator<<=(const Node& rhs)
{
    MonIndex::operator<<=(rhs.num);
    
    index_smolyak.back().push_back(rhs.ind);
    
    return *this;
}

Monomial& Monomial::operator+=(const Monomial& rhs)
{
    for (int i = 0; i < rhs.num_occur; i++) {
	
	this->coeff_smolyak.push_back(rhs.coeff_smolyak[i]);
	
	this->coeff_interpolation.push_back(rhs.coeff_interpolation[i]);
	
	this->index_smolyak.push_back(rhs.index_smolyak[i]);
	
    }
    
    this->num_occur += rhs.num_occur;
    
    return *this;
}

double Monomial::operator()(const ArgVec<SMOLYAK>& X) const
{
    return coeff_total * MonIndex::operator()(X, cheb_type);
}

double Monomial::operator()(const Point& P) const
{
    return coeff_total * MonIndex::operator()(P, cheb_type);
}

void Monomial::differentiate(int i)
{
    cheb_type[i]++;
    coeff_total *= (double)(*this)[i];
    (*this)[i]--;
}

void Monomial::set_coefficients(const std::map<Point, double>& graph)
{
    int dim = this->size();

    for (int i = 0; i < dim; i++)
	cheb_type.push_back(1);

    double theta; // this is what we'll make the coefficient

    // a little lambda to help out
    auto cmi = [](int m, int i) -> double {
	return (i == 0 || i == m) ? 2.0 : 1.0;
    };
    auto CMI = [&cmi](const SmoIndex& I, const Point& P) -> double {
	double out = 1.0;
	for (int i = 0; i < I.size(); i++)
	    out *= cmi(mfun(I[i]), P[i].num);
	return out;
    };
    
    Grid pts;

    coeff_total = 0.0;
    
    // you need to try it for every possible instantiation
    for (int i = 0; i < num_occur; i++) {

	theta = 0.0;

	pts = Grid(index_smolyak[i]);
	
	// loop over the points in the grid
	for (const auto& pt : pts)

	    theta += graph.find(pt)->second * MonIndex::operator()(pt,1) / CMI(index_smolyak[i],pt);

	// scale by with 2^d
	theta *= (double)(1 << dim);

	// scale by some stuff two more times...
	for (int j = 0; j < dim; j++)
	    
	    theta /= (nfun(index_smolyak[i][j]) * cmi(mfun(index_smolyak[i][j]),(*this)[j]));

	// load it into the coeff_interpolation slot
	coeff_interpolation[i] = theta;

	// upgrade the total coefficient
	coeff_total += theta * coeff_smolyak[i];
    }
}

void Monomial::set_coefficients(const std::map<Point, double>* graph)
{
    int dim = this->size();

    for (int i = 0; i < dim; i++)
	cheb_type.push_back(1);

    double theta; // this is what we'll make the coefficient

    // a little lambda to help out
    auto cmi = [](int m, int i) -> double {
	return (i == 0 || i == m) ? 2.0 : 1.0;
    };
    auto CMI = [&cmi](const SmoIndex& I, const Point& P) -> double {
	double out = 1.0;
	for (int i = 0; i < I.size(); i++)
	    out *= cmi(mfun(I[i]), P[i].num);
	return out;
    };
    
    Grid pts;

    coeff_total = 0.0;
    
    // you need to try it for every possible instantiation
    for (int i = 0; i < num_occur; i++) {

	theta = 0.0;

	pts = Grid(index_smolyak[i]);
	
	// loop over the points in the grid
	for (const auto& pt : pts)

	    theta += graph->find(pt)->second * MonIndex::operator()(pt,1) / CMI(index_smolyak[i],pt);

	// scale by with 2^d
	theta *= (double)(1 << dim);

	// scale by some stuff two more times...
	for (int j = 0; j < dim; j++)
	    
	    theta /= (nfun(index_smolyak[i][j]) * cmi(mfun(index_smolyak[i][j]),(*this)[j]));

	// load it into the coeff_interpolation slot
	coeff_interpolation[i] = theta;

	// upgrade the total coefficient
	coeff_total += theta * coeff_smolyak[i];
    }
}


/*************************************
 *                                   *
 *  COMPLEX (FUNCTIONAL) GRID CLASS  *
 *                                   *
 *************************************/

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

/**********************
 *                    *
 *  BINARY OPERATORS  *
 *                    *
 **********************/

bool operator< (const Node& lhs, const Node& rhs)
{
    // make sure that the inequality doesn't flip w/ negatives

    int p = lhs.num;
    int q = lhs.den;

    int r = rhs.num;
    int s = rhs.den;

    if (q < 0) {
	p *= -1;
	q *= -1;
    }
    if (s < 0) {
	r *= -1;
	s *= -1;
    }

    p = q ? p : 1;
    q = q ? q : 2;

    r = s ? r : 1;
    s = s ? s : 2;
    
    return p * s < r * q;
}

bool operator< (const Point& lhs, const Point& rhs)
{
    for (int i = 0; i < lhs.size(); i++)
	if (lhs[i] < rhs[i])
	    return 1;
	else if (rhs[i] < lhs[i])
	    return 0;
    return 0;
}


bool operator< (const MonIndex& lhs, const MonIndex& rhs)
{
    for (int i = 0; i < lhs.size(); i++)
	if (lhs[i] < rhs[i])
	    return 1;
	else if (rhs[i] < lhs[i])
	    return 0;
    return 0;
}

Point operator<<(Point lhs, const Node& rhs)
{
    lhs <<= rhs;
    return lhs;
}

MonIndex operator<<(MonIndex lhs, int rhs)
{
    lhs <<= rhs;
    return lhs;
}

Monomial operator<<(Monomial lhs, const Node& rhs)
{
    lhs <<= rhs;
    return lhs;
}

/************************
 *                      *
 *  PRINTING FUNCTIONS  *
 *                      *
 ************************/

void Point::print(void) const
{
    printf("( ");
    for (const auto& it : *this)
	printf("%2d/%-2d , ", it.num, it.den);
    printf("\b\b) --> ( ");
    for (const auto& it : *this)
	printf("% 5.2f , ", it.extremum());
    printf("\b\b)\n");
}

void Grid::print(void) const
{
    printf("\n");
    for (const auto& it : *this)
	it.print();
    printf("\n");
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

void Monomial::print_receipt(double* coefficients) const
{
    *coefficients = coeff_total;
}

void Monomial::print_receipt(int* indices, double* coefficients) const
{
    *coefficients = coeff_total;
    
    for (int i = 0; i < this->size(); ++i)
	
	indices[i] = MonIndex::operator[](i);
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

void FunctionGrid::print_receipt(double* coefficients) const
{
    int i = 0;
    for (const auto& it : *this) {
	it.second.print_receipt(coefficients+i);
	++i;
    }
}

void FunctionGrid::print_receipt(int** indices, double* coefficients) const
{
    int i = 0;
    for (const auto& it : *this) {
	it.second.print_receipt(indices[i], coefficients+i);
	++i;
    }
}


/*************************
 *                       *
 *  PRODUCING THE GRIDS  *
 *                       *
 *************************/

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

void sparse_grid_recursion(FunctionGrid& G, SmoIndex& ind, int dim, int mu)
{
    int i = 0;
    
    do { // since we want it to happen at least once

	ind.push_back(i);
 
	if (ind.size() < dim) {
	    
	    sparse_grid_recursion(G, ind, dim, mu);
	
	} else {
	    
	    if (ind.sum() > mu) {

		G += FunctionGrid(ind, mu);

	    }
	}
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

/*************************
 *                       *
 *  APPROXIMATION CLASS  *
 *                       *
 *************************/

double Approximation::get_constant(void)
{
    // this is a function that returns the constant in the approximation

    return fun_grid.get_constant();
}

Approximation::Approximation(int d, int m): dim(d), mu(m), coeff_status(0)
{
    fun_grid = smolyak_function_grid(dim, mu);
}

Approximation::Approximation(int d, int m, const char* filename):
    dim(d), mu(m), coeff_status(1), fun_grid(d, m, filename)
{
}

FunctionGrid::FunctionGrid(int d, int m, const char* filename)
{
    *this = smolyak_function_grid(d, m);
    
    MonIndex mon_ind(d, 0);

    std::vector<int> cheb_type(d, 0);
    
    FILE* file = fopen(filename, "r");
    
    while (!feof(file)) {

	for (int i = 0; i < 6; ++i)

	    fscanf(file, "%d(%d)", &mon_ind[i], &cheb_type[i]);

	(this->at(mon_ind)).cheb_type = cheb_type;

	fscanf(file, "%le", &(this->at(mon_ind)).coeff_total);
    }
    
    fclose(file);

}
    

void Approximation::print(void)
{
    fun_grid.print();
}

double Approximation::operator()(const Point& P) const
{
    double out = 0.0;
    if (coeff_status == 0) {
	printf("Interpolating coefficients have not yet been set. (Returning 0.0)\n");
	return out;
    } else {
	for (const auto& it : fun_grid) {
	    out += it.second(P);
	}
	return out;
    }
}

double Approximation::operator()(const ArgVec<SMOLYAK>& X) const
{
    double out = 0.0;
    if (coeff_status == 0) {
	printf("Interpolating coefficients have not yet been set. (Returning 0.0)\n");
	return out;
    } else {
	for (const auto& it : fun_grid)
	    out += it.second(X);
	return out;
    }
}

Approximation& Approximation::operator%=(int i)
{
    this->differentiate(i);

    return *this;
}

Approximation operator%(Approximation lhs, int i)
{
    lhs %= i;

    return lhs;
}

void Approximation::differentiate(int i)
{
    if (coeff_status == 0) {
	printf("Interpolating coefficients have not yet been set. Cannot differentiate.\n");
    } else {
	fun_grid.differentiate(i);
    }
}

void Approximation::set_coefficients(const std::map<Point, double>& graph)
{
    for (auto& it : fun_grid)

	it.second.set_coefficients(graph);

    coeff_status = 1;
    
}

void Approximation::set_coefficients(const std::map<Point, double>* graph)
{
    for (auto& it : fun_grid)

	it.second.set_coefficients(graph);

    coeff_status = 1;
    
}

int Approximation::receipt_length(void) const
{
    return fun_grid.size();
}

void Approximation::print_receipt(double* coefficients) const
{
    fun_grid.print_receipt(coefficients);
}

void Approximation::print_receipt(int** indices, double* coefficients) const
{
    fun_grid.print_receipt(indices, coefficients);
}


int FunctionGrid::save(const char* filename)
{
    FILE* file = fopen(filename, "w");

    for (const auto& it : *this) {

	for (int i = 0; i < it.second.size(); ++i) {

	    fprintf(file, "%03d(%1d) ", it.second.MonIndex::operator[](i),
		    it.second.cheb_type[i]);
	}

	fprintf(file, "%+12.5e\n", it.second.coeff_total);
    }
    
    return fclose(file);
}

int Approximation::save(const char* filename)
{
    return fun_grid.save(filename);
}
