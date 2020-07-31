
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <set>
#include <algorithm> // std::swap


#define PI 3.1415926535




// Chebyshev level: really just an integer
class CLevel {
public:
    int level;
    CLevel(int l) : level(l) {}
    int operator()(void) const { return level; }
    CLevel& operator-=(const int& rhs) {
	level -= rhs;
	return *this;
    }
};
inline CLevel operator-(CLevel lhs, int rhs) {
  lhs -= rhs;
  return lhs;
}

double ChebyPoly(CLevel n, double x) {
    if (n.level == 0)
	return 1.0;
    else if (n.level == 1)
	return x;
    else
	return 2.0*x*ChebyPoly(n-1, x) - ChebyPoly(n-2, x);
}

// Level of Smolyak approximation
class SLevel {
public:
    int level;
    SLevel(int l) : level(l) {}
    operator CLevel() const {
	return CLevel(1 << level);
    }
};


///*

class Atom {
public:
    CLevel level;
    CLevel context;
    Atom(CLevel l_, CLevel c_) : level{l_}, context{c_} {}

    double Extremum(void) {
	int k = level();
	int n = context();
	return context() ? cos((double)k * PI / n) : 0.0;
    }
    
    double operator()(double x) {
	return ChebyPoly(level, x);
    }
};
inline bool operator<(const Atom& lhs, const Atom& rhs) {
    int lhs_num = lhs.context() ? lhs.level()   : 1;
    int lhs_den = lhs.context() ? lhs.context() : 2;
    int rhs_num = rhs.context() ? rhs.level()   : 1;
    int rhs_den = rhs.context() ? rhs.context() : 2;
    return lhs_num * rhs_den < rhs_num * lhs_den;
}




// this is either a Point or a multivariate polynomial
class Molecule : public std::vector<Atom> {
public:
    Molecule(CLevel l_, CLevel c_) :
	std::vector<Atom>(1,Atom(l_,c_)) {}
    
    Molecule& operator<<=(const Molecule& rhs) {
	for (const auto& atom_i : rhs)
	    this->push_back(atom_i);
	return *this;
    }
    double operator()(double* x) {
	double y = 1.0;
	for (size_t i = 0; i < this->size(); ++i)
	    y *= this->operator[](i)(x[i]);
	return y;
    }
};
inline Molecule operator<<(Molecule lhs, const Molecule& rhs) {
    lhs <<= rhs;
  return lhs;
}
inline bool operator< (const Molecule& lhs, const Molecule& rhs) {
    for (size_t i = 0; i < lhs.size(); ++i) {
	if (lhs[i] < rhs[i])
	    return 1;
	if (rhs[i] < lhs[i])
	    return 0;
    }
    return 0;
}




// Coefficients are only relevant in the context of grids.
class Cell : public std::set<Molecule> {
    // Note that Molecule should know how to implicitly convert
    // from integer to CLevel
    Cell(void) : std::set<Molecule>() {}
    Cell(CLevel c) {
	for (int i = 0; i <= c.level; ++i)
	    this->insert(Molecule(i,c));
    }
    Cell& operator*=(const Cell& rhs) {
	Cell out;
	for (const auto& old_mol : *this)
	    for (const auto& new_mol : rhs)
		out.insert(old_mol << new_mol);
	std::swap(*this, out);
	return *this;
    }
    Cell(const std::vector<SLevel>& i_vec) : Cell(i_vec[0]) {
	for (size_t i = 1; i < i_vec.size(); ++i)
	    *this *= Cell(i_vec[i]);
    }
};


//class Approximation {
//    Approximation(int,int,FunctionMapping);    
//}


//typedef PointGrid Cell<int>
//typedef FunctionGrid Cell<std::pair<double,double>>

//=======================================================================*/

int MyAdd(int x, int y) {
    return x + y;
}

int MySubtract(int x, int y) {
    return x - y;
}

int MyAbstract(int (*)(int, int), int, int);

namespace maryland {
    class bird {
    public:
	int x;
	bird(int x_) { x = x_; }
	void disp(void) { printf("Maryland! %d\n", x); }
    };
}
namespace maine {
    class bird {
    public:
	int x;
	bird(int x_) { x = x_; }
	void disp(void) { printf("Maine! %d\n", x); }
	operator maryland::bird() { return maryland::bird(x == 3 ? 0 : x); }
    };
}

class X {
public:
    int x;
    //X(int x_) { x = x_; }
    //X(int x_, int y_) : X(x_) { x *= y_; }
    X(CLevel c) { x = c.level; }
    void disp(void) { printf("%d\n", x); }
};

int main(int argc, char *argv[]) {

    CLevel c(4);
    SLevel s(5);
    X x(10);

    x.disp();

    /*
    int a = MyAbstract(&MyAdd, 3, 4);

    int b = MyAbstract(&MySubtract, 76, 5);

    printf("a = %d. b = %d.\n", a, b);

    maine::bird b1(3);
    maine::bird b2(4);
    
    maryland::bird b3(5);
    maryland::bird b4 = b1;
    maryland::bird b5 = b2;

    b1.disp();
    b2.disp();
    b3.disp();
    b4.disp();
    b5.disp();

    printf("1 << 0 = %d\n", 1 << 0);
    printf("1 << 3 = %d\n", 1 << 3);
    */
    
    return 0;
}



int MyAbstract(int (*fun)(int, int), int x, int y) {
    return fun(x, y);
}

