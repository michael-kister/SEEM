
#include "projection_level_1.h"


/** 
 * If the denominator is non-zero, then p represents the numerator and q
 * represents the denominator. Otherwise, we go to zero.
 * -----------------------------------------------------------------------------
 */
explicit Node::operator double() const
{
    return den ? cos((double)num * PI / den) : 0.0;
}


/** 
 * This should be implemented as if you were comparing rational numbers. If the
 * denominator is negative (which in our case it should never be...) then you
 * need to offload the negative sign onto the numerator.
 *
 * Additionally, if the denominator is zero, then the rational automatically
 * becomes 1/2.
 * -----------------------------------------------------------------------------
 */
bool operator< (const Node& lhs, const Node& rhs)
{
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



/** 
 * The most complicated part of this whole code is how we require Nodes of both
 * basis functions and of locations on a grid... A basis Node should be called
 * a basis I think...
 * -----------------------------------------------------------------------------
 */



// I don't know about these...

std::set<Node> generate_nodes(int m)
{
    NodeSet out;
    for (int k = 0; k <= m; k++)
	out.insert(Node(k,m));
    return out;
}


std::set<Basis> generate_nodes(short int i)
{
    NodeSet out;
    int m = (int)(i ? 1 << i : i);
    for (int k = 0; k <= m; k++)
	out.insert(Node(k,m,i));
    return out;
}




/** 
 * Allow for initialization using braced initializer lists, as in:
 *
 * Point P {{0,2}, {0,0}, {4,4}};
 *
 * (which would resolve to 3d coordinates: (-1.0, 0.5, 1.0)
 * 
 * This is convenient for producing points on the fly, which can come up when
 * debugging.
 * -----------------------------------------------------------------------------
 */
Point::Point(std::initializer_list<Node> init_list): std::vector<Node>()
{
    for (const auto& it : init_list)

	this->emplace_back(it);
}


/** 
 * We overload the bitshift operator for aesthetic reasons.
 * -----------------------------------------------------------------------------
 */
Point& Point::operator<<=(const Node& rhs)
{
    this->push_back(rhs);
    
    return *this;
}


/** 
 * Type conversion is simply given as a vector of converted types
 * -----------------------------------------------------------------------------
 */
explicit Point::operator ArgVec<SMOLYAK>() const
{
    ArgVec<SMOLYAK> vec;
    
    for (const auto& it : *this)
	
	vec.push_back(static_cast<double>(it));
    
    return vec;
}


/** 
 * This is comparable to alphabetic organization, where you check the first
 * letter of a word, and then the next letter, until you find one that's
 * different.
 * -----------------------------------------------------------------------------
 */
bool operator< (const Point& lhs, const Point& rhs)
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

