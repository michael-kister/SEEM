
#ifndef __PROJECTION_LEVEL_1__
#define __PROJECTION_LEVEL_1__


/** 
 * This is a class that specifies the location of an extrema of a Chebyshev
 * polynomial. (Note that zeros are often used as nodes for polynomial
 * interpolation, but we are using extrema.) The formula for the j'th extrema
 * of a Chebyshev polynomial of order n is given by cos(pi*j/n). This holds
 * unless n=0, in which case the extrema is specified as zero.
 *
 * It is specified as a rational number in order to compare non-integers
 * exactly.
 * -----------------------------------------------------------------------------
 */
struct Node
{
    int num, den;
    
    Node(int n, int d): num(n), den(d) {}

    explicit operator double() const;
};

bool operator< (const Node& lhs, const Node& rhs);



/** 
 * A Basis is similar to a Node, except it represents a basis function, 
 * -----------------------------------------------------------------------------
 */
struct Basis: public Node
{
    short int ind;

    Basis(int n, ind d, short int i): Node(n, d), ind(i) {}    
};





/** 
 * NodeSet is... TODO
 * -----------------------------------------------------------------------------
 */
typedef std::set<Node> NodeSet;

NodeSet generate_nodes(int i);

NodeSet generate_nodes_preset(int m);


/** 
 * A Point is an ordered set (vector) of Nodes, that gives a point in an
 * n-dimensional space. The bitshift operator is used to insert elements into
 * the vector.
 * -----------------------------------------------------------------------------
 */
struct Point: public std::vector<Node>
{
    Point(const Node& nod): std::vector<Node>(1, nod) {}

    Point(std::initializer_list<Node> init_list);
    
    Point& operator<<=(const Node& rhs);

    Point& operator<<=(const Node&& rhs);

    void print(void) const;
    
    explicit operator ArgVec<SMOLYAK>() const;
};

bool operator< (const Point& lhs, const Point& rhs);

Point operator<<(Point lhs, const Node& rhs);





#endif
