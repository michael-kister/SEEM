#include "index.h"

/**
 *
 *------------------------------------------------------------------------------ 
 */
Index::Index
(const intvec& i) :
    intvec{i}, dimension{int(i.size())}, status{1}
{
}

/**
 *
 *------------------------------------------------------------------------------ 
 */
void Index::print0
(void) const
{
    printf("[");
    for (int i = 0; i < dimension; ++i)
	printf("%d,", this->at(i));
    printf("\b]");
}

/**
 *
 *------------------------------------------------------------------------------ 
 */
void Index::print
(void) const
{
    this->print0();
    printf("\n");
}

/**
 *
 *------------------------------------------------------------------------------ 
 */
Index Index::Permute
(const intvec& P) const
{
    Index out(*this);
    for (int i = 0; i < dimension; ++i)
	out[i] = this->at(P[i]);
    return out;
}
