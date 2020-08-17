#ifndef INDEX_H
#define INDEX_H

typedef std::vector<int> intvec;

class Index : public intvec {
public:

    // Members
    int dimension;
    int status;

    // Constructor
    Index(const intvec& i);

    // Methods for displaying an index
    void print0(void) const;
    void print(void) const;

    // Re-order elements of an index
    Index Permute(const intvec& P) const;
};

#endif
