
// template for pert/proj ?
class Solution {

};

class Projection {

    // function to be approximating
    void (*ObjectiveFunction)(double*, double*);

    // method for evaluating approximation
    void operator()(double*, double*) const;

    // a constructor for instantiating an approximation
    Projection(int dimension, int q_level);

    void SetCoefficients
};
