// for std::swap_ranges
#include <algorithm>
// this is just a wrapper to keep separate the resource management
class PerturbationManager {
    
    int LocalFactorial(int n) {
	int f = 1;
	for (int i = 2; i <= n; ++i)
	    f *= i;
	return f;
    }
    
    int LocalBinomialCoefficient(int n, int k) {
	return LocalFactorial(n)/(LocalFactorial(k)*LocalFactorial(n-k));
    }

    void Free2D(double** X, int s1) {
	for (int i = 0; i < s1; ++i) {
	    delete[] X[i];
	}
	delete[] X;
    }

    void Free3D(double*** X, int s1, int s2) {
	for (int i = 0; i < s1; ++i) {
	    for (int j = 0; j < s2; ++j) {
		delete[] X[i][j];
	    }
	    delete[] X[i];
	}
	delete[] X;
    }
    
public:

    // specification parameters
    int num_state;
    int num_control;
    int num_shock;
    int num_param;
    int degree;
    
    // resources to be managed
    double *gx;
    double *hx;
    double *xss_xss_yss_yss;
    double *parameters;
    locint *param_loc;
    adouble *X_ad;
    adouble *Y_ad;
    double **S;
    double **adolc_tensor;
    double ***index_map;
    double ***derivatives;


    // (0) Default constructor
    PerturbationManager(int ns_, int nc_, int ne_, int np_, int d_) :
	num_state{ns_}, num_control{nc_}, num_shock{ne_}, num_param{np_}, degree{d_}
    {
	// helper scalars
	int num_variable = num_state + num_control;
	int tensor_length = LocalBinomialCoefficient(2*num_variable + degree, degree);

	// one-dimensional arrays
	gx = new double [num_control*num_state];
	hx = new double [num_state*num_state];
	parameters       = new double  [num_param];
	param_loc        = new locint  [num_param];
	xss_xss_yss_yss  = new double  [2*num_variable];
	X_ad             = new adouble [2*num_variable];
	Y_ad             = new adouble [1*num_variable];

	// seed matrix
	double **S = new double* [2*num_variable];
	for (int i = 0; i < 2*num_variable; ++i) {
	    S[i] = new double [2*num_variable](); // set them all to zero
	    S[i][i] = 1.0;
	}

	// ADOL-C initial output
	double **adolc_tensor = new double* [1*num_variable];
	for (int i = 0; i < num_variable; ++i) {
	    adolc_tensor[i] = new double [tensor_length];
	}

	// derivative mapping
	int group_size[] = {1,num_state,num_state,num_control,num_control};
	int ***index_map = new int** [5];
	for (int i = 0; i < 5; ++i) {
	    index_map[i] = new int* [5];
	    for (int j = 0; j < 5; ++j) {
		index_map[i][j] = new int [group_size[i] * group_size[j]];
	    }
	}

	// derivatives themselves
	int block_size;  
	double ***derivatives = new double** [5];
	for (int i = 0; i < 5; ++i) {
	    derivatives[i] = new double* [5];
	    for (int j = 0; j < 5; ++j) {
		block_size = group_size[i] * group_size[j];
		derivatives[i][j] = new double [num_variable * block_size];
	    }
	}
    }

    // (1) Destructor
    ~PerturbationManager() {
	delete[] gx;
	delete[] hx;
	delete[] parameters;
	delete[] param_loc;
	delete[] xss_xss_yss_yss;
	delete[] X_ad;
	delete[] Y_ad;
	Free2D(S,            2*(num_state+num_control));
	Free2D(adolc_tensor, num_state+num_control);
	Free3D(index_map,    5, 5);
	Free3D(derivatives,  5, 5);
    }

    // (2) Copy-constructor
    PerturbationManager(const PerturbationManager& x) :
	PerturbationManager(x.num_state, x.num_control, x.num_shock,
			    x.num_param, x.degree)
    {
	// helper scalars
	int num_variable = num_state + num_control;
	int tensor_length = LocalBinomialCoefficient(2*num_variable + degree, degree);
	size_t sd = sizeof(double);

	memcpy(gx,              x.gx,              sd*num_control*num_state);
	memcpy(hx,              x.hx,              sd*num_state*num_state);
	memcpy(parameters,      x.parameters,      sd*num_param);
	memcpy(param_loc,       x.param_loc,       sizeof(locint)*num_param);
	memcpy(xss_xss_yss_yss, x.xss_xss_yss_yss, sd*2*num_variable);

	// manual copying for adoubles
	for (int i = 0; i < num_variable; ++i) {
	    X_ad[i] = x.X_ad[i];
	    X_ad[2*i] = x.X_ad[2*i];
	    Y_ad[i] = x.Y_ad[i];
	}

	// seed matrix
	memcpy(S, x.S, sd*2*num_variable*2*num_variable);

	// ADOL-C initial output
	memcpy(adolc_tensor, x.adolc_tensor, sd*num_variable*tensor_length);

	// derivative mapping
	int group_size[] = {1,num_state,num_state,num_control,num_control};
	for (int i = 0; i < 5; ++i)
	    for (int j = 0; j < 5; ++j)
		memcpy(index_map[i][j], x.index_map[i][j],
		       sd*group_size[i]*group_size[j]);

	// derivatives themselves
	for (int i = 0; i < 5; ++i)
	    for (int j = 0; j < 5; ++j)
		memcpy(derivatives[i][j], x.derivatives[i][j],
		       sd*num_variable*group_size[i]*group_size[j]);

    }

    // (3.0) public friend swap
    friend void swap(PerturbationManager& x, PerturbationManager& y) {
	PerturbationManager z(x);
	x = y;
	y = z;
    }

    // (3.1) Copy-assigment operator
    PerturbationManager& operator=(PerturbationManager x) {
	swap(*this, x);
	return *this;
    }

    // (4) move-constructor
    PerturbationManager(PerturbationManager&& x) noexcept :
	PerturbationManager(0,0,0,0,0)
    {
	swap(*this, x);
    }
};
