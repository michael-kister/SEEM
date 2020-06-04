
#include "include/simulated_annealing.h"

int val_converged
(const std::deque<double>& val_check, double fp, double tol)
{
    for (int i = 0; i < val_check.size(); i++)

	if (fabs(fp - val_check[i]) > tol)

	    return 0;

    return 1;
}

int simulated_anneal
(std::function<double(const double*)> &fun, int dim, double* x_out,
 const double* lb, const double* ub, int verbose)
{
    
    if (verbose >= 1) {

	for (int i = 0; i < 80; i++) printf("=");
	printf("\n");
	printf("                      SIMULATED ANNEALING ROUTINE\n\n");
    }

    double T0 = 1.0;
    
    // constants that define the optimization algorithm
    int num_cycle = 20;
    int num_temp  = 100;

    int num_temp_iter = (100 > 5*dim) ? 100 : 5*dim;
    
    int max_eval  = 1 << 20;
    int num_eval  = 0;

    int num_eps = 4;
    std::deque<double> val_check(num_eps, 0.0);
    double tol = 1.0e-10;

    double temp_reduc = 0.85; // by how much we reduce the temperature
    double step_adj  = 2.0;
    double ratio;
    
    // P-RNG (obviously not a great generator, but this isn't the focus)
    srand(9224);
    
    double f = fun(x_out); // initial value
    ++num_eval;

    val_check.pop_front(); val_check.push_back(f);

    double fp;     // proposed value
    double fb = f; // best value

    // array with number of successes
    int* num_accept = new int [dim];
    
    // step size array
    double* vm = new double [dim]();

    for (int i = 0; i < dim; i++)

	vm[i] = (ub[i] - lb[i]) / 3.0;

    // initialization
    double* x  = new double [dim]();

    double* xp = new double [dim]();

    for (int i = 0; i < dim; i++)
	
	xp[i] = x[i] = x_out[i];

    // the main thing we do is decrease the temperature
    for (double T = T0; ; T *= temp_reduc) {

	for (int i_temp = 0; i_temp < num_temp_iter; i_temp++) {

	    if ((verbose > 0 && i_temp == 0) || (verbose > 1)) {

		printf("\nTemperature......... %12.10f\n", T);
		printf("Total evaluations... %d\n", num_eval);
		printf("Minimum value....... %12.10f\n", fb);
		printf("Most recent value... %12.10f\n", f);
		for (int i = 0; i < 80; i++) printf("-");
		printf("\n");	    
	    }
	    
	    for (int i = 0; i < dim; i++)

		num_accept[i] = 0;
	    
	    // after num_cycle loops over the whole vector, we adjust step_length
	    for (int i_cyc = 0; i_cyc < num_cycle; i_cyc++) {

		// sample a point for each dimension of the vector
		for (int i_dim = 0; i_dim < dim; i_dim++) {

		    xp[i_dim] += UNIF_RAND_2 * vm[i_dim];
		    
		    if (xp[i_dim] < lb[i_dim] || xp[i_dim] > ub[i_dim])
			xp[i_dim] = lb[i_dim] + UNIF_RAND * (ub[i_dim] - lb[i_dim]);
		    fp = fun(xp);
		    ++num_eval;

		    if (verbose > 3) {
			printf("Attempt: ");
			for (int i = 0; i < dim; i++)
			    printf(" %+9.4f", xp[i]);
			printf(" --> (%+9.2e) ", fp);			
		    }

		    if ((fp < f) || (exp((f-fp)/T) > UNIF_RAND)) {		    

			if (verbose > 3) {
			    if (fp < f) {
				if (fp < fb) printf("***");
				else printf("**");
			    } else {
				printf("*");
			    }
			}

			x[i_dim] = xp[i_dim];
			f = fp;
			num_accept[i_dim]++;

			if (f < fb) {
			    fb = f;
			    for (int i = 0; i < dim; i++)
				x_out[i] = x[i];
			}
			if (num_eval > num_eps && val_converged(val_check, fp, tol)) {
			    if (fabs(f - fp) < tol) {
				delete[] vm;
				delete[] x;
				delete[] xp;
				goto SUCCESS;
			    } else {
				for (int i = 0; i < dim; i++)
				    x[i] = x_out[i];
				f = fb;
			    }
			}
			val_check.pop_front(); val_check.push_back(f);
		    } else {
			xp[i_dim] = x[i_dim];
		    }
		    if (verbose > 3) printf("\n");
		    if (num_eval > max_eval) {
			delete[] vm;
			delete[] x;
			delete[] xp;
			goto EVAL_LIMIT;
		    }
		} // looping over dimensions

		/*
		if (verbose > 2) {

		    printf(" Optimum:");
		    for (int i = 0; i < dim; i++)
			printf(" %+9.4f", x_out[i]);
		    printf(" ----------------> (%+9.2e)\n", fb);
		}
		*/
	    
	    } // loops with the same vm

	    if (verbose > 1) {

		printf("Acceptance ratios:");
		for (int i = 0; i < dim; i++)
		    printf("   %03d/%03d", num_accept[i], num_cycle);
		printf("\n");
		printf("Jump sizes:       ");
		for (int i = 0; i < dim; i++)
		    printf("   %7.5f", vm[i]);
		printf("\n");		
	    }

	    for (int i = 0; i < dim; i++) {

		ratio = (double)num_accept[i] / num_cycle;
		
		if (ratio > 0.6)

		    vm[i] *= (1.0 + (step_adj * (ratio - 0.6) / 0.4));

		else if (ratio < 0.4)

		    vm[i] /= (1.0 + (step_adj * (0.4 - ratio) / 0.4));
		
		if (vm[i] > ub[i] - lb[i])

		    vm[i] = 0.9 * (ub[i] - lb[i]);
	    }

	    if (verbose > 1) {

		printf("Jump sizes:       ");
		for (int i = 0; i < dim; i++)
		    printf("   %7.5f", vm[i]);
		printf("\n");		
	    }

	} // loops with the same temperature

    } // looping to decrease temperature

    // this really shouldn't happen, but it means that the temperature reached
    // zero without achieving convergence.
    return 99;
    
    // if we reached the maximum number of iterations
    SUCCESS:
    {
	if (verbose > 3) printf("\n");

	printf("%f %f %f %f \n", val_check[0], val_check[2], val_check[3], val_check[4]);
		    
	return 0;
    } 

    // if we reached the maximum number of iterations
    EVAL_LIMIT:
    {
	return 1;
    } 
}
