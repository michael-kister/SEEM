// this has all the solution and estimation methods, but requires a
// descendant to specify the ADOL-C routine.
class Perturbation : public PerturbationManager {

    // Only called from public "Solve" method
    //void solve_gx_hx(void);
    //void solve_gxx_hxx(void);
    //void solve_gss_hss(void);
    
public:

    // the constructor essentially just calls the base class to
    // manage the necessary resources.
    Perturbation(int ns_, int nc_, int ne_, int np_, int d_) :
	PerturbationManager(ns_,nc_,ne_,np_,d_) {}
    
    // Member tensors
    Tensor derivatives[5][5];
    Tensor yss, xss, gx, hx, gxx, hxx, gss, hss;

    // virtual function because we don't know what the model looks like
    virtual void PerformAutomaticDifferentiation(void) = 0;

    // then we arrange the results from the auto-diff
    void Arrange(void);

    // then we solve the matrices using linear algebra
    void Solve(void);

    // if we then update the parameters, we redo some stuff
    void UpdateParameters(double* P);

    // we'll need to evaluate this as a functor
    void operator()(void) const;
    
};





void Arrange(void) {
	
    /***********************************************************************
     * REARRANGING DATA
     **********************************************************************/

    /*
    for (int v = 0; v < num_variable; ++v) {
	printf("f(%d) : \n", v+1);
	for (int i = 0; i <= 2*num_variable; ++i) {
	    printf("   ");
	    for (int j = 0; j <= i; ++j) {
		if (adolc_tensor[v][(i*(i+1)/2)+j] > 0.00001 ||
		    adolc_tensor[v][(i*(i+1)/2)+j] < -.00001)
		    printf("%10.7f  ", adolc_tensor[v][(i*(i+1)/2)+j]);
		else
		    printf("    -       ");
	    }
	    printf("\n");
	}
	printf("\n");
    }
    */
    
    // first we want a mapping to the indices we want
    int group_size[] = {1,num_state,num_state,num_control,num_control};
    int r, c, ii, jj;
    for (int i = 0; i < 5; ++i) {
	for (int j = 0; j < 5; ++j) {
	    for (int u = 0; u < group_size[i]; ++u) {
		r = u;
		for (int k = 0; k < i; ++k) { r += group_size[k]; }
		for (int v = 0; v < group_size[j]; ++v) {
		    c = v;
		    for (int k = 0; k < j; ++k) { c += group_size[k]; }
		    ii = (r > c) ? r : c;
		    jj = (r > c) ? c : r;
		    index_map[i][j][group_size[j]*u+v] = (ii*(ii+1)/2)+jj;
		}
	    }
	}
    }

    // then we actually want to construct our necessary arrays
    int block_size;
    for (int i = 0; i < 5; ++i) {
	for (int j = 0; j < 5; ++j) {
	    block_size = group_size[i] * group_size[j];
	    for (int u = 0; u < num_variable; ++u) {
		for (int v = 0; v < block_size; ++v) {
		    derivatives[i][j][block_size*u+v] =
			adolc_tensor[u][index_map[i][j][v]];
		}
	    }
	}
    }

    // fill the tensors with the stuff
    int nx = num_state;
    int ny = num_control;
    int widths[] = {1,nx,nx,ny,ny};
    for (int i = 0; i < 5; ++i)
	for (int j = 0; j < 5; ++j)
	    derivatives_T[i][j] = Tensor
		({num_state+num_control,widths[i],1,widths[j]},
		 derivatives[i][j]);

}





// Solve the model
void Perturbation::Solve(void) {

    /***********************************************************************
     * SOLVE GX HX
     **********************************************************************/
    //double *gx  = new double [ny * nx]();
    //double *hx  = new double [nx * nx]();
    solve_gx_hx(gx, hx, derivatives, num_control, num_state);
  
    Tensor gx_T({ny,nx},gx);
    Tensor hx_T({nx,nx},hx);
    printf("gx: (lt, ct) (measurement)\n");
    gx_T.print(); 
    printf("hx: (lt, ct) (measurement)\n");
    hx_T.print(); 

    /***********************************************************************
     * SOLVE GXX, HXX, GSS, HSS.
     **********************************************************************/
    Tensor gxx_T({ny,nx,1,nx});
    Tensor hxx_T({nx,nx,1,nx});
    solve_gxx_hxx(gxx_T, hxx_T, derivatives_T, num_control, num_state, gx_T, hx_T);
    printf("gxx: (lt, ct) (measurement)\n");
    gxx_T.print(); 
    printf("hxx: (lt, ct) (measurement)\n");
    hxx_T.print(); 

    int num_shock = 1;
    int neps = 1;
    Tensor gss_T({ny,1,1,1});
    Tensor hss_T({ny,1,1,1});
    Tensor eta_T({nx,neps});
    eta_T.X[0] = 0;
    eta_T.X[1] = 1;

    solve_gss_hss(gss_T, hss_T, derivatives_T, num_control, num_state, num_shock,
		  gx_T, gxx_T, eta_T);
    printf("gss: (lt, ct) (measurement)\n");
    gss_T.print(); 
    printf("hss: (lt, ct) (measurement)\n");
    hss_T.print(); 

}

