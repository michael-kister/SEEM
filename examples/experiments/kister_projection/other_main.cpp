    // START THE NEW APPROXIMATION
    /*
    G = smolyak_grid(dim, ++mu);
    num_collocation = G.size();
    
    Approximation Vfun2(dim, mu);
    Approximation ifun2(dim, mu);
    Approximation lfun2(dim, mu);
    Approximation Pfun2(dim, mu);
    Approximation xfun2(dim, mu);

    coeffs_V1 = new double[num_collocation];
    coeffs_i1 = new double[num_collocation];
    coeffs_l1 = new double[num_collocation];
    coeffs_P1 = new double[num_collocation];
    coeffs_x1 = new double[num_collocation];
    
    coeffs_V2 = new double[num_collocation];
    coeffs_i2 = new double[num_collocation];
    coeffs_l2 = new double[num_collocation];
    coeffs_P2 = new double[num_collocation];
    coeffs_x2 = new double[num_collocation];

    order_v = new int[num_collocation];
    order_R = new int[num_collocation];
    order_k = new int[num_collocation];
    order_A = new int[num_collocation];
    order_sg = new int[num_collocation];
    order_xi = new int[num_collocation];
    
    for (int i = 0; i < max_proj_iter; i++) {
	/*
	printf("bounds[[%d]] <- list(", ib);
	for (int ii = 0; ii < 5; ii++)
	    printf("c(%8.4f,%8.4f), ", lb_i[ii], ub_i[ii]);
	printf("\b\b)\n");
	*//*
	for (const auto& it : G) {

	    st = static_cast<std::vector<double>>(it);
		
	    if (i == 0) {
		
		Vfun_graph.insert(std::pair<Point,double>(it,Vfun(st)));
		ifun_graph.insert(std::pair<Point,double>(it,ifun(st)));
		lfun_graph.insert(std::pair<Point,double>(it,lfun(st)));
		Pfun_graph.insert(std::pair<Point,double>(it,Pfun(st)));
		xfun_graph.insert(std::pair<Point,double>(it,xfun(st)));

	    } else {

		it.print();
		
		yt[0] = Vfun2(st);
		yt[1] = ifun2(st);
		yt[2] = lfun2(st);
		yt[3] = Pfun2(st);
		yt[4] = xfun2(st);
		
		solve_at_point(st, yt, Vfun2, ifun2, lfun2, Pfun2, xfun2, lb_i, ub_i, lb_i, ub_i, 1, 1.0e-13);
		
		Vfun_graph.insert(std::pair<Point,double>(it,yt[0]));
		ifun_graph.insert(std::pair<Point,double>(it,yt[1]));
		lfun_graph.insert(std::pair<Point,double>(it,yt[2]));
		Pfun_graph.insert(std::pair<Point,double>(it,yt[3]));
		xfun_graph.insert(std::pair<Point,double>(it,yt[4]));
	    }
	}
	Vfun2.set_coefficients(Vfun_graph);
	ifun2.set_coefficients(ifun_graph);
	lfun2.set_coefficients(lfun_graph);
	Pfun2.set_coefficients(Pfun_graph);
	xfun2.set_coefficients(xfun_graph);
	
	Vfun_graph.clear();
	ifun_graph.clear();
	lfun_graph.clear();
	Pfun_graph.clear();
	xfun_graph.clear();
	
	if (i % 2 == 0) {
	    Vfun2.print_receipt(coeffs_V1, order_v, order_R, order_k, order_A, order_sg, order_xi);
	    ifun2.print_receipt(coeffs_i1, order_v, order_R, order_k, order_A, order_sg, order_xi);
	    lfun2.print_receipt(coeffs_l1, order_v, order_R, order_k, order_A, order_sg, order_xi);
	    Pfun2.print_receipt(coeffs_P1, order_v, order_R, order_k, order_A, order_sg, order_xi);
	    xfun2.print_receipt(coeffs_x1, order_v, order_R, order_k, order_A, order_sg, order_xi);
	} else {
	    Vfun2.print_receipt(coeffs_V2, order_v, order_R, order_k, order_A, order_sg, order_xi);
	    ifun2.print_receipt(coeffs_i2, order_v, order_R, order_k, order_A, order_sg, order_xi);
	    lfun2.print_receipt(coeffs_l2, order_v, order_R, order_k, order_A, order_sg, order_xi);
	    Pfun2.print_receipt(coeffs_P2, order_v, order_R, order_k, order_A, order_sg, order_xi);
	    xfun2.print_receipt(coeffs_x2, order_v, order_R, order_k, order_A, order_sg, order_xi);
	}
	/*
	if (i % 2 == 0) {
	    printf("coeffs[[%d]] <- c(", ib++);
	    for (int j = 0; j < num_collocation; j++)
		printf("%+8.4f, ", coeffs_i1[j]);
	    printf("\b\b)\n");
	} else {
	    printf("coeffs[[%d]] <- c(", ib++);
	    for (int j = 0; j < num_collocation; j++)
		printf("%+8.4f, ", coeffs_i2[j]);
	    printf("\b\b)\n");
	}
	*//*
	if (i > 0) {
	    if (is_converged(coeffs_V1, coeffs_V2, num_collocation) == 0) {
		continue;
	    } else if (is_converged(coeffs_i1, coeffs_i2, num_collocation) == 0) {
		continue;
	    } else if (is_converged(coeffs_l1, coeffs_l2, num_collocation) == 0) {
		continue;
	    } else if (is_converged(coeffs_P1, coeffs_P2, num_collocation) == 0) {
		continue;
	    } else if (is_converged(coeffs_x1, coeffs_x2, num_collocation) == 0) {
		continue;
	    } else {
		break;
	    }
	}
    }
    //ifun.print();
    //ifun2.print();
    printf("\n");

    delete[] coeffs_V1;
    delete[] coeffs_i1;
    delete[] coeffs_l1;
    delete[] coeffs_P1;
    delete[] coeffs_x1;

    delete[] coeffs_V2;
    delete[] coeffs_i2;
    delete[] coeffs_l2;
    delete[] coeffs_P2;
    delete[] coeffs_x2;

    delete[] order_v;
    delete[] order_R;
    delete[] order_k;
    delete[] order_A;
    delete[] order_sg;
    delete[] order_xi;
*/
