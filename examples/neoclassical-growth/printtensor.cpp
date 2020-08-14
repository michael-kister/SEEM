void Tensor::print(void) const {

    Tensor T(*this);
    while (T.dimension < 4) {
	T.sizes.push_back(1);
	++T.dimension;
    }
    T.FlattenPartial();

    int d0 = dimension > 0 ? sizes[0] : 1;
    int d1 = dimension > 1 ? sizes[1] : 1;
    int d2 = dimension > 2 ? sizes[2] : 1;
    int d3 = dimension > 3 ? sizes[3] : 1;

    // the indexing is based on the true scenario,
    // while the access goes to a twisted version.
    Index I(intvec(T.dimension,0));

    printf("┌");
    for (int k = 0; k < d1; ++k) {
	for (int l = 0; l < d3; ++l) {
	    printf("────────────");
	}
	printf("\b┬");
    }
    printf("\b┐\n");
    for (int i = 0; i < d0; ++i) {
	if (i > 0) {
	    printf("├");
	    for (int k = 0; k < d1; ++k) {
		for (int l = 0; l < d3; ++l) {
		    printf("────────────");
		}
		printf("\b┼");
	    }
	    printf("\b┤\n");
	}
	for (int j = 0; j < d2; ++j) {
	    for (int k = 0; k < d1; ++k) {
		printf("│");
		for (int l = 0; l < d3; ++l) {
		    printf(" % 9.6f  ", T[I]);
		    T.Increment(I);
		}
		printf("\b");
	    }
	    printf("│\n");
	}
    }
    printf("└");
    for (int k = 0; k < d1; ++k) {
	for (int l = 0; l < d3; ++l)
	    printf("────────────");
	printf("\b┴");
    }
    printf("\b┘\n");
}
