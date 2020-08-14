void Tensor::print(void) const {

    auto SmartPrint =
	[](double x) -> void {
	    if (x <= -10000 || x >= 10000) {
		printf(" % 9.2f  ", x);
	    } else if (x <= -1000 || x >= 1000) {
		printf(" % 9.3f  ", x);
	    } else if (x <= -100 || x >= 100) {
		printf(" % 9.4f  ", x);
	    } else if (x <= -10 || x >= 10) {
		printf(" % 9.5f  ", x);
	    } else {
		printf(" % 9.6f  ", x);
	    }
	};

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
		    SmartPrint(T[I]);
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
