
__device__ void chebyshev_polynomial(double x, double* y, int o)
{
    switch(o) {
    case 0:
	*y = 1.0;
	break;
    case 1:
	*y = x;
	break;
    case 2:
	*y = 2.0*x*x - 1.0;
	break;
    case 3:
	*y = x*(4.0*x*x - 3.0);
	break;
    case 4:
	*y = 8.0*x*x*x*x - 8.0*x*x + 1.0;
	break;
    case 5:
	*y = x*(16.0*x*x*x*x - 20.0*x*x + 5.0);
	break;
    case 6:
	*y = 32.0*x*x*x*x*x*x - 48.0*x*x*x*x + 18.0*x*x - 1.0;
	break;
    case 7:
	*y = x*(64.0*x*x*x*x*x*x - 112.0*x*x*x*x + 56.0*x*x - 7.0);
	break;
    case 8:
	*y = 128.0*x*x*x*x*x*x*x*x - 256.0*x*x*x*x*x*x + 160.0*x*x*x*x - 32.0*x*x + 1.0;
	break;
    case 9:
	*y = x*(256.0*x*x*x*x*x*x*x*x - 576.0*x*x*x*x*x*x + 432.0*x*x*x*x - 120.0*x*x + 9.0);
	break;
    case 10:
	*y = 512.0*x*x*x*x*x*x*x*x*x*x - 1280.0*x*x*x*x*x*x*x*x + 1120.0*x*x*x*x*x*x - 400.0*x*x*x*x + 50.0*x*x - 1.0;
	break;
    case 11:
	*y = x*(1024.0*x*x*x*x*x*x*x*x*x*x - 2816.0*x*x*x*x*x*x*x*x + 2816.0*x*x*x*x*x*x - 1232.0*x*x*x*x + 220.0*x*x - 11.0);
	break;
    case 12:
	*y = 2048.0*x*x*x*x*x*x*x*x*x*x*x*x - 6144.0*x*x*x*x*x*x*x*x*x*x + 6912.0*x*x*x*x*x*x*x*x - 3584.0*x*x*x*x*x*x + 840.0*x*x*x*x - 72.0*x*x + 1.0;
	break;
    case 13:
	*y = x*(4096.0*x*x*x*x*x*x*x*x*x*x*x*x - 13312.0*x*x*x*x*x*x*x*x*x*x + 16640.0*x*x*x*x*x*x*x*x - 9984.0*x*x*x*x*x*x + 2912.0*x*x*x*x - 364.0*x*x + 13.0);
	break;
    case 14:
	*y = 8192.0*x*x*x*x*x*x*x*x*x*x*x*x*x*x - 28672.0*x*x*x*x*x*x*x*x*x*x*x*x + 39424.0*x*x*x*x*x*x*x*x*x*x - 26880.0*x*x*x*x*x*x*x*x + 9408.0*x*x*x*x*x*x - 1568.0*x*x*x*x + 98.0*x*x - 1.0;
	break;
    case 15:
	*y = x*(16384.0*x*x*x*x*x*x*x*x*x*x*x*x*x*x - 61440.0*x*x*x*x*x*x*x*x*x*x*x*x + 92160.0*x*x*x*x*x*x*x*x*x*x - 70400.0*x*x*x*x*x*x*x*x + 28800.0*x*x*x*x*x*x - 6048.0*x*x*x*x + 560.0*x*x - 15.0);
	break;
    case 16:
	*y = 32768.0*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x - 131072.0*x*x*x*x*x*x*x*x*x*x*x*x*x*x + 212992.0*x*x*x*x*x*x*x*x*x*x*x*x - 180224.0*x*x*x*x*x*x*x*x*x*x + 84480.0*x*x*x*x*x*x*x*x - 21504.0*x*x*x*x*x*x + 2688.0*x*x*x*x - 128.0*x*x + 1.0;
	break;
    default:
	*y = 0.0;
	break;
    }
}
