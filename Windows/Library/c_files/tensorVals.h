#define EXPORT __declspec(dllexport)

EXPORT void tensor_vals (int* indices, 
						int numIndices, 
						int* ranks, 
						int* numChebPts, 
						int dim, 
						double* cores, 
						double* results);//, double* result);//, double** cores);