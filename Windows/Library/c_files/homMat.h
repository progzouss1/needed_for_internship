#define EXPORT __declspec(dllexport)

EXPORT void hom_matrix(int* indices, 
                int numIndices, 
                int* ranks, 
                int* numChebPts, 
                int dim, 
                int core_idx, 
                double* cores, 
                double* results);