typedef struct {
    int type;  // Type identifier: 1 for char**, 2 for int*, 3 for double*
    void* array;  // Pointer to the array
    int size;  // Size of the array
    } ArrayInfo;

typedef struct {
    int type;  // Type identifier: 1 for char**, 2 for int*, 3 for double*
    void* array;  // Pointer to the array
    int size;  // Size of the array
    } ArrayInfo1;

extern int freecarray(ArrayInfo1*, int, int);
extern int udfwrapper(char* funcname, int paramscount, ArrayInfo1*,  double* result, int* result1, char** result2, int rowcount, int isliteral, char** errormessage) ;
extern int udfwrapper_parallel(char* funcname, int paramscount, ArrayInfo1*,  double* result, int* result1, char** result2, int rowcount, char** errormessage) ;
extern int rowtablewrapper(char*, int, ArrayInfo1*, int, ArrayInfo1*, int);
extern int rowtuplewrapper(char*, int, ArrayInfo1*, int, ArrayInfo1*, int);
extern int tableudfwrapper(char*, int, ArrayInfo1*, int, ArrayInfo1*, int, char*, char** errormessage);
extern int systemwrapper(char*, int, ArrayInfo1*, int, ArrayInfo1*, int, char*, char** errormessage);
extern int numpyudfwrapper(char*, int, ArrayInfo1*, int, ArrayInfo1*, int, char*, char** errormessage);
extern int numpyudfwrapperwithtests(char*, int, ArrayInfo1*, int, ArrayInfo1*, int, char*);
extern int tablematerwrapper(int, ArrayInfo1*, int, ArrayInfo1*, int);
extern int tabletuple(char*, int, ArrayInfo1*, int, ArrayInfo1*, int);
extern int tablevalue(char*, int, ArrayInfo1*, int, ArrayInfo1*, int);
extern int call_gc();
extern int disable_gc();
extern int myfree(void* array);
extern int expandwrapper(char*, int, ArrayInfo1*, int, ArrayInfo1*, int*, int, int, int*, char** errormessage);
extern int expandaggrwrapper(char*, int, ArrayInfo1*, int, ArrayInfo1*, int*, int, int, int*, char** errormessage);
extern int aggregatewrapper(char * funcname,int input_count,  int paramscount, ArrayInfo1 *, double *result, int *result1, char **result2, int groups, size_t* count_per_group, char** errormessage);
extern int scalarfusionwrapper(char** funcname, int funcnum, int paramscount, ArrayInfo1*,  double* result, int* result1, char** result2, int rowcount, int isliteral);
extern int aggregatefusionwrapper(char** funcname, int paramscount, ArrayInfo1 *, double *result, int *result1, char **result2, int groups, size_t* count_per_group);
extern int tablefusionwrapper(char**, int, ArrayInfo1*, int, ArrayInfo1*, int);


extern int fusionwrapper(char**, int, int, ArrayInfo1 *, double*, int*,char**, int, ArrayInfo1*, int, char*, int, int);
