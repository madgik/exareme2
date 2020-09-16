CREATE or replace AGGREGATE numpy_sum(val BIGINT) 
RETURNS BIGINT
LANGUAGE PYTHON {
    return numpy.sum(val)
};

CREATE or replace AGGREGATE numpy_count(val BIGINT) 
RETURNS BIGINT
LANGUAGE PYTHON {
    import time
    time.sleep(4)
    return val.size
};


CREATE or replace FUNCTION pearson_local(val1 FLOAT, val2 FLOAT) 
RETURNS TABLE(sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT) 
LANGUAGE PYTHON {
    import sys
    sys.path.append("/home/openaire/monetdb_federated_poc/algorithms")
    import pearson_lib
    return pearson_lib.local(val1,val2)

};

CREATE or replace FUNCTION pearson_global(sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT) 
RETURNS TABLE(res FLOAT)
LANGUAGE PYTHON {
        import sys
        sys.path.append("/home/openaire/monetdb_federated_poc/algorithms")
        import pearson_lib
        return pearson_lib.merge(sx,sxx,sxy,sy,syy,n)
   
};
