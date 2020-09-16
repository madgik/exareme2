def _local(viewlocaltable, parameters, attr):
    return "select * from pearson_local((select * from %s));" %viewlocaltable
    
def _global(globaltable, parameters, attr):
    return "select * from pearson_global((select * from %s));" %globaltable

## select pearson_global(sum(sx),sum(sxx),sum(sxy),sum(sy),sum(syy),sum(n)) from globaltable;
