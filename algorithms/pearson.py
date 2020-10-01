def dataflow(viewlocaltable, parameters, globaltable, attributes, globalresulttable):
    iternum = 0
    _local(iternum, viewlocaltable, parameters, attributes, globalresulttable)
    return _global(iternum, globaltable, parameters, attributes)


def _local(iternum, viewlocaltable, parameters, attributes, globalresulttable):
    return "select * from pearson_local((select * from %s));" % viewlocaltable


def _global(iternum, globaltable, parameters, attributes):
    return "select * from pearson_global((select * from %s));" % globaltable


## select pearson_global(sum(sx),sum(sxx),sum(sxy),sum(sy),sum(syy),sum(n)) from globaltable;
