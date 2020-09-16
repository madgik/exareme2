def _local_init(viewlocaltable, parameters, attr):
    return "select numpy_count(%s) as c1 from %s;" %(attr[0], viewlocaltable)
    
def _global_iter(globaltable, parameters, attr):
    return "select numpy_sum(%s) as c1 from %s;" %(attr[0],globaltable)
    
def _local_iter(globalresulttable, parameters, attr):
    return "select numpy_sum(%s) as c1 from %s;" %(attr[0], globalresulttable)
    
def _global(globaltable, parameters, attr):
    return "select numpy_sum(%s) as c1 from %s;" %(attr[0],globaltable)
