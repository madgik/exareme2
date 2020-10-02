class Algorithm:
    def __init__(self):
        pass

    ### this function is not in the current execution flow - if it is, it works as an input to run_algorithm.dataflow_parse_and_execute
    def dataflow(self, viewlocaltable, parameters, globaltable, attributes, globalresulttable):
        iternum = 0
        self._local(iternum, viewlocaltable, parameters, attributes, globalresulttable)
        return self._global(iternum, globaltable, parameters, attributes)


    def _local(self, iternum, viewlocaltable, parameters, attributes, globalresulttable):
        return "select * from pearson_local((select * from %s));" % viewlocaltable


    def _global(self, iternum, globaltable, parameters, attributes):
        return "select * from pearson_global((select * from %s));" % globaltable


## select pearson_global(sum(sx),sum(sxx),sum(sxy),sum(sy),sum(syy),sum(n)) from globaltable;
