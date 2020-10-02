class Algorithm:
    def __init__(self):
        pass
    
    ### this function is not in the current execution flow - if it is, it is an input to run_algorithm.dataflow_parse_and_execute
    def dataflow(self, viewlocaltable, globaltable, globalresulttable, parameters, attributes):
        res = 0
        for iternum in range(100):
            self._local(iternum, viewlocaltable, parameters, attributes, globalresulttable)
            res = self._global(iternum, globaltable, parameters, attributes)
            if res[0][0] > 1000000:
                break
        return res


    def _local(self, iternum, viewlocaltable, parameters, attributes, globalresulttable):
        if iternum == 0:
            return "select count(%s) as c1 from %s;" % (attributes[0], viewlocaltable)
        else:
            return "select sum(%s) as c1 from %s;" % (attributes[0], globalresulttable)


    def _global(self, iternum, globaltable, parameters, attributes):
        return "select sum(%s) as c1 from %s;" % (attributes[0], globaltable)
