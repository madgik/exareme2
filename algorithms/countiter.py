class Algorithm:
    def __init__(self):
        pass
    
    ### this function is not in the current execution flow - if it is, it works as an input to run_algorithm.dataflow_parse_and_execute
    def algorithm(self, viewlocaltable, globaltable,  parameters, attributes, globalresulttable):
        res = 0
        for iternum in range(60):
            yield self._local(iternum, viewlocaltable, parameters, attributes, globalresulttable)
            yield self._global(iternum, globaltable, parameters, attributes)
            res = (yield)
            if res[0][0] > 1000000:
                break


    def _local(self, iternum, viewlocaltable, parameters, attributes, globalresulttable):
        #### todo convert schema to a list and not string
        schema = "c1 BIGINT"
        if iternum == 0:
            sqlscript = "select count(%s) as c1 from %s;" % (attributes[0], viewlocaltable)
            return schema, sqlscript
        else:
            sqlscript = "select sum(%s) as c1 from %s;" % (attributes[0], globalresulttable)
            return schema, sqlscript


    def _global(self, iternum, globaltable, parameters, attributes):
        #### todo convert schema to a list and not string
        schema = "c1 BIGINT"
        sqlscript = "select sum(%s) as c1 from %s;" % (attributes[0], globaltable)
        return schema, sqlscript
