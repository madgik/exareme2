class Algorithm:
    def __init__(self):
        pass

    def algorithm(self, viewlocaltable,globaltable,  parameters, attributes, globalresulttable):
        iternum = 0
        yield self._local(iternum, viewlocaltable, parameters, attributes, globalresulttable)
        yield self._global(iternum, globaltable, parameters, attributes)


    def _local(self, iternum, viewlocaltable, parameters, attributes, globalresulttable):
        #### todo convert schema to a list and not string
        schema = "sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT"
        sqlscript = "select * from pearson_local((select * from %s));" % viewlocaltable
        return schema, sqlscript


    def _global(self, iternum, globaltable, parameters, attributes):
        #### todo convert schema to a list and not string
        schema = "result FLOAT"
        sqlscript  = "select * from pearson_global((select * from %s));" % globaltable
        return schema, sqlscript


## select pearson_global(sum(sx),sum(sxx),sum(sxy),sum(sy),sum(syy),sum(n)) from globaltable;
