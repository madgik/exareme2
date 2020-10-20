class Algorithm:

    def algorithm(self, data_table, merged_local_results,  parameters, attributes, result_table):
        iternum = 0
        yield self._local(iternum, data_table, parameters, attributes, result_table)
        yield self._global(iternum, merged_local_results, parameters, attributes)


    def _local(self, iternum, data_table, parameters, attributes, result_table):
        #### todo convert schema to a list and not string
        schema = "sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT"
        sqlscript = "select * from pearson_local((select * from %s))" % data_table
        return schema, sqlscript


    def _global(self, iternum, merged_local_results, parameters, attributes):
        #### todo convert schema to a list and not string
        schema = "result FLOAT"
        sqlscript  = "select * from pearson_global((select * from %s))" % merged_local_results
        return schema, sqlscript


## select pearson_global(sum(sx),sum(sxx),sum(sxy),sum(sy),sum(syy),sum(n)) from merged_local_results;
