class Algorithm:

    def algorithm(self, data_table, merged_local_results,  parameters, attributes, result_table):
        res = 0
        for iternum in range(60):
            yield self._local(iternum, data_table, parameters, attributes, result_table)
            yield self._global(iternum, merged_local_results, parameters, attributes)
            res = (yield)
            if res[0][0] > 1000000:
                break


    def _local(self, iternum, data_table, parameters, attributes, result_table):
        #### todo convert schema to a list and not string
        schema = "c1 BIGINT"
        if iternum == 0:
            sqlscript = "select count(%s) as c1 from %s;" % (attributes[0], data_table)
            return schema, sqlscript
        else:
            sqlscript = "select sum(%s) as c1 from %s;" % (attributes[0], result_table)
            return schema, sqlscript


    def _global(self, iternum, merged_local_results, parameters, attributes):
        #### todo convert schema to a list and not string
        schema = "c1 BIGINT"
        sqlscript = "select sum(%s) as c1 from %s;" % (attributes[0], merged_local_results)
        return schema, sqlscript
