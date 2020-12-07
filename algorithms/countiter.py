class Algorithm:

    def algorithm(self, data_table, merged_local_results, parameters, attributes, result_table):
        res = 0

        for iternum in range(60):
            yield self._local(iternum, data_table, parameters, attributes, result_table)
            yield self._global(iternum, merged_local_results, parameters, attributes)
            res = yield
            if res[0][0] > 100000000:
                break


    def _local(self, iternum, data_table, parameters, attributes, result_table):
        #### todo convert schema to a list and not string
        schema = "c1 BIGINT"
        if iternum == 0:
            sqlscript = f'''
                            SELECT 
                                    COUNT({attributes[0]}) AS c1 
                            FROM {data_table}
                        '''
            return {'run_local': {'schema':schema, 'sqlscript':sqlscript}}
        else:
            sqlscript = f'''
                            SELECT
                                   SUM({attributes[0]}) AS c1 
                            FROM {result_table} 
                        '''
            return {'run_local': {'schema':schema, 'sqlscript':sqlscript}}


    def _global(self, iternum, merged_local_results, parameters, attributes):
        #### todo convert schema to a list and not string
        schema = "c1 BIGINT"
        sqlscript = f'''
                        SELECT 
                            SUM({attributes[0]}) as c1 
                        FROM {merged_local_results}
                    '''
        return {'run_global' : {'schema': schema, 'sqlscript': sqlscript}}
