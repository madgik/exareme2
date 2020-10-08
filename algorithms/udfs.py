udf_info = {}

local_pearson_udf_declaration = '''
CREATE or replace FUNCTION local_pearson(val1 FLOAT, val2 FLOAT) 
RETURNS TABLE(sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT) 
LANGUAGE PYTHON {
    import math
    import numpy
    result = {}
    X = val1
    Y = val2
    result["sx"] = X.sum(axis=0)
    result["sxx"] = (X ** 2).sum(axis=0)
    result["sxy"] = (X * Y).sum(axis=0)
    result["sy"] = Y.sum(axis=0)
    result["syy"] = (Y ** 2).sum(axis=0)
    result["n"] = X.size
    return result
};
'''
local_pearson_return_schema = "sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT"

local_pearson = {"declaration": local_pearson_udf_declaration, "return_schema": local_pearson_return_schema}
udf_info["local_pearson"] = local_pearson

global_pearson_udf_declaration = '''
CREATE or replace FUNCTION global_pearson(sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT) 
RETURNS TABLE(res FLOAT)
LANGUAGE PYTHON {
    import math
    import numpy
    n = numpy.sum(n)
    sx = numpy.sum(sx)
    sxx = numpy.sum(sxx)
    sxy = numpy.sum(sxy)
    sy = numpy.sum(sy)
    syy = numpy.sum(syy)
    d = math.sqrt(n * sxx - sx * sx) * math.sqrt(n * syy - sy * sy)
    return float((n * sxy - sx * sy) / d)
};
'''
global_pearson_return_schema = "result FLOAT"

global_pearson = {"declaration": global_pearson_udf_declaration, "return_schema": global_pearson_return_schema}
udf_info["global_pearson"] = global_pearson
