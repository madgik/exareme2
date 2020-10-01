def dataflow(viewlocaltable, globaltable, globalresulttable, parameters, attributes):
    res = 0
    for iternum in range(100):
        _local(iternum, viewlocaltable, parameters, attributes, globalresulttable)
        res = _global(iternum, globaltable, parameters, attributes)
        if res[0][0] > 1000000:
            break
    return res


def _local(iternum, viewlocaltable, parameters, attributes, globalresulttable):
    if iternum == 0:
        return "select count(%s) as c1 from %s;" % (attributes[0], viewlocaltable)
    else:
        return "select sum(%s) as c1 from %s;" % (attributes[0], globalresulttable)


def _global(iternum, globaltable, parameters, attributes):
    return "select sum(%s) as c1 from %s;" % (attributes[0], globaltable)
