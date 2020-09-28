def dataflow(viewlocaltable, globaltable, globalresulttable, parameters, attr):
    res = 0
    for iternum in range(100):
        _local(iternum, viewlocaltable, parameters, attr, globalresulttable)
        res = _global(iternum, globaltable, parameters, attr)
        if res[0][0] > 1000000:
            break
    return res


def _local(iternum, viewlocaltable, parameters, attr, globalresulttable):
    if iternum == 0:
        return "select count(%s) as c1 from %s;" %(attr[0], viewlocaltable)
    else:
        return "select sum(%s) as c1 from %s;" % (attr[0], globalresulttable)


def _global(iternum, globaltable, parameters, attr):
    return "select sum(%s) as c1 from %s;" %(attr[0],globaltable)