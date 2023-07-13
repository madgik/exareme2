from exareme2.udfgen.adhoc_udfgenerator import AdhocUdfGenerator
from exareme2.udfgen.decorator import UDFBadCall
from exareme2.udfgen.py_udfgenerator import PyUdfGenerator


def get_udfgenerator(
    udfregistry,
    func_name,
    flowargs,
    flowkwargs,
    smpc_used,
    request_id,
    output_schema,
    min_row_count,
):
    if func_name in udfregistry:
        return PyUdfGenerator(
            udfregistry,
            func_name,
            flowargs,
            flowkwargs,
            smpc_used,
            request_id,
            output_schema,
            min_row_count,
        )
    elif AdhocUdfGenerator.is_registered(func_name):
        udfgen_class = AdhocUdfGenerator.get_subclass(func_name)
        if flowargs:
            msg = f"flowargs cannot be used with {udfgen_class.__name__}"
            raise UDFBadCall(msg)
        return udfgen_class(flowkwargs, smpc_used, output_schema, min_row_count)
    raise ValueError(f"UDF named '{func_name}' not found in registries.")
