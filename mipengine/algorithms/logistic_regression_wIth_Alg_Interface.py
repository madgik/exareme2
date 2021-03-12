
PREC = 1e-6

def logistic_regression(algo_interface: 'AlgorithmExecutionInterface'):

    run_on_locals = algo_interface.run_udf_on_local_nodes
    run_on_global = algo_interface.run_udf_on_global_node
    get_table_schema = algo_interface.get_table_schema
    get_table_data = algo_interface.get_table_data

    X: 'LocalNodeTable' = algo_interface.initial_view_tables['x']
    y: 'LocalNodeTable' = algo_interface.initial_view_tables['y']
    classes: List[str] = algo_interface.parameters
    # OR
    # def logistic_regression(algo_interface, Dict[LocalNodeTable], parameters):

    num_of_coeffs = len(get_table_schema(X).columns)  # get_table_schema(X) returns TableSchema obj
    logloss = 1e6

    ybin: 'LocalNodeTable' = run_on_locals('binarize_labels', y, classes)  # no sharing to global

    while True:
        z: 'LocalNodeTable' = run_on_locals('matrix_dot_vector', X, num_of_coeffs)
        s = run_on_locals('tensor_expit', z)

        tmp = run_on_locals('const_tensor_sub', 1, s)
        d = run_on_locals('tensor_mult', tmp)

        tmp = run_on_locals('tensor_sub', ybin, s)
        y_ratio = run_on_locals('tensor_div', tmp, d)

        hessian: 'GlobalNodeTable' = run_on_locals(X, d, share_to_global=True)

        tmp = run_on_locals('tensor_add', z, y_ratio)
        grad: 'GlobalNodeTable' = run_on_locals('mat_transp_dot_diag_dot_vec', X, d, tmp, share_to_global=True)

        newlogloss = run_on_locals('logistic_loss', ybin, s)

        # ******** Global part ******** #
        tmp = run_on_global('mat_inverse', hessian)
        coeff = run_on_global('matrix_dot_vector', tmp, grad)

        newlogloss_pulled = get_table_data(newlogloss).data  # get_table_data returns a TableData obj
        if abs(newlogloss_pulled - logloss) <= PREC:
            break
        logloss = newlogloss_pulled

    return get_table_data(coeff)
