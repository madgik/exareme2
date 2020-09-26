import pymonetdb

def means_by_index( indices, datapoints, data_node, nodes_broadcast):
    '''
    Calculates means of the data_table that have the same index
    
    Parameters:
    indices: an array of indices that group the data_table in some way
    datapoints:the NxK table, where N the number of datapoints and K the number of dimensions (the table must be "visible" by node_native)
    node_native: the node where the resulting table will be created
    nodes_broadcast: the nodes where REMOTE tables will be created, if empty list, the created table will only be "visible" to node_native

    Returns:
    string: Nx1 table, where N the number of different indices
    '''
    cursor=node_native.cursor()

    result_table_name=generate_unique_name()
    schema="means FLOAT"

    create_table(node_native,table_name,schema)

    query=f"INSERT INTO {result_table_name} (SELECT * means_by_index( ..indices,datapoints..))"
    cursor.execute(query)

    if nodes_broadcast:
        broadcast_table(result_table,schema,node_native,nodes_broadcast):

    return result_table_name


def min_column(table,node_native,nodes_broadcast):
    '''
    From a table NxK it returns a Nx1 selecting the minimum value from each column

    Parameters:
    table_name:the NxK table, the table must be "visible" by node_native
    node_native: the node where the resulting table will be created
    nodes_broadcast: the nodes where REMOTE tables will be created, if empty list, the created table will only be "visible" to node_native

    Returns:
    string: the name of the Nx1 table, containing the index of the column that had the minimum value within the row 
    '''
    cursor=node_native.cursor()

    result_table=generate_unique_name()
    schema=#infer schema from points_a_table & points_b_table 

    create_table(node_native,result_table,schema)

    query=f"INSERT INTO {result_table} (SELECT * min_column_udf( ..table_name..))"
    cursor.execute(query)

    if nodes_broadcast:
        broadcast_table(table,schemanode_native,nodes_broadcast):
        
    return result_table_name


def calculate_norm(datapoints_a_table,datapoints_b_table,node_native,nodes_broadcast):
    '''
    Calculates cartesian distance of each point of points_b to each point_a

    Parameters:
    points_a_table_name:the name of the table containing points a, the table must be "visible" by node_native 
    points_b_table_name:the name of the table containing points b, the table must be "visible" by node_native 
    node_native: the node where the resulting table will be created
    nodes_broadcast: the nodes where REMOTE tables will be created, if empty list, the created table will only be "visible" to node_native

    Returns:
    string: the name of the table containing the distances
    '''
    cursor=node_native.cursor()

    result_table=generate_unique_name()
    schema=#infer schema from points_a_table & points_b_table 

    create_table(node_native,result_table,schema)

    query=f"INSERT INTO {result_table} (SELECT * calculate_norm_udf( ..points_a_table,points_b_table..))"
    cursor.execute(query)

    if nodes_broadcast:
        broadcast_table(result_table,schema,node_native,nodes_broadcast):

    return result_table

def generate_random(params,node_native,nodes_broadcast):
    '''
    params: size of the desired random table_name, type of variables,range,...
    node_native: the node where the table will be created
    nodes_broadcast: the nodes where REMOTE tables will be created

    Returns:
    string: the name of the table containing the random floats
    '''
    cursor=node_native.cursor()

    result_table=generate_unique_name()
    schema=#from params
    
    create_table(result_table,schema,node_native)

    query=f"INSERT INTO {result_table} (SELECT GENERATE_RANDOM(..params..))"
    cursor.execute(query)

    if nodes_broadcast:
        broadcast_table(result_table,schema,node_native,nodes_broadcast):

    return result_table

def broadcast_table(table_name,schema,node_native,nodes_broadcast):
    for node in nodes_broadcast:
        cursor=node.cursor()
        query=f"CREATE REMOTE TABLE {table_name} ({schema}) ON 'mapi:monetdb://..native_node..'"
        cursor.execute(query)

def create_table(node,table_name,schema):
    cursor=node.cursor()

    query=f"CREATE TABLE {table_name} ({schema})"
    cursor.execute(query)

def create_merge_table(node,merge_table_name,schema,tables_to_merge):
    cursor=node.cursor()
    
    query=f"CREATE MERGE TABLE {merge_table_name} ({schema})"
    cursor.execute(query)

    for table in tables_to_merge:
        query=f"ALTER TABLE {merge_table} ADD TABLE {table}"
        cursor.execute(query)

