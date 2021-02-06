from AlgorithmExecutor import AlgorithmExecutor


algorithm_params={
    "algorithmFolder":"dummy",
    "algorithmFlowFile": "dummy_flow",
    "algorithmUDFsFile":"dummy_udfs",
    "input_par1": 1.234,
    "input_par2":5.678
}

node_params={
            "globalNode":{
                "url":"amqp://user:password@localhost:5673/user_vhost"
            },
            "localNodes":[
                {
                    "url":"amqp://user:password@localhost:5673/user_vhost",
                    "baseViewName":"viewTable1"
                },
                {
                    "url":"amqp://user:password@localhost:5673/user_vhost",
                    "baseViewName":"viewTable2"
                },
                {
                    "url":"amqp://user:password@localhost:5673/user_vhost",
                    "baseViewName":"viewTable3"
                }
            ]
        }

print("(test_AlgorithmExecutor) just in")
algEx=AlgorithmExecutor(algorithm_params,node_params)
