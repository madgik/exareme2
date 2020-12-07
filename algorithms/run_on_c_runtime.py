#just some ideas
#if we have a runtime in C (or whatever), we do not need to rewrite/edit the algorithms
# we only need some kind of middleware which communicates in some way (e.g., rest api, unix sockets whatever)
# with the runtime and translates python yielded dictionaries
# into something that C (or whatever) can understand (e.g., json, bson, msgpack whatever)

import json

def get_parameters_from_runtime(runtime_input):
    algorithm, params = json.loads(runtime_input)


def python2C(algorithm, params):
    task_generator = algorithm(params)
    while True:
        command = json.dumps(next(task_generator))
        command.send_to_runtime()
        result = receive_from_runtime()
        result = json.loads(runtime_input)
        task_generator.send(result)
        .
        .
        .
