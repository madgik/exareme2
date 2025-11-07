import json
import random
import re
import sys
import time
import unicodedata
from typing import Generator
from typing import Iterator
from typing import Literal

import cffi
import numpy
from cffi import FFI

ffi = FFI()
lalak = 4
fofo = [1, 2, 3]
index = -1
import gc

counter = 0


def process_data(data):
    yield data * 2


process_data.registered = True


def hidden(args: Generator, rest) -> Iterator[int]:
    # j = 0
    # while True:
    # try:
    #    i = next(args)  # Get the next item from the generator
    #    if b'lalaaaklfjslfgjgd' not  in i:
    #        raise 'not in i'
    # except StopIteration:
    #    break      # Exit the loop when the generator is exhausted
    a = list(args)
    yield 1


hidden.registered = True


import gc


def returncount() -> int:
    return counter


returncount.registered = True


def gccall(*args) -> int:
    gc.collect()
    return 100000


gccall.registered = True


def mylower1(val):
    return val.lower()


mylower1.registered = True


def mytest1(val1, val2):
    return str(val1).encode() + val2


mytest1.registered = True


def filter(val):
    return val > b"s"


def jsonsplit(val) -> str:
    li = json.loads(val)
    for val in li:
        yield val.encode()


jsonsplit.registered = True


def myvals(input, insize):
    for i in range(insize):
        value = mylower1(ffi.string(input[0][i]))
        if filter(value):
            yield value


def mymylen(val) -> int:
    return len(val)


mymylen.registered = True


def distinct_cffi_char_array_fast(ffi, arr, n):
    seen = set()
    for i in range(n):
        s = ffi.string(arr[i])
        seen.add(s)
    return list(seen)


def myfilt3(
    inputd: ffi.CData, resultnum, resultarrays, insize, ffi, lib
) -> Iterator[str]:
    outputd = [None] * resultnum
    retlen = insize

    # Allocate only the string result column
    if resultarrays[0].type == 0:
        resultarrays[0].array = ffi.cast(
            "char**", lib.malloc(retlen * ffi.sizeof("char *"))
        )
        outputd[0] = ffi.cast("char **", resultarrays[0].array)

    count = 0

    y = distinct_cffi_char_array_fast(ffi, inputd[0], insize)
    for i in y:
        outputd[0][count] = ffi.from_buffer(memoryview(i))
        count += 1

    resultarrays[0].size = count
    print("Filtered count:", count)
    yield 1


myfilt3.registered = True


def myfilt2(
    inputd: ffi.CData, resultnum, resultarrays, insize, ffi, lib
) -> Iterator[str]:
    outputd = [None] * resultnum
    retlen = insize

    # Allocate only the string result column
    if resultarrays[0].type == 0:
        resultarrays[0].array = ffi.cast(
            "char**", lib.malloc(retlen * ffi.sizeof("char *"))
        )
        outputd[0] = ffi.cast("char **", resultarrays[0].array)

    count = 0

    x = [
        ffi.string(inputd[0][i]).lower()
        for i in range(insize)
        if inputd[1][i] > 600000.0
    ]
    for i in x:
        outputd[0][count] = ffi.from_buffer(memoryview(i))
        count += 1

    resultarrays[0].size = count
    print("Filtered count:", count)
    yield 1


myfilt2.registered = True


def myfilt(
    inputd: ffi.CData, resultnum, resultarrays, insize, ffi, lib
) -> Iterator[str]:
    outputd = [None] * resultnum
    retlen = insize
    for i in range(resultnum):
        if resultarrays[i].type == 0:
            resultarrays[i].array = ffi.cast(
                "char**", lib.malloc(retlen * ffi.sizeof("char *"))
            )
            outputd[i] = ffi.cast("char **", resultarrays[i].array)
        elif resultarrays[i].type == 1:
            resultarrays[i].array = ffi.cast(
                "int*", lib.malloc(retlen * ffi.sizeof("int"))
            )
            outputd[i] = ffi.cast("int *", resultarrays[i].array)
        elif resultarrays[i].type == 2:
            resultarrays[i].array = ffi.cast(
                "double*", lib.malloc(retlen * ffi.sizeof("double"))
            )
            outputd[i] = ffi.cast("double *", resultarrays[i].array)
    count = 0
    # myvalss = myvals(inputd,insize)
    # x = list(myvalss
    x = [
        my_str
        for i in range(insize)
        if filter(mylower1(my_str := ffi.string(inputd[0][i])))
    ]
    # x = [ffi.string(inputd[0][i]) for i in range(insize) if filter(mylower1(ffi.string(inputd[0][i]))) ]
    for i in x:
        # val = ffi.string(inputd[0][i]).lower()
        # vv = mymylen(ffi.string(inputd[0][i]))
        outputd[0][count] = ffi.from_buffer(memoryview(i))
        count += 1
    # for i in range(count):
    #    print(ffi.string(outputd[0][i]))
    resultarrays[0].size = count
    print("lala:", count)
    yield 1


myfilt.registered = True


def convertffi(val):
    return ffi.string(val)


from collections import OrderedDict


def ml(val):
    return val.lower()


import random
import string


def generate_random_string(length):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_random(input: Generator, param) -> Iterator[str]:

    total_strings = 100000
    string_length = 20
    distinct_percentage = 80
    # for total_strings, string_length, distinct_persentage in input:
    duplicate_percentage = 100 - distinct_percentage
    print(total_strings, string_length, distinct_percentage)
    distinct_strings_count = int(total_strings * (distinct_percentage / 100))

    distinct_strings = set()
    while len(distinct_strings) < distinct_strings_count:
        distinct_strings.add(generate_random_string(string_length))

    duplicate_strings_count = total_strings - distinct_strings_count
    duplicate_strings = [
        random.choice(list(distinct_strings)) for _ in range(duplicate_strings_count)
    ]

    result = list(distinct_strings) + duplicate_strings
    random.shuffle(result)
    for row in result:
        yield row.encode()
    # return result


generate_random.registered = True

# Example usage
# total_strings = 20
# string_length = 8
# distinct_percentage = 10

# random_string_list = generate_random_string_list(total_strings, string_length, distinct_percentage)
# print(random_string_list)
# print(len(set(random_string_list)))


def mydistinct(
    inputd: ffi.CData, resultnum, resultarrays, insize, ffi, lib
) -> Iterator[str]:
    outputd = [None] * resultnum
    retlen = insize
    for i in range(resultnum):
        if resultarrays[i].type == 0:
            resultarrays[i].array = ffi.cast(
                "char**", lib.malloc(retlen * ffi.sizeof("char *"))
            )
            outputd[i] = ffi.cast("char **", resultarrays[i].array)
        elif resultarrays[i].type == 1:
            resultarrays[i].array = ffi.cast(
                "int*", lib.malloc(retlen * ffi.sizeof("int"))
            )
            outputd[i] = ffi.cast("int *", resultarrays[i].array)
        elif resultarrays[i].type == 2:
            resultarrays[i].array = ffi.cast(
                "double*", lib.malloc(retlen * ffi.sizeof("double"))
            )
            outputd[i] = ffi.cast("double *", resultarrays[i].array)

    # dset = [len(distset.add(ffi.string(input[0][i])) or distset) for i in range(insize)]
    # dset = {ffi.string(inputd[0][i]):i for i in range(insize)}
    # for count, item in enumerate(dset.keys()):
    #     outputd[0][count] = inputd[0][dset[item]]
    distset = set()
    distset = OrderedDict()
    count = 0
    for i in range(insize):
        val = ffi.string(inputd[0][i])
        # val = ml(val)
        if val not in distset:
            distset[val] = True
            outputd[0][count] = lib.strdup(
                ffi.from_buffer(memoryview(val))
            )  # inputd[0][i]
            count += 1

    # seen_values = set()
    # res = [(outputd[0][i] = inputd[0][i]) for i in range(insize) if (val := ffi.string(inputd[0][i])) not in seen_values and not seen_values.add(val)]
    # for count,val in enumerate(output_list):
    #    outputd[0][count] = val
    resultarrays[0].size = count
    yield 1


mydistinct.registered = True


def filt(input: Generator, insize, lala, params=None) -> Iterator[str]:
    for i in range(insize):
        # val = ffi.string(input[0][i])
        if ffi.string(input[0][i]) > b"99":
            yield input[0][i]


filt.registered = True


def mmlower(arg):
    return arg.lower()


mmlower.registered = True


def iteratorf2(input: Generator):
    for i in range(100):
        yield i, i + 1


iteratorf2.registered = True


def iteratorf(input: Generator):
    for value in input:
        yield value


iteratorf.registered = True


def reservoir_sampling_generator(iterator, N):
    global counter

    reservoir = []
    for i, item in enumerate(iterator):
        if i < N:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < N:
                reservoir[j] = item

    for item in reservoir:
        yield item


def rsample(input: Generator, otherargs):
    samplesize = 1000
    if otherargs != None:
        samplesize = int(otherargs[0])
    sample = reservoir_sampling_generator(input, samplesize)
    for value in sample:
        yield value


rsample.registered = True


def iterfun1(input: Generator, otherargs) -> Generator[str, int, None]:
    for row in input:
        yield row.lower(), len(row)


iterfun1.registered = True


def iterfun2(input: Generator):
    for row in input:
        yield row.lower(), len(row)


iterfun2.registered = True


def replaceuni(input_string):
    cleaned_string = re.sub(r"[^\x00-\x7F]+", "", input_string)
    return cleaned_string


replaceuni.registered = True


def myjpack(*args):
    return jpacks.jlen(jpacks.jpack(*args))


myjpack.registered = True


def mylower(input2):
    # normalized_str = unicodedata.normalize('NFKD', input2.decode('utf-8')).encode('utf-8')

    # Convert the normalized string to lowercase
    # lower_str = normalized_str.lower()

    # return lower_str
    return 100
    return input2.lower()


mylower.registered = True


def mylowergen(input2):
    global lalak
    global index
    index += 1
    if index > len(fofo) - 1:
        index = 0
    y = [x * lalak for x in fofo]
    return input2.lower(), y[index]


def mylowergener(input2):
    yield input2.lower(), len(input2)


mylowergen.registered = True
mylower.registered = True


def jtokens(input):
    return json.dumps(input.split())


jtokens.registered = True


def jl(input):
    return len(json.loads(input))


jl.registered = True


def jfuse(input):
    return jl(jtokens(input))


jfuse.registered = True


def mylen(input2) -> int:
    return len(input2)


mylen.registered = True


def myfuse(input1):
    return mylen(mylower(input1))


myfuse.registered = True


def lala(input):
    return input.lower()


def mydouble(input):
    """
    lala
    """
    return 1.5


mydouble.registered = True


def mydouble2(input):
    return "1.5"


mydouble2.registered = True


def mydouble3(input):
    return "1.5"


def myfunc2(inp):
    return inp.lower()


myfunc2.registered = True


mydouble3.registered = True

lala.registered = True
