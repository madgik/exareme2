import cffi

ffibuilder = cffi.FFI()

with open("udfs.h") as f:
    # read plugin.h and pass it to embedding_api(), manually
    # removing the '#' directives and the CFFI_DLLEXPORT
    data = "".join([line for line in f if not line.startswith("#")])
    data = data.replace("CFFI_DLLEXPORT", "")
    ffibuilder.embedding_api(data)

ffibuilder.set_source(
    "my_plugin",
    r"""
    #include "udfs.h"
    #include <stdlib.h>
    #include <string.h>
    #include <stdint.h>
    void custom_free(char** ptr_array) {
        int i = 0;
        printf("automatic gc collect\n");
        while (1) {
            if (ptr_array[i] != NULL) {
                free(ptr_array[i]);
            }
            else break;
            i++;
        }
        free(ptr_array);
    }
    void custom_free2(char** ptr_array) {
        printf("automatic gc collect\n");
        free(ptr_array);
    }
    void custom_free3(int* ptr_array) {
        printf("automatic gc collect\n");
        free(ptr_array);
    }
""",
)

ffibuilder.cdef("void*  malloc(size_t s);")
ffibuilder.cdef("void strcpy(char* dest, char* src);")
ffibuilder.cdef("size_t strlen(char* s);")
ffibuilder.cdef("char *strdup(const char *);")
ffibuilder.cdef("void *memcpy(void *dest, const void * src, size_t n);")
ffibuilder.cdef("void *realloc(void *ptr, size_t size);")
ffibuilder.cdef("void free(void *ptr);")
ffibuilder.cdef("void custom_free(char ** ptr_array);")
ffibuilder.cdef("void custom_free2(char ** ptr_array);")
ffibuilder.cdef("void custom_free3(int* ptr_array);")


ffibuilder.embedding_init_code(
    """
    from fractions import Fraction
    import threading
    import traceback
    import math
    import numpy as np
    import multiprocessing
    import multiprocessing.shared_memory
    import ctypes
    import weakref
    import os
    import struct
    import sys
    from my_plugin import  lib, ffi
    import json
    import importlib
    from importlib import reload
    env=os.path.expanduser(os.path.expandvars('$PYTHONPATH')) # "source" directory with python script
    mypath = env.split(';')
    for p in mypath:
        sys.path.insert(0, p)
    import inspect
    import time
    import gc
    import os
    import hashlib
    import functions
    try:
        import __pypy__
    except:
        pass
    #gc.disable()
    #tmpstrs = {}
    iters = {}
    global_lock = threading.Lock()
    tmpstrs = []
    checksums = {}
    #li = [None]*10
    @ffi.def_extern()
    def disable_gc():
        gc.disable()
    inc = 0
    @ffi.def_extern()
    def call_gc():
        gc.enable()
        gc.collect()
        global tmpstrs
        del tmpstrs[:]
        global iters
        del iters
        iters = {}

    @ffi.def_extern()
    def freecarray(arrays, num, count):
        #for i in range(num):
        #    for r in range(count):
        #      if ffi.cast("char **", arrays[i].array )[r] != ffi.NULL:
        #           lib.free(ffi.cast("char **", arrays[i].array )[r])

        #for i in range(num):
            #ffi.release(arrays[i].array)
            #del arrays[i].array
            #lib.free(arrays[i].array)
        #del arrays
        #print('ksuhgarghfarghfausdhfhahsdkfhasdghfasdhfjhasdfjhaghsdjhfgjashdgfasdygfjsbfvjhsbfej')
        return 1

    @ffi.def_extern()
    def udfwrapper_parallel(funcname, paramsnum, arrays, resultdouble, resultint, resultstring, insize, errormessage):
      #gc.collect()
      if resultint!= ffi.NULL:
        shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size = ffi.sizeof("ArrayInfo1") * paramsnum )
        array_info_array_shared = ffi.from_buffer("ArrayInfo1[]",shared_memory.buf)
        shared_memory2 = multiprocessing.shared_memory.SharedMemory(create=True, size = ffi.sizeof("int") * insize )
        array_info_array_shared2 = ffi.from_buffer("int[]",shared_memory2.buf)
        array_info_array_shared2 = resultint
        for i in range(paramsnum):
            array_info_array_shared[i] = arrays[i]
        process = multiprocessing.Process(target=udfwrapperprocess, args=(funcname, paramsnum, shared_memory, resultdouble, shared_memory2, resultstring, insize))
        process.start()
        process.join()
        array_info_array_shared2 = ffi.from_buffer("int[]",shared_memory2.buf)
        for i in range(insize):
            resultint[i] = array_info_array_shared2[i]
        shared_memory.close()
        shared_memory.unlink()
        shared_memory2.close()
        shared_memory2.unlink()
        return 1

    @ffi.def_extern()
    def udfwrapper(funcname, paramsnum, arrays, resultdouble, resultint, resultstring, insize, isliteral, errormessage):
       global iters
       global global_lock
       global tmpstrs
       errorm = 0;
       try:
         with global_lock:
           #del tmpstrs[:]
           tmpstrs = []
           funcname = ffi.string(funcname).decode()
           funcname = funcname.split('.')
           inputd = [None]*paramsnum
           for i in range(paramsnum):
             if arrays[i].type == 0:
                 inputd[i] = ffi.cast("char **", arrays[i].array )
             elif arrays[i].type == 1:
                 inputd[i] = ffi.cast("int *", arrays[i].array )
             elif arrays[i].type == 2:
                 inputd[i] = ffi.cast("double *", arrays[i].array)
             elif arrays[i].type == 4:
                 inputd[i] = ffi.cast("int8_t *", arrays[i].array)
             elif arrays[i].type == 5:
                 inputd[i] = ffi.cast("int16_t *", arrays[i].array)
           if resultstring != ffi.NULL:
               resultparam = 'resultstring'
           elif resultint != ffi.NULL:
               resultparam = 'resultint'
           elif resultdouble != ffi.NULL:
               resultparam = 'resultdouble'
           code_params = ', '.join(['inputd['+str(j)+'][n]' if arrays[j].type != 0 else 'ffi.string(ffi.cast("char **", arrays['+str(j)+'].array )[n])' for j in range(paramsnum)])
           ##TODO to correct this with all datatypes
           #code_params = ', '.join(['ffi.cast("int*", arrays['+str(j)+'].array )[n]' if arrays[j].type != 0 else 'ffi.string(ffi.cast("char **", arrays['+str(j)+'].array )[n])' for j in range(paramsnum)])

           if resultparam != 'resultstring':
               code_string = '''
    try:
     try:
        from '''+funcname[0]+''' import '''+funcname[1]+'''
        if functions.DEBUG:
          mypath = get_module_path(funcname[0]+'.'+funcname[1])
          #print(mypath)
          if mypath not in checksums:
            checksums[mypath] = calculate_checksum(mypath)
          elif mypath in checksums:
            csum = calculate_checksum(mypath)
            if checksums[mypath] != csum:
              importlib.reload('''+funcname[1]+''')
              checksums[mypath] = csum
        for n in range(insize):
            '''+resultparam+'''[n] = '''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+''')
     except ImportError as import_err:
      importlib.reload('''+funcname[0]+''')
      from  '''+funcname[0]+''' import '''+funcname[1]+'''

      for n in range(insize):
          '''+resultparam+'''[n] = '''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+''')
    except Exception as e:
           errormessage[0] = lib.strdup(ffi.from_buffer(memoryview(b'UDFError '+ funcname[0].encode()+b'.'+funcname[1].encode()+b'.'+funcname[2].upper().encode() + b': ' + traceback.format_exc().encode())))
           errorm = 1
         '''
           elif isliteral == 0:
               code_string = '''
    try:
     try:
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
        if functions.DEBUG:
          mypath = get_module_path(funcname[0]+'.'+funcname[1])
          #print(mypath)
          if mypath not in checksums:
            checksums[mypath] = calculate_checksum(mypath)
          elif mypath in checksums:
            csum = calculate_checksum(mypath)
            if checksums[mypath] != csum:
              importlib.reload('''+funcname[1]+''')
              checksums[mypath] = csum
     except ImportError as import_err:
        importlib.reload('''+funcname[0]+''')
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
     try:
        if inspect.isgeneratorfunction('''+funcname[1]+'''.'''+funcname[2]+'''):
            for n in range(insize):
                func = '''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+''')
                val = str(func)[-19:-1]
                iters[val] = func
                resultstring[n] = ffi.from_buffer(memoryview(val))
        else:
            def tmpfunc(arrays, inputd, insize, func):
                return [func('''+code_params+''') for n in range(insize)]
            tmpstrs = tmpfunc(arrays,inputd, insize, '''+funcname[1]+'''.'''+funcname[2]+''')
            #def tmpfunc(arrays, insize, func):
            #    return [ffi.cast("char **", arrays[0].array )[n] for n in range(insize)]
            #tmpstrs = tmpfunc(arrays, insize, '''+funcname[1]+'''.'''+funcname[2]+''')
            for n in range(insize):
                #resultstring[n] = tmpstrs[n]
                resultstring[n] = ffi.from_buffer(memoryview(tmpstrs[n]))
                #resultstring[n] = lib.strdup(ffi.from_buffer(memoryview('''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+'''))))
     except ImportError as import_err:
        importlib.reload('''+funcname[0]+''')
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
        if inspect.isgeneratorfunction('''+funcname[1]+'''.'''+funcname[2]+'''):
            for n in range(insize):
                func = '''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+''')
                val = str(func)[-19:-1]
                iters[val] = func
                resultstring[n] = ffi.from_buffer(memoryview(val))
        else:
            def tmpfunc(arrays, inputd, insize, func):
                return [func('''+code_params+''') for n in range(insize)]
            tmpstrs = tmpfunc(arrays,inputd, insize, '''+funcname[1]+'''.'''+funcname[2]+''')
            for n in range(insize):
                resultstring[n] = ffi.from_buffer(memoryview(tmpstrs[n]))
                #resultstring[n] = lib.strdup(ffi.from_buffer(memoryview('''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+'''))))
    except Exception as e:
           errormessage[0] = lib.strdup(ffi.from_buffer(memoryview(b'UDFError '+ funcname[0].encode()+b'.'+funcname[1].encode()+b'.'+funcname[2].upper().encode() + b': ' + traceback.format_exc().encode())))
           errorm = 1
           '''
           elif isliteral == 1:
               code_string = '''
    try:
      try:
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
      except:
        importlib.reload('''+funcname[0]+''')
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
      try:
        if inspect.isgeneratorfunction('''+funcname[1]+'''.'''+funcname[2]+'''):
            for n in range(insize):
                func = '''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+''')
                val = str(func)[-19:-1]
                iters[val] = func
                resultstring[n] = ffi.from_buffer(memoryview(val))
        else:
            for n in range(insize):
                resultstring[n] = ffi.from_buffer(memoryview('''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+''')))
      except:
        importlib.reload('''+funcname[0]+''')
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
        if inspect.isgeneratorfunction('''+funcname[1]+'''.'''+funcname[2]+'''):
            for n in range(insize):
                func = '''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+''')
                val = str(func)[-19:-1]
                iters[val] = func
                resultstring[n] = ffi.from_buffer(memoryview(val))
        else:
            for n in range(insize):
                resultstring[n] = ffi.from_buffer(memoryview('''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+''')))
    except Exception as e:
           errormessage[0] = lib.strdup(ffi.from_buffer(memoryview(b'UDFError '+ funcname[0].encode()+b'.'+funcname[1].encode()+b'.'+funcname[2].upper().encode() + b': ' + traceback.format_exc().encode())))
           errorm = 1
           '''
           print(code_string)
           exec(code_string)
           if (errorm == 1):
               return 0
           else:
               return 1
       except Exception as e:
        # Capture and return the error message
           errormessage[0] = lib.strdup(ffi.from_buffer(memoryview(b'UDFError '+ funcname[0].encode()+b'.'+funcname[1].encode()+b'.'+funcname[2].upper().encode() + b': ' + traceback.format_exc().encode())))
           return 0

    @ffi.def_extern()
    def myfree(array):
        lib.free(array)

    @ffi.def_extern()
    def scalarfusionwrapper(funcnames, nfuncs, paramsnum, arrays, resultdouble, resultint, resultstring, insize, isliteral):
       global global_lock
       with global_lock:
           fnames = [None]*nfuncs
           for i in range(nfuncs):
               tmpfuncname = ffi.string(funcnames[i]).decode()
               fnames[i] = tmpfuncname.split('.')
           inputd = [None]*paramsnum
           for i in range(paramsnum):
             if arrays[i].type == 0:
                 inputd[i] = ffi.cast("char **", arrays[i].array )
             elif arrays[i].type == 1:
                 inputd[i] = ffi.cast("int *", arrays[i].array )
             elif arrays[i].type == 2:
                 inputd[i] = ffi.cast("double *", arrays[i].array)
           if resultstring != ffi.NULL:
               resultparam = 'resultstring'
           elif resultint != ffi.NULL:
               resultparam = 'resultint'
           elif resultdouble != ffi.NULL:
               resultparam = 'resultdouble'
           code_params = ', '.join(['inputd['+str(j)+'][n]' if arrays[j].type != 0 else 'ffi.string(ffi.cast("char **", arrays['+str(j)+'].array )[n])' for j in range(paramsnum)])
           importcode = '\\n    '.join([f'from {fnames[j][0]} import {fnames[j][1]}' for j in range(nfuncs)])
           reloadcode = '\\n    '.join([f'importlib.reload({fnames[j][0]})' for j in range(nfuncs)])
           fused_code_call = ''.join(f"{v2}.{v3}(" for _, v2, v3 in fnames) + code_params + ')' * nfuncs
           fused_code_call_inside = ''.join(f"fnames[{str(i)}](" for i, v in enumerate(fnames)) + code_params + ')' * nfuncs
           fnamesembed = ','.join(f"{v2}.{v3}" for _, v2, v3 in fnames)
           if resultparam != 'resultstring':
               code_string = '''
    try:
        '''+importcode+'''
        for n in range(insize):
          '''+resultparam+'''[n] = '''+fused_code_call+'''
    except:
        '''+reloadcode+'''
        '''+importcode+'''
        for n in range(insize):
          '''+resultparam+'''[n] = '''+fused_code_call+'''

         '''
           elif isliteral == 0:
               code_string = '''
    try:
        '''+importcode+'''
    except:
        '''+reloadcode+'''
        '''+importcode+'''

    def tmpfunc(*args):
        arrays = args[0]
        insize = args[1]
        fnames = args[2:]
        return ['''+fused_code_call_inside+''' for n in range(insize)]
    tmpstrs = tmpfunc(arrays, insize, '''+fnamesembed+''')
    for n in range(insize):
        resultstring[n] = ffi.from_buffer(memoryview(tmpstrs[n]))
       #resultstring[n] = lib.strdup(ffi.from_buffer(memoryview('''+fused_code_call+'''))))
           '''
           elif isliteral == 1:
               code_string = '''
    try:
        '''+importcode+'''
    except:
         '''+reloadcode+'''
         '''+importcode+'''
    for n in range(insize):
        resultstring[n] = ffi.from_buffer(memoryview('''+fused_code_call+''')))
           '''
           print(code_string)
           exec(code_string)
           return 1


    @ffi.def_extern()
    def rowtablewrapper(funcname, paramsnum, arrays, resultnum, resultarrays, insize):
         global tmpstrs
         del tmpstrs[:]
         tmpstrs = []
         #global inc
         funcname = ffi.string(funcname).decode()
         funcname = funcname.split('.')
         inputd = [None]*paramsnum
         outputd = [None]*resultnum
         for i in range(paramsnum):
           if arrays[i].type == 0:
               inputd[i] = ffi.cast("char **", arrays[i].array )
           elif arrays[i].type == 1:
               inputd[i] = ffi.cast("int *", arrays[i].array )
           elif arrays[i].type == 2:
               inputd[i] = ffi.cast("double *", arrays[i].array)
         code_params = ', '.join(['int(inputd['+str(j)+'][n])' if arrays[j].type == 1 else 'ffi.string(ffi.cast("char **", arrays['+str(j)+'].array )[n])' if arrays[j].type == 0 else 'float(inputd['+str(j)+'][n])' for j in range(paramsnum)])
         retlen = insize*2
         for i in range(resultnum):
             if resultarrays[i].type == 0:
                 resultarrays[i].array = ffi.cast("char**", lib.malloc(retlen*ffi.sizeof("char *")))
                 outputd[i] = ffi.cast("char **", resultarrays[i].array )
             elif resultarrays[i].type == 1:
                 resultarrays[i].array = ffi.cast("int*", lib.malloc(retlen*ffi.sizeof("int")))
                 outputd[i] = ffi.cast("int *", resultarrays[i].array )
             elif resultarrays[i].type == 2:
                 resultarrays[i].array = ffi.cast("double*", lib.malloc(retlen*ffi.sizeof("double")))
                 outputd[i] = ffi.cast("double *", resultarrays[i].array)
         if resultnum > 1:
             codeloop = '\\n    '.join(['outputd['+str(i)+'][cc] = lib.strdup(ffi.from_buffer(memoryview(row['+str(i)+'])))' if resultarrays[i].type == 0 else 'outputd['+str(i)+'][cc] = row['+str(i)+']' for i in range(resultnum)])
         else:
             codeloop = 'outputd['+str(i)+'][cc] = lib.strdup(ffi.from_buffer(memoryview(row)))' if resultarrays[i].type == 0 else 'outputd['+str(i)+'][cc] = row'
         code_string = '''
    try:
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
    except:
        importlib.reload('''+funcname[0]+''')
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
    cc = -1
    for n in range(insize):
      for row in '''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+'''):
        cc += 1
        resultarrays[0].size = cc+1
        if cc >= retlen:
            retlen = retlen*2
            for i in range(resultnum):
             if resultarrays[i].type == 0:
                 resultarrays[i].array = ffi.cast("char**", lib.realloc(resultarrays[i].array,retlen*ffi.sizeof("char *")))
                 outputd[i] = ffi.cast("char **", resultarrays[i].array )
             elif resultarrays[i].type == 1:
                 resultarrays[i].array = ffi.cast("int*", lib.realloc(resultarrays[i].array, retlen*ffi.sizeof("int")))
                 outputd[i] = ffi.cast("int*", resultarrays[i].array )
             elif resultarrays[i].type == 2:
                 resultarrays[i].array = ffi.cast("double*", lib.realloc(resultarrays[i].array, retlen*ffi.sizeof("double")))
                 outputd[i] = ffi.cast("double", resultarrays[i].array )
        '''+codeloop

         print(code_string)
         exec(code_string)
         return 1

    @ffi.def_extern()
    def rowtuplewrapper(funcname, paramsnum, arrays, resultnum, resultarrays, insize):
         #global tmpstrs
         #del tmpstrs[:]
         #tmpstrs = []
         #global inc

         funcname = ffi.string(funcname).decode()
         funcname = funcname.split('.')
         inputd = [None]*paramsnum
         outputd = [None]*resultnum
         for i in range(paramsnum):
           if arrays[i].type == 0:
               inputd[i] = ffi.cast("char **", arrays[i].array )
           elif arrays[i].type == 1:
               inputd[i] = ffi.cast("int *", arrays[i].array )
           elif arrays[i].type == 2:
               inputd[i] = ffi.cast("double *", arrays[i].array)
         for i in range(resultnum):
           if resultarrays[i].type == 0:
               outputd[i] = ffi.cast("char **", resultarrays[i].array )
           elif resultarrays[i].type == 1:
               outputd[i] = ffi.cast("int *", resultarrays[i].array )
           elif resultarrays[i].type == 2:
               outputd[i] = ffi.cast("double *", resultarrays[i].array)
         code_params = ', '.join(['inputd['+str(j)+'][n]' if arrays[j].type != 0 else 'ffi.string(inputd['+str(j)+'][n])' for j in range(paramsnum)])
         code_string = '''
    try:
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
    except:
        importlib.reload('''+funcname[0]+''')
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
    #for n in range(insize):
    #    tmpstrs.append('''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+'''))
         '''
         print(code_string)
         exec(code_string)
         myvals = ','.join(['val'+str(n) for n in range(resultnum)])
         funccallcode = myvals+ ' ='+funcname[1]+'.'+funcname[2]+'('+code_params+')'
         codeloop = '\\n    '.join(['outputd['+str(i)+'][n] = lib.strdup(ffi.from_buffer(memoryview(val'+str(i)+')))' if resultarrays[i].type == 0 else 'outputd['+str(i)+'][n] = val'+str(i)+'' for i in range(resultnum)])
         code_string = '''
    for n in range(insize):
        '''+funccallcode+'''
        '''+codeloop+'''
         '''
         print(code_string)
         exec(code_string)
         return 1

    @ffi.def_extern()
    def tablematerwrapper(paramsnum, arrays, resultnum, resultarrays, insize):
         inputd = [None]*paramsnum
         import functions
         outputd = [None]*resultnum
         for i in range(paramsnum):
           if arrays[i].type == 0:
               inputd[i] = ffi.cast("char **", arrays[i].array )
           elif arrays[i].type == 1:
               inputd[i] = ffi.cast("int *", arrays[i].array )
           elif arrays[i].type == 2:
               inputd[i] = ffi.cast("double *", arrays[i].array)
         import uuid

         results = [None]*paramsnum
         for i in range(paramsnum):
             if arrays[i].type != 0:
                 results[i] = ffi.unpack(inputd[i], insize)
             else:
                 results[i] = [ffi.string(inputd[i][x]) for x in range(insize)]

         resultarrays[0].array = ffi.cast("char**", lib.malloc(1*ffi.sizeof("char *")))
         outputd[0] = ffi.cast("char **", resultarrays[0].array )
         resultarrays[0].size = 1
         uid = uuid.uuid4()
         functions.results[str(uid.bytes)] = results
         outputd[0][0] = lib.strdup(ffi.from_buffer(memoryview(str(uid.bytes).encode('utf-8'))))
         return 1

    @ffi.def_extern()
    def expandwrapper(funcname, paramsnum, arrays, resultnum, resultarrays, genindices, lengenindices, insize, udfargs ,errormessage):
      inputd = [None]*paramsnum
      outputd = [None]*resultnum
      funcname = ffi.string(funcname).decode()
      funcname = funcname.split('.')
      if lengenindices == 2:
          for i in range(paramsnum):
              if arrays[i].type == 0:
                  inputd[i] = ffi.cast("char **", arrays[i].array )
              elif arrays[i].type == 1:
                  inputd[i] = ffi.cast("int*", arrays[i].array )
              elif arrays[i].type == 2:
                  inputd[i] = ffi.cast("double *", arrays[i].array)
              elif arrays[i].type == 4:
                 inputd[i] = ffi.cast("int8_t *", arrays[i].array)
              elif arrays[i].type == 5:
                 inputd[i] = ffi.cast("int16_t *", arrays[i].array)
          retlen = insize+1000000*2
          for i in range(resultnum):
               if resultarrays[i].type == 0:
                   resultarrays[i].array = lib.malloc(retlen * ffi.sizeof("char*"))
                   outputd[i] = ffi.cast("char **", resultarrays[i].array)
               elif resultarrays[i].type == 1:
                   resultarrays[i].array = ffi.cast("int *", lib.malloc(retlen*ffi.sizeof("int")))
                   outputd[i] = ffi.cast("int *", resultarrays[i].array )
               elif resultarrays[i].type == 2:
                   resultarrays[i].array = ffi.cast("double*", lib.malloc(retlen*ffi.sizeof("double")))
                   outputd[i] = ffi.cast("double *", resultarrays[i].array)
          cc = -1
          try:
              code_params = ', '.join(['inputd['+str(j)+'][n]' if arrays[j].type != 0 else 'ffi.string(ffi.cast("char **", arrays['+str(j)+'].array )[n])' for j in range(genindices[0], genindices[0]+udfargs[0])])
          except:
              code_params = ''
          #code_params = ', '.join(['2' if arrays[j].type != 0 else 'ffi.string(ffi.cast("char **", arrays['+str(j)+'].array )[n])' for j in range(genindices[0], genindices[0]+udfargs[0])])
          code_loop_iterator = '\\n              '.join(['outputd['+str(i)+'][cc] = lib.strdup(ffi.from_buffer(memoryview(val'+str(kk)+')))' if resultarrays[i].type == 0 else 'outputd['+str(i)+'][cc] = val'+str(kk)+'' for kk, i in enumerate(range(genindices[0], genindices[0]+genindices[1]))])
          code_loop_rest_prev = '\\n              '.join(['outputd['+str(i)+'][cc] = inputd['+str(i)+'][n]'  for i in range(genindices[0])])
          code_loop_rest_next = '\\n              '.join(['outputd['+str(i)+'][cc] = inputd['+str(i-genindices[1]+udfargs[0])+'][n]'  for i in range(genindices[0]+genindices[1], resultnum)])
          strdups = {}
          iterids = [kk for kk in range(genindices[0], genindices[0]+genindices[1])]
          for id in iterids:
              strdups[id] = True
          code_string = '''
    try:
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
        if functions.DEBUG:
          mypath = get_module_path(funcname[0]+'.'+funcname[1])
          if mypath not in checksums:
            checksums[mypath] = calculate_checksum(mypath)
          elif mypath in checksums:
            csum = calculate_checksum(mypath)
            if checksums[mypath] != csum:
              importlib.reload('''+funcname[1]+''')
              checksums[mypath] = csum
    except ImportError as import_err:
        importlib.reload('''+funcname[0]+''')
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
    try:
      resultarrays[0].size = 0
      for n in range(insize):
               #for '''+','.join(['val'+str(x) for x in range(genindices[1])])+''' in iters[ffi.string(inputd['''+str(genindices[0])+'''][n])]:
               for '''+','.join(['val'+str(x) for x in range(genindices[1])])+'''  in '''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+'''): #ffi.string(inputd['''+str(genindices[0])+'''][n])]:
                  cc += 1
                  resultarrays[0].size = cc+1
                  if cc >= retlen:
                    retlen = retlen*2
                    #print('double realloc: ',cc)
                    for i in range(resultnum):
                      if resultarrays[i].type == 0:
                          resultarrays[i].array = lib.realloc(resultarrays[i].array, (retlen+1) * ffi.sizeof("char*"))
                          # Wrap the allocated memory with ffi.gc to ensure it is freed
                          #resultarrays[i].array = ffi.gc(ffi.cast("char**", resultarrays[i].array), free_strings, size=retlen)
                          outputd[i] = ffi.cast("char **", resultarrays[i].array)
                      elif resultarrays[i].type == 1:
                          resultarrays[i].array = ffi.cast("int*", lib.realloc(resultarrays[i].array, retlen*ffi.sizeof("int")))
                          outputd[i] = ffi.cast("int*", resultarrays[i].array )
                      elif resultarrays[i].type == 2:
                          resultarrays[i].array = ffi.cast("double*", lib.realloc(resultarrays[i].array, retlen*ffi.sizeof("double")))
                          outputd[i] = ffi.cast("double*", resultarrays[i].array )
                  '''+code_loop_iterator+'''
                  '''+code_loop_rest_prev+'''
                  '''+code_loop_rest_next+'''
                  #del iters[ffi.string(inputd['''+str(genindices[0])+'''][n])]
                  #outputd[0][cc] = lib.strdup(ffi.from_buffer(memoryview(prev)))
                  #outputd[1][cc] = lib.strdup(ffi.from_buffer(memoryview(middle.encode())))
                  #outputd[2][cc] = lib.strdup(ffi.from_buffer(memoryview(next.encode())))
                  ##outputd[0][cc] = inputd[0][n]
                  ##outputd[1][cc] = inputd[1][n]
      for i in range(resultnum):
                      #print('all: ', i)
                      if resultarrays[i].type == 0 and i in strdups:
                          #print('dups: ', i)
                          #resultarrays[i].array = ffi.gc(ffi.cast("char**", resultarrays[i].array), lib.custom_free, size=resultarrays[0].size)
                          try:
                              __pypy__.add_memory_pressure(retlen*2*ffi.sizeof("char*"))
                              resultarrays[i].array = ffi.gc(ffi.cast("char**", resultarrays[i].array), lib.custom_free, size=resultarrays[0].size)
                          except:
                              pass
                          outputd[i] = ffi.cast("char **", resultarrays[i].array)
                          outputd[i][resultarrays[0].size] = ffi.NULL
    except Exception as e:
          errormessage[0] = lib.strdup(ffi.from_buffer(memoryview(b'UDFError '+ funcname[0].encode()+b'.'+funcname[1].encode()+b'.'+funcname[2].upper().encode() + b': ' + traceback.format_exc().encode())))
          '''
          print(code_string)
          exec(code_string)
          return 1
      if lengenindices > 2:
          errormessage[0] = lib.strdup(ffi.from_buffer(memoryview(b'UDFError '+ funcname[0].encode()+b'.'+funcname[1].encode()+b'.'+funcname[2].upper().encode() + b': Multiple multiset UDFs in the same projection is not supported yet')))
          pass     ## TODO cross join - 2 OR MORE MULTISET UDFs  NOT SUPPORTED YET in a single projection
      return 1



    @ffi.def_extern()
    def expandaggrwrapper(funcname, paramsnum, arrays, resultnum, resultarrays, genindices, lengenindices, insize, udfargs, errormessage):
      inputd = [None]*paramsnum
      outputd = [None]*resultnum
      funcname = ffi.string(funcname).decode()
      funcname = funcname.split('.')
      #print('lala: ', arrays[0].array[0] , arrays[0].array[1])
      if lengenindices == 2:
          for i in range(paramsnum):
              if arrays[i].type == 0:
                  inputd[i] = ffi.cast("char **", arrays[i].array )
              elif arrays[i].type == 1:
                  inputd[i] = ffi.cast("int*", arrays[i].array )
              elif arrays[i].type == 2:
                  inputd[i] = ffi.cast("double *", arrays[i].array)
              elif arrays[i].type == 4:
                 inputd[i] = ffi.cast("int8_t *", arrays[i].array)
              elif arrays[i].type == 5:
                 inputd[i] = ffi.cast("int16_t *", arrays[i].array)
          retlen = insize+1000000*2
          for i in range(resultnum):
               if resultarrays[i].type == 0:
                   resultarrays[i].array = ffi.cast("char**", lib.malloc(retlen*ffi.sizeof("double")))
                   outputd[i] = ffi.cast("char **", resultarrays[i].array )
               elif resultarrays[i].type == 1:
                   resultarrays[i].array = ffi.cast("int *", lib.malloc(retlen*ffi.sizeof("double")))
                   outputd[i] = ffi.cast("int *", resultarrays[i].array )
               elif resultarrays[i].type == 2:
                   resultarrays[i].array = ffi.cast("float*", lib.malloc(retlen*ffi.sizeof("double")))
                   outputd[i] = ffi.cast("double *", resultarrays[i].array)
          cc = -1
          try:
              code_params = ', '.join(['inputd['+str(j)+'][n]' if arrays[j].type != 0 else 'ffi.string(ffi.cast("char **", arrays['+str(j)+'].array )[n])' for j in range(genindices[0], genindices[0]+udfargs[0])])
          except:
              code_params = ''

          #code_params = ', '.join(['2' if arrays[j].type != 0 else 'ffi.string(ffi.cast("char **", arrays['+str(j)+'].array )[n])' for j in range(genindices[0], genindices[0]+udfargs[0])])
          code_loop_iterator = '\\n              '.join(['outputd['+str(i)+'][cc] = lib.strdup(ffi.from_buffer(memoryview(val'+str(kk)+')))' if resultarrays[i].type == 0 else 'outputd['+str(i)+'][cc] = val'+str(kk)+'' for kk, i in enumerate(range(genindices[0], genindices[0]+genindices[1]))])
          code_loop_rest_prev = '\\n              '.join(['outputd['+str(i)+'][cc] = inputd['+str(i)+'][n]'  for i in range(genindices[0])])
          code_loop_rest_next = '\\n              '.join(['outputd['+str(i)+'][cc] = inputd['+str(i-genindices[1]+udfargs[0])+'][n]'  for i in range(genindices[0]+genindices[1], resultnum)])
          code_string = '''
    try:
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
        if functions.DEBUG:
          mypath = get_module_path(funcname[0]+'.'+funcname[1])
          #print(mypath)
          if mypath not in checksums:
            checksums[mypath] = calculate_checksum(mypath)
          elif mypath in checksums:
            csum = calculate_checksum(mypath)
            if checksums[mypath] != csum:
              importlib.reload('''+funcname[1]+''')
              checksums[mypath] = csum
    except ImportError as import_err:
        importlib.reload('''+funcname[0]+''')
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
    try:
        resultarrays[0].size = 0
        for n in range(insize):
               for '''+','.join(['val'+str(x) for x in range(genindices[1])])+''' in iters[inputd['''+str(genindices[0])+'''][n]]:
               #for '''+','.join(['val'+str(x) for x in range(genindices[1])])+'''  in '''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+'''): #ffi.string(inputd['''+str(genindices[0])+'''][n])]:
                  cc += 1
                  resultarrays[0].size = cc+1
                  if cc >= retlen:
                    retlen = retlen*2
                    for i in range(resultnum):
                      if resultarrays[i].type == 0:
                          resultarrays[i].array = ffi.cast("char**", lib.realloc(resultarrays[i].array,retlen*ffi.sizeof("char *")))
                          outputd[i] = ffi.cast("char **", resultarrays[i].array )
                      elif resultarrays[i].type == 1:
                          resultarrays[i].array = ffi.cast("int*", lib.realloc(resultarrays[i].array, retlen*ffi.sizeof("int")))
                          outputd[i] = ffi.cast("int*", resultarrays[i].array )
                      elif resultarrays[i].type == 2:
                          resultarrays[i].array = ffi.cast("double*", lib.realloc(resultarrays[i].array, retlen*ffi.sizeof("double")))
                          outputd[i] = ffi.cast("double*", resultarrays[i].array )
                  '''+code_loop_iterator+'''
                  '''+code_loop_rest_prev+'''
                  '''+code_loop_rest_next+'''
               del iters[inputd['''+str(genindices[0])+'''][n]]
                  #outputd[0][cc] = lib.strdup(ffi.from_buffer(memoryview(prev)))
                  #outputd[1][cc] = lib.strdup(ffi.from_buffer(memoryview(middle.encode())))
                  #outputd[2][cc] = lib.strdup(ffi.from_buffer(memoryview(next.encode())))
                  ##outputd[0][cc] = inputd[0][n]
                  ##outputd[1][cc] = inputd[1][n]
    except Exception as e:
          errormessage[0] = lib.strdup(ffi.from_buffer(memoryview(b'UDFError '+ funcname[0].encode()+b'.'+funcname[1].encode()+b'.'+funcname[2].upper().encode() + b': ' + traceback.format_exc().encode())))
          '''
          print(code_string)
          exec(code_string)
          return 1
      if lengenindices > 2:
          errormessage[0] = lib.strdup(ffi.from_buffer(memoryview(b'UDFError '+ funcname[0].encode()+b'.'+funcname[1].encode()+b'.'+funcname[2].upper().encode() + b': Multiple multiset UDFs in the same projection is not supported yet')))
          pass     ## TODO cross join - 2 OR MORE MULTISET UDFs  NOT SUPPORTED YET in a single projection
      return 1


    @ffi.def_extern()
    def aggregatewrapper(funcname, inputcount, paramsnum, arrays, resultdouble, resultint, resultstring, groups, count_per_group, errormessage):
        global iters
        errorm = 0
        #print('fofo: ',inputcount, groups)
        funcname = ffi.string(funcname).decode()
        funcname = funcname.split('.')
        li = [None]*paramsnum
        types = [None]*paramsnum
        #for c in range(groups):
        #    print(count_per_group[c], inputcount)
        for i in range(paramsnum):
            if arrays[i].type == 0:
                li[i] = ffi.cast("char ***", arrays[i].array )
            elif arrays[i].type == 1:
                li[i] = ffi.cast("int **", arrays[i].array )
            elif arrays[i].type == 2:
                li[i] = ffi.cast("double **", arrays[i].array)
            elif arrays[i].type == 4:
                li[i] = ffi.cast("int8_t **", arrays[i].array)
            elif arrays[i].type == 5:
                li[i] = ffi.cast("int16_t **", arrays[i].array)
        #print('lala: ', sum([li[1][0][x] for x in range(inputcount)]) )
        if resultstring != ffi.NULL:
            resultparam = 'resultstring'
        elif resultint != ffi.NULL:
            resultparam = 'resultint'
        elif resultdouble != ffi.NULL:
            resultparam = 'resultdouble'
        code_params = ', '.join(['ffi.string(li['+str(j)+'][grp][col])' if arrays[j].type == 0 else 'li['+str(j)+'][grp][col]' for j in range(paramsnum)])
        if resultparam != 'resultstring':
            code_string = '''
    try:
     from '''+funcname[0]+''' import '''+funcname[1]+'''
     if functions.DEBUG:
          mypath = get_module_path(funcname[0]+'.'+funcname[1])
          #print(mypath)
          if mypath not in checksums:
            checksums[mypath] = calculate_checksum(mypath)
          elif mypath in checksums:
            csum = calculate_checksum(mypath)
            if checksums[mypath] != csum:
              importlib.reload('''+funcname[1]+''')
              checksums[mypath] = csum
     if inspect.isgeneratorfunction('''+funcname[1]+'''.'''+funcname[2]+'''.final):
      #print('is a generator')
      for grp in range(groups):
        aggr='''+funcname[1]+'''.'''+funcname[2]+'''()
        for col in range(count_per_group[grp]):
            aggr.step('''+code_params+''')
        func = aggr.final()
        myhash = (hash(str(func)) % (10**9))/(10**9)
        iters[myhash] = func
        '''+resultparam+'''[grp] = myhash

     else:
      for grp in range(groups):
        aggr='''+funcname[1]+'''.'''+funcname[2]+'''()
        for col in range(count_per_group[grp]):
            aggr.step('''+code_params+''')
        '''+resultparam+'''[grp] = aggr.final()
    except Exception as e:
        errormessage[0] = lib.strdup(ffi.from_buffer(memoryview(b'UDFError '+ funcname[0].encode()+b'.'+funcname[1].encode()+b'.'+funcname[2].upper().encode() + b': ' + traceback.format_exc().encode())))
        errorm = 1
        '''
        else:
            code_string = '''
    try:
     from '''+funcname[0]+''' import '''+funcname[1]+'''
     if functions.DEBUG:
          mypath = get_module_path(funcname[0]+'.'+funcname[1])
          #print(mypath)
          if mypath not in checksums:
            checksums[mypath] = calculate_checksum(mypath)
          elif mypath in checksums:
            csum = calculate_checksum(mypath)
            if checksums[mypath] != csum:
              importlib.reload('''+funcname[1]+''')
              checksums[mypath] = csum
     for grp in range(groups):
        aggr = '''+funcname[1]+'''.'''+funcname[2]+'''()
        for col in range(count_per_group[grp]):
            aggr.step('''+code_params+''')
        resultstring[grp] = ffi.from_buffer(memoryview(aggr.final()))
    except Exception as e:
        errormessage[0] = lib.strdup(ffi.from_buffer(memoryview(b'UDFError '+ funcname[0].encode()+b'.'+funcname[1].encode()+b'.'+funcname[2].upper().encode() + b': ' + traceback.format_exc().encode())))
        errorm = 1
        '''
        print (code_string)
        exec(code_string)
        if errorm == 1:
            return 0
        else:
            return 1


    @ffi.def_extern()
    def numpyudfwrapper(funcname, paramsnum, arrays, resultnum, resultarrays, insize, extraparams, errormessage):
        inputd = [None]*paramsnum
        errorm = 0
        outputd = [None]*resultnum
        funcname = ffi.string(funcname).decode()
        funcname = funcname.split('.')
        inputtype = ""
        nptype = ""
        for i in range(paramsnum):
           if arrays[i].type == 0:
               inputd[i] = ffi.cast("char **", arrays[i].array )
               inputtype = "char *"
               nptype = "str"
           elif arrays[i].type == 1:
               inputd[i] = ffi.cast("int *", arrays[i].array )
               inputtype = "int"
               nptype = np.int32
           elif arrays[i].type == 2:
               inputd[i] = ffi.cast("double *", arrays[i].array)
               inputtype = "double"
               nptype = np.float64
        if paramsnum>1:
            #combined_array = ffi.cast(inputtype+" *", lib.malloc(insize*paramsnum * ffi.sizeof(inputtype)))
            combined_array = ffi.new(inputtype+" []", insize*paramsnum)
            # Use memmove to copy data from float_pointer1 to the new array
            for i in range(paramsnum):
                ffi.memmove(combined_array+i*insize, arrays[i].array, insize * ffi.sizeof(inputtype))

            numpy_array = np.frombuffer(ffi.buffer(combined_array, ffi.sizeof(inputtype)*insize*paramsnum), dtype=nptype)
            data = numpy_array.reshape(paramsnum, -1)
        else:
            data = np.frombuffer(ffi.buffer(inputd[0], ffi.sizeof(inputtype)*insize*paramsnum), dtype=nptype)
        code_string = '''
    try:
        from '''+funcname[0]+''' import '''+funcname[1]+'''
        if functions.DEBUG:
          mypath = get_module_path(funcname[0]+'.'+funcname[1])
          #print(mypath)
          if mypath not in checksums:
            checksums[mypath] = calculate_checksum(mypath)
          elif mypath in checksums:
            csum = calculate_checksum(mypath)
            if checksums[mypath] != csum:
              importlib.reload('''+funcname[1]+''')
              checksums[mypath] = csum
        result = '''+funcname[1]+'''.'''+funcname[2]+'''(data)
        result = next(result)
        if resultnum==1:
            retlen = result.size
        else:
            retlen = result[0].size
        resultarrays[0].size = retlen
        if resultnum == 1 and retlen == 1:
          for i in range(resultnum):
             if resultarrays[i].type == 0:
                 resultarrays[i].array = ffi.cast("char**", lib.malloc(retlen*ffi.sizeof("char *")))
                 outputd[i] = ffi.cast("char **", resultarrays[i].array )
             elif resultarrays[i].type == 1:
                 resultarrays[i].array = ffi.cast("int*", lib.malloc(retlen*ffi.sizeof("int")))
                 outputd[i] = ffi.cast("int *", resultarrays[i].array )
             elif resultarrays[i].type == 2:
                 resultarrays[i].array = ffi.cast("double*", lib.malloc(retlen*ffi.sizeof("double")))
                 outputd[i] = ffi.cast("double *", resultarrays[i].array)
        if resultnum == 1 and retlen == 1:
           outputd[0][0] = ffi.from_buffer(result[0])
           #resultarrays[0].array[0] = ffi.from_buffer(result)
        elif resultnum == 1:
            resultarrays[0].array = ffi.from_buffer(result)
        else:
          for r in range(resultnum):
            resultarrays[r].array = ffi.from_buffer(result[r])
    except Exception as e:
        errormessage[0] = lib.strdup(ffi.from_buffer(memoryview(b'UDFError '+ funcname[0].encode()+b'.'+funcname[1].encode()+b'.'+funcname[2].upper().encode() + b': ' + traceback.format_exc().encode())))
        errorm = 1
        '''
        print (code_string)
        exec(code_string)

        #lib.free(combined_array)
        #del combined_array
        #del data
        #del numpy_array
        if errorm == 1:
            return 0
        else:
            return 1





    @ffi.def_extern()
    def numpyudfwrapperwithtests(funcname, paramsnum, arrays, resultnum, resultarrays, insize, extraparams):
        inputd = [None]*paramsnum
        outputd = [None]*resultnum
        funcname = ffi.string(funcname).decode()
        funcname = funcname.split('.')
        for i in range(paramsnum):
           if arrays[i].type == 0:
               inputd[i] = ffi.cast("char **", arrays[i].array )
           elif arrays[i].type == 1:
               inputd[i] = ffi.cast("int *", arrays[i].array )
           elif arrays[i].type == 2:
               inputd[i] = ffi.cast("double *", arrays[i].array)
        retlen = 1
        resultarrays[0].size = 1
        for i in range(resultnum):
             if resultarrays[i].type == 0:
                 resultarrays[i].array = ffi.cast("char**", lib.malloc(retlen*ffi.sizeof("char *")))
                 outputd[i] = ffi.cast("char **", resultarrays[i].array )
             elif resultarrays[i].type == 1:
                 resultarrays[i].array = ffi.cast("int*", lib.malloc(retlen*ffi.sizeof("int")))
                 outputd[i] = ffi.cast("int *", resultarrays[i].array )
             elif resultarrays[i].type == 2:
                 resultarrays[i].array = ffi.cast("double*", lib.malloc(retlen*ffi.sizeof("double")))
                 outputd[i] = ffi.cast("double *", resultarrays[i].array)

        outputd[0][0] = pearson_correlation(inputd[0],inputd[1],insize)
        return 1
        #arrays[i].array = ffi.cast("double*", lib.realloc(arrays[i].array, insize*paramsnum*ffi.sizeof("double")))
        #inputd[i] = ffi.cast("double", arrays[i].array )
        #inputd[0] = lib.realloc(inputd[0],insize*paramsnum*ffi.sizeof("float"))
        #combined_array = ffi.new("float[]", insize*2)
        float_size = insize
        #combined_array = ffi.new("float[]",1)
        #combined_array = ffi.cast("float*", lib.malloc(insize*2 * ffi.sizeof("float")))
        #combined_array = ffi.new("float[]", float_size * 2)

        # Use memmove to copy data from float_pointer1 to the new array
        #ffi.memmove(combined_array, arrays[0].array, float_size * ffi.sizeof("float"))

        # Use memmove to copy data from float_pointer2 to the new array
        #ffi.memmove(ffi.cast("char*", combined_array) + float_size * ffi.sizeof("float"), arrays[1].array, float_size * ffi.sizeof("float"))

        #for i in range(insize*2):
        #    if i<insize:
        #        combined_array[i] = inputd[0][i]
        #    else:
        #        combined_array[i] = inputd[1][i-insize]

        #combined_array[0:insize] = [inputd[0][x] for x in range(insize)]
        #combined_array[insize:insize*2] = [inputd[1][x] for x in range(insize)]
        #inputd[0][insize:insize*2] = inputd[1]
        #print('kiki')
        #numpy_array = np.frombuffer(ffi.buffer(combined_array), dtype=np.float32)
        #outputd[0][0] = -1
        #return 1
        numpy_array1 = np.frombuffer(ffi.buffer(arrays[0].array, insize*8), dtype=np.float64)
        numpy_array2 = np.frombuffer(ffi.buffer(arrays[1].array, insize*8), dtype=np.float64)
        numpy_array3 = numpy_array1*2
        numpy_array4 = numpy_array2*2
        #data = numpy_array.reshape(2, -1)
        #print(np.corrcoef(data[0], data[1])[0, 1])
        #outputd[0][0] = np.corrcoef(data[0], data[1])[0, 1]
        outputd[0][0] = np.corrcoef(numpy_array3, numpy_array4)[0, 1]
        #ffi.free(combined_array)
        #lib.free(combined_array)
        #del combined_array
        #del data
        del numpy_array
        return 1


    @ffi.def_extern()
    def fusionwrapper(fusiblecomponent, nops, paramsnum, arrays, resultdouble, resultint, resultstring, resultnum, resultarrays, insize, extraparams, fusedudftype, isliteral):
        inputd = [None]*paramsnum
        outputd = [None]*resultnum
        for i in range(paramsnum):
           if arrays[i].type == 0:
               inputd[i] = ffi.cast("char **", arrays[i].array )
           elif arrays[i].type == 1:
               inputd[i] = ffi.cast("int *", arrays[i].array )
           elif arrays[i].type == 2:
               inputd[i] = ffi.cast("double *", arrays[i].array)
        if fusedudftype == 1:
            scalarfusionwrapper(fusiblecomponent, nops, paramsnum, arrays, resultdouble, resultint, resultstring, insize, isliteral)
        if fusedudftype == 2:
            ## aggregate case (cases with aggregates and scalars)
            pass
        if fusedudftype == 3:
            # table case, all the other cases
            pass
        return 1

    @ffi.def_extern()
    def systemwrapper(funcname, paramsnum, arrays, resultnum, resultarrays, insize, extraparams, errormessage):
        funcname = ffi.string(funcname).decode()
        funcname = funcname.split('.')
        inputd = [None]*paramsnum
        for i in range(paramsnum):
           if arrays[i].type == 0:
               inputd[i] = ffi.cast("char **", arrays[i].array )
           elif arrays[i].type == 1:
               inputd[i] = ffi.cast("int *", arrays[i].array )
           elif arrays[i].type == 2:
               inputd[i] = ffi.cast("double *", arrays[i].array)
        codeparams = 'inputd, resultnum, resultarrays, insize, ffi, lib'
        code_string = '''
    try:
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
    except ImportError as import_err:
        importlib.reload('''+funcname[0]+''')
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
    iter = '''+funcname[1]+'''.'''+funcname[2]+'''('''+codeparams+''')
    next(iter)
        '''
        print(code_string)
        exec(code_string)
        return 1

    @ffi.def_extern()
    def tableudfwrapper(funcname, paramsnum, arrays, resultnum, resultarrays, insize, extraparams, errormessage):
      global global_lock
      errorm = 0
      with global_lock:
         errorm = 0
         global tmpstrs
         del tmpstrs[:]
         tmpstrs = []
         #global inc
         funcname = ffi.string(funcname).decode()
         funcname = funcname.split('.')
         inputd = [None]*paramsnum
         outputd = [None]*resultnum
         scparams = None
         start = 0
         extrapar = 0
         try:
             if arrays[0].type == 0:
                 scparams = json.loads(ffi.string(ffi.cast("char **", arrays[0].array )[0]).decode().replace("'",'"')  )['scpar']
             else:
                 raise
         except:
             try:
                 if extraparams != ffi.NULL:
                     scparams = json.loads(ffi.string(extraparams).decode().replace("''",'"').strip("'")  )['scpar']
                     extrapar = 1
                 else:
                     scparams = None
             except:
                 scparams = None
         if scparams!=None and extrapar == 0:
             start = 1
         for i in range(start, paramsnum):
           if arrays[i].type == 0:
               inputd[i] = ffi.cast("char **", arrays[i].array )
           elif arrays[i].type == 1:
               inputd[i] = ffi.cast("int *", arrays[i].array )
           elif arrays[i].type == 2:
               inputd[i] = ffi.cast("double *", arrays[i].array)
           elif arrays[i].type == 4:
                 inputd[i] = ffi.cast("int8_t *", arrays[i].array)
           elif arrays[i].type == 5:
                 inputd[i] = ffi.cast("int16_t *", arrays[i].array)
         code_params = ', '.join(['int(inputd['+str(j)+'][n])' if arrays[j].type == 1 else 'ffi.string(inputd['+str(j)+'][n])' if arrays[j].type == 0 else 'float(inputd['+str(j)+'][n])' for j in range(start, paramsnum)])
         retlen = insize*2
         #for i in range(resultnum):
         #    if resultarrays[i].type == 0:
         #        resultarrays[i].array = ffi.cast("char**", lib.malloc(retlen*ffi.sizeof("char *")))
         #        outputd[i] = ffi.cast("char **", resultarrays[i].array )
         #    elif resultarrays[i].type == 1:
         #        resultarrays[i].array = ffi.cast("int*", lib.malloc(retlen*ffi.sizeof("int")))
         #        outputd[i] = ffi.cast("int *", resultarrays[i].array )
         #    elif resultarrays[i].type == 2:
         #        resultarrays[i].array = ffi.cast("double*", lib.malloc(retlen*ffi.sizeof("double")))
         #        outputd[i] = ffi.cast("double *", resultarrays[i].array)
         if resultnum > 1:
             codeloop = '\\n    '.join(['outputd['+str(i)+'][cc] = lib.strdup(ffi.from_buffer(memoryview(row['+str(i)+'])))' if resultarrays[i].type == 0 else 'outputd['+str(i)+'][cc] = row['+str(i)+']' for i in range(resultnum)])
         else:
             codeloop = 'outputd[0][cc] = lib.strdup(ffi.from_buffer(memoryview(row)))' if resultarrays[0].type == 0 else 'outputd[0][cc] = row'  ## to do type == 0
         code_string = '''
    try:
      try:
        #importlib.reload('''+funcname[0]+''')
        from  '''+funcname[0]+''' import '''+funcname[1]+'''
        if functions.DEBUG:
          mypath = get_module_path(funcname[0]+'.'+funcname[1])
          print(mypath)
          if mypath not in checksums:
            checksums[mypath] = calculate_checksum(mypath)
          elif mypath in checksums:
            csum = calculate_checksum(mypath)
            if checksums[mypath] != csum:
              importlib.reload('''+funcname[1]+''')
              checksums[mypath] = csum
      except ImportError as import_err:
        importlib.reload('''+funcname[0]+''')
        from  '''+funcname[0]+''' import '''+funcname[1]+'''

      cc = -1
      def inputgen(inputd, insize, arrays):
        for n in range(insize):
            yield '''+code_params+'''
      def inputgen2(scparams):
          #if len(scparams) == 1:
          #    yield scparams[0]
          #else:
              yield tuple(x for x in scparams)
      #for row in '''+funcname[1]+'''.'''+funcname[2]+'''(inputgen(inputd, insize, arrays)):
      ll = None
      if insize>0:
          if insize == 1 and scparams == None:
            if paramsnum == 1:
               ll = '''+funcname[1]+'''.'''+funcname[2]+'''(next(inputgen(inputd, insize, arrays))) #scparams  ))
            else:
               ll = '''+funcname[1]+'''.'''+funcname[2]+'''(*next(inputgen(inputd, insize, arrays))) #scparams  ))
          else:
            ll = '''+funcname[1]+'''.'''+funcname[2]+'''(inputgen(inputd, insize, arrays), scparams) #scparams  ))
      else:
          ll = '''+funcname[1]+'''.'''+funcname[2]+'''(*next(inputgen2(scparams)))
      #lll = list(ll)
      #lll = list('''+funcname[1]+'''.'''+funcname[2]+'''(inputd, insize, arrays, scparams)) #scparams  )) #pass materialized array
      resultarrays[0].size = insize*2+1000000 #len(lll)
      retlen = resultarrays[0].size
      for i in range(resultnum):
             if resultarrays[i].type == 0:
                 resultarrays[i].array = ffi.cast("char**", lib.malloc(retlen*ffi.sizeof("char *")))
                 #resultarrays[i].array = ffi.gc(ffi.cast("char**", resultarrays[i].array), lib.custom_free2, size=resultarrays[0].size)
                 outputd[i] = ffi.cast("char **", resultarrays[i].array )
             elif resultarrays[i].type == 1:
                 resultarrays[i].array = ffi.cast("int*", lib.malloc(retlen*ffi.sizeof("int")))
                 #resultarrays[i].array = ffi.gc(ffi.cast("int*", resultarrays[i].array), lib.custom_free3, size=resultarrays[0].size)
                 outputd[i] = ffi.cast("int *", resultarrays[i].array )
             elif resultarrays[i].type == 2:
                 resultarrays[i].array = ffi.cast("double*", lib.malloc(retlen*ffi.sizeof("double")))
                 outputd[i] = ffi.cast("double *", resultarrays[i].array)
      cc = -1
      for row in ll:
      #for row in lll:
      #for n in range(insize):
      #for row in '''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+'''):
        cc += 1
        resultarrays[0].size = cc+1
        if cc >= retlen:
          retlen = retlen*2
          for i in range(resultnum):
             if resultarrays[i].type == 0:
                 resultarrays[i].array = ffi.cast("char**", lib.realloc(resultarrays[i].array,(retlen+1)*ffi.sizeof("char *")))
                 outputd[i] = ffi.cast("char **", resultarrays[i].array )
             elif resultarrays[i].type == 1:
                 resultarrays[i].array = ffi.cast("int*", lib.realloc(resultarrays[i].array, retlen*ffi.sizeof("int")))
                 outputd[i] = ffi.cast("int*", resultarrays[i].array )
             elif resultarrays[i].type == 2:
                 resultarrays[i].array = ffi.cast("double*", lib.realloc(resultarrays[i].array, retlen*ffi.sizeof("double")))
                 outputd[i] = ffi.cast("double", resultarrays[i].array )
        '''+codeloop+'''
      #print('lala')
      for i in range(resultnum):
             if resultarrays[i].type == 0:
                 try:
                    import __pypy__
                    resultarrays[i].array = ffi.gc(ffi.cast("char**", resultarrays[i].array), lib.custom_free, size=resultarrays[0].size)
                 except:
                    pass
                 outputd[i] = ffi.cast("char **", resultarrays[i].array)
                 outputd[i][resultarrays[0].size] = ffi.NULL
             elif resultarrays[i].type == 1:
                 try:
                    import __pypy__
                    resultarrays[i].array = ffi.gc(ffi.cast("int*", resultarrays[i].array), lib.custom_free3, size=resultarrays[0].size+1)
                 except:
                    pass
    except Exception as e:
           errormessage[0] = lib.strdup(ffi.from_buffer(memoryview(b'UDFError '+ funcname[0].encode()+b'.'+funcname[1].encode()+b'.'+funcname[2].upper().encode() + b': ' + traceback.format_exc().encode())))
           errorm = 1
    '''
         print(code_string)
         exec(code_string)
         if errorm == 0:
             return 1
         else:
             return 0
    def free_strings(ptr):
       try:
         i = 0
         print('automatic gc collection of c objects')
         while True:
             if ptr[i] != ffi.NULL:
                #print(ffi.string(ptr[i]))
                #pass
                lib.free(ptr[i])
             else:
                break
             i+=1
       except Exception as e:
         print(str(e))
       lib.free(ptr)

    def calculate_checksum(file_path):
      hasher = hashlib.md5()
      with open(file_path, 'rb') as file:
        buf = file.read()
        hasher.update(buf)
      return hasher.hexdigest()

    def has_file_changed(file_path, initial_checksum):
      current_checksum = calculate_checksum(file_path)
      return current_checksum != initial_checksum

    def get_module_path(module_name):
      try:
        module = importlib.import_module(module_name)
        module_file = inspect.getfile(module)
        return os.path.abspath(module_file)
      except (ImportError, TypeError) as e:
        print(f"Error importing module {module_name}: {e}")
        return None


    def mysum(x, n):
        s = 0.0
        for i in range(n):
            s+=x[i]
        return s

    def mean(lst, n):
        return mysum(lst,n) / n

    class pearson:
      def __init__(self,N):
        self.sX=0
        self.sX2=0
        self.sY=0
        self.sY2=0
        self.sXY=0
        self.n=N

      def step(self,x,y):
        self.sX+=x
        self.sY+=y
        self.sX2+=x*x
        self.sY2+=y*y
        self.sXY+=x*y

      def final(self):
        numerator = self.n * self.sXY - self.sX * self.sY
        denominator_X = self.n * self.sX2 - self.sX**2
        denominator_Y = self.n * self.sY2 - self.sY**2

        r = numerator / (denominator_X**0.5 * denominator_Y**0.5)
        return r
        #d = (math.sqrt(self.n*self.sX2-self.sX*self.sX)*math.sqrt(self.n*self.sY2-self.sY*self.sY))
        #return (self.n*self.sXY-self.sX*self.sY)/d

    def multi(x):
        return x*2

    def multiv(X,n):
        for i in range(n):
            X[i] = 2*X[i]
        return X
    def pearson_correlation(X, Y, n):
      #person = pearson(n)
      #for i in range(n):
      #    person.step(multi(X[i]),multi(Y[i]))
      #return person.final()
      X = multiv(X,n)
      Y = multiv(Y,n)
      sum_X = 0
      sum_Y = 0
      sum_XY = 0
      sum_X_squared = 0
      sum_Y_squared = 0

      for i in range(n):
        x = X[i]
        y = Y[i]

        sum_X += x
        sum_Y += y
        sum_XY += x * y
        sum_X_squared += x**2
        sum_Y_squared += y**2

      # Calculate Pearson correlation coefficient
      numerator = n * sum_XY - sum_X * sum_Y
      denominator_X = n * sum_X_squared - sum_X**2
      denominator_Y = n * sum_Y_squared - sum_Y**2

      r = numerator / (denominator_X**0.5 * denominator_Y**0.5)
      print(r)
      return r

    def parse_args(input_list):
      kv_dict = {}
      rest = []

      def parse_value(val):
        val_lower = val.lower()
        if val_lower == 'true':
            return True
        elif val_lower == 'false':
            return False
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val  # return as string if all conversions fail

      for item in input_list:
        if ':' in item:
            key, value = item.split(':', 1)
            kv_dict[key] = parse_value(value)
        else:
            rest.append(item)

      return kv_dict, rest

"""
)

# for n in range(insize):
#             resultstring[n] = lib.strdup(ffi.from_buffer(memoryview('''+funcname[1]+'''.'''+funcname[2]+'''('''+code_params+''').encode())))

# for n in range(insize):
#         resultstring[n] = lib.strdup(ffi.from_buffer(memoryview('''+ffi.string(funcname).decode()+'''('''+code_params+''').encode())))
#
#    tmpstrs = ['''+ffi.string(funcname).decode()+'''('''+code_params+''') for n in range(insize)]
#    for n in range(insize):
#       resultstring[n] = ffi.from_buffer(memoryview(tmpstrs[n].encode()))
# ffibuilder.compile(target="libwrappedudfs.*")

import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <target_path>")
        sys.exit(1)

    target_path = sys.argv[1]
    ffibuilder.compile(target=target_path)
