# coding: utf-8

import re

from lib.stopwordlist import stopwords

from . import setpath


def filterstopwords(*args):
    """
    .. function:: filterstopwords(str) -> str

    Returns the input text with the stopwords removed. The case of the first letter matters.

    Examples:

    >>> table1('''
    ... 'this and wood'         'NO more No words'
    ... 'No more stop words'    'more free time'
    ... ''')
    >>> sql("select filterstopwords(a,b) from table1")
    filterstopwords(a,b)
    --------------------
    wood NO words
    stop words free time
    """
    arg = args[0].decode()
    print(arg)
    if len(args) == 1:
        return " ".join(
            [
                k
                for k in arg.split(" ")
                if k != "" and k[0].lower() + k[1:] not in stopwords
            ]
        ).encode()

    out = []
    for i in args:
        out.append(
            " ".join(
                [
                    k
                    for k in i.split(" ")
                    if k != "" and k[0].lower() + k[1:] not in stopwords
                ]
            )
        )

    return " ".join(out)


filterstopwords.registered = True


if not ("." in __name__):
    """
    This is needed to be able to test the function, put it at the end of every
    new function you create
    """
    import sys

    from functions import *

    from . import setpath

    testfunction()
    if __name__ == "__main__":
        reload(sys)
        sys.setdefaultencoding("utf-8")
        import doctest

        doctest.testmod()
