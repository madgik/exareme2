class StdOutputLogger:
    def info(msg: str, *args, **kwargs):
        print(msg)

    def debug(msg: str, *args, **kwargs):
        print(msg)

    def error(msg: str, *args, **kwargs):
        print(msg)
