from termcolor import colored


class StdOutputLogger:
    def info(self, msg: str, *args, **kwargs):
        prefix = colored("INFO", "white", "on_green")
        print(f"{prefix} - ", end="")
        print(msg)

    def debug(self, msg: str, *args, **kwargs):
        prefix = colored("DEBUG", "white", "on_purple")
        print(f"{prefix} - ", end="")
        print(msg)

    def warning(self, msg: str, *args, **kwargs):
        prefix = colored("ERROR", "white", "on_yellow")
        print(f"{prefix} - ", end="")
        print(msg)

    def error(self, msg: str, *args, **kwargs):
        prefix = colored("ERROR", "white", "on_red")
        print(f"{prefix} - ", end="")
        print(msg)

    def critical(self, msg: str, *args, **kwargs):
        prefix = colored("ERROR", "red")
        print(f"{prefix} - ", end="")
        print(msg)
