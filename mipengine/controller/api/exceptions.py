class BadRequest(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class BadUserInput(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
