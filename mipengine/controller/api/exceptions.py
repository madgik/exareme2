class BadRequest(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.status_code = 400
        self.message = message


class BadUserInput(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.status_code = 460
        self.message = message


class UnexpectedException(Exception):
    def __init__(self):
        super().__init__()
        self.status_code = 500
