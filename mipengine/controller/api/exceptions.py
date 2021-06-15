class BadRequest(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.status_code = 400
        self.message = message


class BadUserInput(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.status_code = 200
        self.message = create_response("text/plain+user_error", message)


def create_response(mime_type: str, message: str):
    return {mime_type: message}
