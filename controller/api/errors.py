class BadRequest(Exception):
    def __init__(self, message):
        Exception.__init__(self)
        self.status_code = 400
        self.message = message


class UserError(Exception):
    pass
