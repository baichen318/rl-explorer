
class NotFoundException(Exception):
    def __init__(self, target):
        self.msg = '[ERROR]: ' + target + ' not found.'

    def __str__(self):
        return self.msg

class UnDefinedException(Exception):
    def __init__(self, target):
        self.msg = '[ERROR]: ' + target + ' is undefined.'

    def __str__(self):
        return self.msg

