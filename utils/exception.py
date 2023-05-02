# Author: baichen318@gmail.com


class NotFoundException(Exception):
    def __init__(self, target):
        self.msg = target + " is not found."

    def __str__(self):
        return self.msg


class UnSupportedException(Exception):
    def __init__(self, target):
        self.msg = target

    def __str__(self):
        return self.msg


class EvaluateException(Exception):
    def __init__(self, target):
        self.msg = target

    def __str__(self):
        return self.msg
