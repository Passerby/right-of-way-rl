import time


class Moni:

    def __init__(self, msg_type, interval_time):
        self.msg_type = msg_type
        self.interval_time = interval_time
        self.last_send_time = time.time() + 60
        self.data = {"msg_type": self.msg_type}

    def record(self, d):
        for key, value in d.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def result(self):
        result = self.data.copy()
        self.data = {"msg_type": self.msg_type}
        self.last_send_time += self.interval_time
        return result
