import time

class SessionAnalyzer:

    def __init__(self, duration_minutes):
        self.start_time = time.time()
        self.duration = duration_minutes * 60

    def is_active(self):
        return (time.time() - self.start_time) < self.duration

    def time_left(self):
        return max(0, self.duration - (time.time() - self.start_time))