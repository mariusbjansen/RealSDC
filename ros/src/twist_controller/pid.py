
class PIDController(object):
    def __init__(
        self,
        kp, # Proportional factor
        ki, # Integral factor
        kd, # Differential factor
        mn, # Minimal value of the regulator
        mx, # Maximal value of the regulator
        int_buffer_size # Size of the integral buffer
        ):

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.mn = mn
        self.mx = mx
        self.int_buffer_size = int_buffer_size

        self.last_error = 0.0
        self.int_val = 0.0
        self.int_buff = []

    def reset(self):
        self.int_val = 0.0
        self.int_buff = []

    def step(self, error, dt):
        self.int_val += error * dt
        self.int_buff.append((error, dt))
        while len(self.int_buff) > self.int_buffer_size:
            tmp_error, tmp_dt = self.int_buff[0]
            self.int_val -= tmp_error * tmp_dt

            del self.int_buff[0]

        derror = (error - self.last_error) / dt

        val = self.kp * error + self.ki * self.int_val + self.kd * derror
        val = max(min(val, self.mx), self.mn)

        self.last_error = error

        return val
