
class PIDController(object):
    def __init__(
        self,
        kp, # Proportional factor
        ki, # Integral factor
        kd, # Differential factor
        mn, # Minimal value of the regulator
        mx, # Maximal value of the regulator
        integral_period_sec # Size of the integral buffer
        ):

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.mn = mn
        self.mx = mx
        self.integral_period_sec = integral_period_sec

        self.last_error = 0.0
        self.int_val = 0.0
        self.int_time = 0.0
        self.int_buff = []

    def reset(self):
        self.int_val = 0.0
        self.int_time = 0.0
        self.int_buff = []

    def step(self, error, dt):
        self.int_val += error * dt
        self.int_time += dt

        self.int_buff.append((error, dt))
        while (self.int_time > self.integral_period_sec) and (len(self.int_buff) > 0):
            tmp_error, tmp_dt = self.int_buff[0]
            self.int_val -= tmp_error * tmp_dt
            self.int_time -= tmp_dt

            del self.int_buff[0]

        derror = (error - self.last_error) / dt

        val = self.kp * error + self.ki * self.int_val + self.kd * derror
        val = max(min(val, self.mx), self.mn)

        self.last_error = error

        return val
