from gpiozero import Motor, PWMOutputDevice
from time import sleep

motor = Motor(forward=16, backward=18)
pwm = PWMOutputDevice(22)
pwm.value = 0.5  # 50% duty cycle

motor.forward()
sleep(1000)
