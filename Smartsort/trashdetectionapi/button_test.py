import lgpio

chip = lgpio.gpiochip_open(0)

BUTTON_GPIO = 10

# Setup pull-down resistor
# lgpio.gpio_set_pull(chip, BUTTON_GPIO, lgpio.SET_PULL_DOWN)
# lgpio.gpio
# Setup rising edge alert
'''

callback_id = lgpio.gpio_claim_alert(
    chip,
    BUTTON_GPIO,
    lgpio.RISING_EDGE,
    lambda chip, gpio, level, tick: print("Button was pushed!") if level == 1 else None
)
'''
callback_id = lgpio.gpio_claim_alert(
    chip,
    BUTTON_GPIO,
    lgpio.RISING_EDGE,
    #lambda chip, gpio, level, tick: print("Button was pushed!") if level == 1 else None
)

try:
    input("Press Enter to quit\n\n")
finally:
    lgpio.gpiochip_close(chip)


# def button_callback(channel):
#     print("Button was pushed!")

# GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
# GPIO.add_event_detect(10,GPIO.RISING,callback=button_callback)

# message = input("Press enter to quit\n\n")

# GPIO.cleanup()
 