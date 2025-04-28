import lgpio
import time

# Open GPIO chip
chip = lgpio.gpiochip_open(0)

# Button GPIO pin
BUTTON_GPIO = 10

# Configure button pin as input with pull-up resistor
lgpio.gpio_claim_input(chip, BUTTON_GPIO, lgpio.SET_PULL_UP)

# Define callback function
def button_callback(chip, gpio, level, tick):
    if level == 1:
        print("Button was pushed!")

# Setup rising edge detection - FIXED PARAMETER ORDER
callback_id = lgpio.gpio_claim_alert(
    chip,
    BUTTON_GPIO,
    lgpio.RISING_EDGE,
    button_callback
)

print("Button detection running. Press Enter to quit.")

try:
    # Keep program running
    input()
finally:
    # Clean up resources
    if 'callback_id' in locals() and callback_id >= 0:
        lgpio.gpio_free_alert(chip, BUTTON_GPIO, callback_id)
    lgpio.gpiochip_close(chip)
    print("GPIO resources released")