import gpiod

# Replace 'gpiochip4' with the appropriate chip name from gpiodetect
chip = gpiod.Chip('gpiochip4')

# Replace 23 with the GPIO line number you wish to control
line = chip.get_line(23)

# Request the line as an output
line.request(consumer="my_gpio_app", type=gpiod.LINE_REQ_DIR_OUT)

# Set the line to high
line.set_value(1)

# Release the line when done
line.release()
