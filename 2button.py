import gpiod
import time
from gpiod.line import Direction, Edge

# Button GPIO pin
BUTTON_GPIO = 10

def main():
    try:
        # Open the GPIO chip
        chip = gpiod.Chip('gpiochip4')
        
        # Request the line (GPIO pin)
        config = gpiod.LineConfig()
        config.direction = Direction.INPUT
        config.edge_detection = Edge.RISING
        
        # Request line with the specified configuration
        button_line = chip.get_line(BUTTON_GPIO)
        button_line.request(config, consumer="button_test")
        
        print("Button detection running. Press Ctrl+C to quit.")
        
        # Main loop to check for button presses
        while True:
            # Wait for edge event with timeout (1 second)
            if button_line.event_wait(1):
                event = button_line.event_read()
                print("Button was pushed!")
            time.sleep(0.1)  # Small delay to reduce CPU usage
            
    except KeyboardInterrupt:
        print("\nExiting program")
    finally:
        # Clean up resources
        if 'button_line' in locals():
            button_line.release()
        if 'chip' in locals():
            chip.close()
        print("GPIO resources released")

if __name__ == "__main__":
    main()