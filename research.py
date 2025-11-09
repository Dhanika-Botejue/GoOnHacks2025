import time
import pyautogui

# Wait 3 seconds
print("Waiting 3 seconds...")
time.sleep(3)

# Get current mouse position
x, y = pyautogui.position()
print(f"Current mouse position: ({x}, {y})")

# Card 1 Pos: (622, 752)
# Card 2 Pos: (684, 752)
# Card 3 Pos: (752, 752)
# Card 4 Pos: (819, 752)

# Middle Center: (691, 481)