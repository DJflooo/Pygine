import glfw

# --- Keyboard States ---
_current_key_states = {}
_just_pressed_keys = set()
_just_released_keys = set()

# A mapping from Pygine's friendly key names to GLFW's key codes
KEY_MAP = {
    "A": glfw.KEY_A, "B": glfw.KEY_B, "C": glfw.KEY_C, "D": glfw.KEY_D, "E": glfw.KEY_E,
    "F": glfw.KEY_F, "G": glfw.KEY_G, "H": glfw.KEY_H, "I": glfw.KEY_I, "J": glfw.KEY_J,
    "K": glfw.KEY_K, "L": glfw.KEY_L, "M": glfw.KEY_M, "N": glfw.KEY_N, "O": glfw.KEY_O,
    "P": glfw.KEY_P, "Q": glfw.KEY_Q, "R": glfw.KEY_R, "S": glfw.KEY_S, "T": glfw.KEY_T,
    "U": glfw.KEY_U, "V": glfw.KEY_V, "W": glfw.KEY_W, "X": glfw.KEY_X, "Y": glfw.KEY_Y,
    "Z": glfw.KEY_Z,
    "SPACE": glfw.KEY_SPACE,
    "LEFT_SHIFT": glfw.KEY_LEFT_SHIFT,
    "RIGHT_SHIFT": glfw.KEY_RIGHT_SHIFT,
    "LEFT_CONTROL": glfw.KEY_LEFT_CONTROL,
    "RIGHT_CONTROL": glfw.KEY_RIGHT_CONTROL,
    "ENTER": glfw.KEY_ENTER,
    "ESCAPE": glfw.KEY_ESCAPE,
    "UP": glfw.KEY_UP,
    "DOWN": glfw.KEY_DOWN,
    "LEFT": glfw.KEY_LEFT,
    "RIGHT": glfw.KEY_RIGHT,
    "NUM0": glfw.KEY_0, "NUM1": glfw.KEY_1, "NUM2": glfw.KEY_2, "NUM3": glfw.KEY_3,
    "NUM4": glfw.KEY_4, "NUM5": glfw.KEY_5, "NUM6": glfw.KEY_6, "NUM7": glfw.KEY_7,
    "NUM8": glfw.KEY_8, "NUM9": glfw.KEY_9,
    "F1": glfw.KEY_F1, "F2": glfw.KEY_F2, "F3": glfw.KEY_F3, "F4": glfw.KEY_F4,
    "F5": glfw.KEY_F5, "F6": glfw.KEY_F6, "F7": glfw.KEY_F7, "F8": glfw.KEY_F8,
    "F9": glfw.KEY_F9, "F10": glfw.KEY_F10, "F11": glfw.KEY_F11, "F12": glfw.KEY_F12,
}

# --- Mouse States ---
_current_mouse_button_states = {}
_just_pressed_mouse_buttons = set()
_just_released_mouse_buttons = set()

_mouse_x = 0.0 # Current X position of the mouse cursor
_mouse_y = 0.0 # Current Y position of the mouse cursor
_last_mouse_x = 0.0 # Mouse X position from the previous frame
_last_mouse_y = 0.0 # Mouse Y position from the previous frame
_mouse_dx = 0.0 # Delta X movement since last frame
_mouse_dy = 0.0 # Delta Y movement since last frame

_scroll_y_offset = 0.0 # Scroll wheel Y offset in this frame

# A mapping from Pygine's friendly mouse button names to GLFW's button codes
MOUSE_BUTTON_MAP = {
    "LEFT": glfw.MOUSE_BUTTON_LEFT,
    "RIGHT": glfw.MOUSE_BUTTON_RIGHT,
    "MIDDLE": glfw.MOUSE_BUTTON_MIDDLE,
    # Add more mouse buttons if needed
}

# --- GLFW Callbacks ---

def _key_callback(window, key, scancode, action, mods):
    """
    Internal GLFW key callback function. Updates raw key states.
    """
    global _current_key_states, _just_pressed_keys, _just_released_keys

    friendly_key_name = None
    for name, code in KEY_MAP.items():
        if code == key:
            friendly_key_name = name
            break

    if friendly_key_name:
        if action == glfw.PRESS:
            if not _current_key_states.get(friendly_key_name, False):
                _just_pressed_keys.add(friendly_key_name)
            _current_key_states[friendly_key_name] = True
        elif action == glfw.RELEASE:
            if _current_key_states.get(friendly_key_name, False):
                _just_released_keys.add(friendly_key_name)
            _current_key_states[friendly_key_name] = False

def _mouse_button_callback(window, button, action, mods):
    """
    Internal GLFW mouse button callback function. Updates raw mouse button states.
    """
    global _current_mouse_button_states, _just_pressed_mouse_buttons, _just_released_mouse_buttons

    friendly_button_name = None
    for name, code in MOUSE_BUTTON_MAP.items():
        if code == button:
            friendly_button_name = name
            break

    if friendly_button_name:
        if action == glfw.PRESS:
            if not _current_mouse_button_states.get(friendly_button_name, False):
                _just_pressed_mouse_buttons.add(friendly_button_name)
            _current_mouse_button_states[friendly_button_name] = True
        elif action == glfw.RELEASE:
            if _current_mouse_button_states.get(friendly_button_name, False):
                _just_released_mouse_buttons.add(friendly_button_name)
            _current_mouse_button_states[friendly_button_name] = False

def _cursor_pos_callback(window, xpos, ypos):
    """
    Internal GLFW cursor position callback function. Updates mouse position and calculates delta.
    """
    global _mouse_x, _mouse_y, _last_mouse_x, _last_mouse_y, _mouse_dx, _mouse_dy
    
    # Calculate delta movement
    _mouse_dx = xpos - _last_mouse_x
    _mouse_dy = ypos - _last_mouse_y
    
    # Update current and last positions
    _mouse_x = xpos
    _mouse_y = ypos
    _last_mouse_x = xpos
    _last_mouse_y = ypos

def _scroll_callback(window, xoffset, yoffset):
    """
    Internal GLFW scroll callback function. Updates scroll offset.
    """
    global _scroll_y_offset
    _scroll_y_offset = yoffset # We only care about vertical scroll for now

def _reset_frame_states():
    """
    Resets the 'just pressed/released' states and mouse deltas at the start of a new frame.
    """
    global _just_pressed_keys, _just_released_keys
    global _just_pressed_mouse_buttons, _just_released_mouse_buttons
    global _mouse_dx, _mouse_dy, _scroll_y_offset

    _just_pressed_keys.clear()
    _just_released_keys.clear()
    _just_pressed_mouse_buttons.clear()
    _just_released_mouse_buttons.clear()
    
    # Reset deltas and scroll for the new frame
    _mouse_dx = 0.0
    _mouse_dy = 0.0
    _scroll_y_offset = 0.0

# --- Public Keyboard Input Functions ---

def is_key_down(key_name):
    """
    Checks if a key is currently being held down.
    Example: Input.is_key_down("W")
    """
    return _current_key_states.get(key_name.upper(), False)

def is_key_pressed(key_name):
    """
    Checks if a key was just pressed in this frame.
    Example: Input.is_key_pressed("SPACE")
    """
    return key_name.upper() in _just_pressed_keys

def is_key_released(key_name):
    """
    Checks if a key was just released in this frame.
    Example: Input.is_key_released("LEFT_CONTROL")
    """
    return key_name.upper() in _just_released_keys

# --- Public Mouse Input Functions ---

def is_mouse_button_down(button_name):
    """
    Checks if a mouse button is currently being held down.
    Example: Input.is_mouse_button_down("LEFT")
    """
    return _current_mouse_button_states.get(button_name.upper(), False)

def is_mouse_button_pressed(button_name):
    """
    Checks if a mouse button was just pressed in this frame.
    Example: Input.is_mouse_button_pressed("RIGHT")
    """
    return button_name.upper() in _just_pressed_mouse_buttons

def is_mouse_button_released(button_name):
    """
    Checks if a mouse button was just released in this frame.
    Example: Input.is_mouse_button_released("MIDDLE")
    """
    return button_name.upper() in _just_released_mouse_buttons

def get_mouse_position():
    """
    Returns the current mouse cursor position as (x, y).
    """
    return (_mouse_x, _mouse_y)

def get_mouse_delta():
    """
    Returns the change in mouse cursor position from the previous frame as (dx, dy).
    """
    return (_mouse_dx, _mouse_dy)

def get_scroll_offset():
    """
    Returns the vertical scroll wheel offset for the current frame.
    Positive for scrolling up, negative for scrolling down.
    """
    return _scroll_y_offset