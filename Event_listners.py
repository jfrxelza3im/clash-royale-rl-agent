from numpy import integer  # Importing the integer type from the numpy library
from pynput import keyboard, mouse  # Importing keyboard and mouse listeners from the pynput library
from threading import Thread  # Importing Thread for multithreading
import StatePredictor  # Importing a custom module for state prediction
import ClashRoyalData  # Importing a custom module for Clash Royale data handling
from CardPredictor import ExtractSlots  # Importing the ExtractSlots function from the CardPredictor module
import win32gui  # Importing win32gui for interacting with Windows GUI elements

def CurrentCard(keypressed):
        """
        Updates the current card based on the key pressed.

        :param keypressed: The key pressed by the user, used to identify the card slot.
        """
        current = None
        Slots = ExtractSlots()  # Extract available card slots
        current = Slots.get(f"slot_{keypressed}")  # Get the card in the corresponding slot
        if current:
            print(f"Current card selected: {current}")
            ClashRoyalData.CurrentCard = current  # Update the current card in ClashRoyalData
        else:
            print(f"No card found in slot {keypressed}")

def convert_to_bluestacks_coords(global_x, global_y, bluestacks_resolution=(960, 540)):
        """
        Converts global mouse coordinates to BlueStacks window coordinates.

        :param global_x: Mouse X coordinate on the Windows desktop.
        :param global_y: Mouse Y coordinate on the Windows desktop.
        :param bluestacks_resolution: Tuple representing the internal resolution of BlueStacks (width, height).
        :return: Tuple (x, y) representing the coordinates inside the BlueStacks screen.
        """
        # Find BlueStacks window handle by title
        hwnd = win32gui.FindWindow(None, "BlueStacks App Player")
        if not hwnd:
            raise RuntimeError("BlueStacks window not found")

        # Get BlueStacks window position and size
    left2 , top2, right2, bottom2 = win32gui.GetClientRect(hwnd)

    window_largeur = right2 - left2
    window_hauteur = bottom2 - top2



    # Convertir l'origine client (0,0) en coordonnées écran
    origin_x ,  origin_y = win32gui.ClientToScreen(hwnd, (0, 0))

    # Calculate relative position inside the window
    rel_x = global_x - origin_x
    rel_y = global_y - origin_y

    # what do rel_x and rel_y represent here
    # Clamp relative position to window bounds
    # rel_x and rel_y should be between 0 and window_largeur/window_hauteur
    rel_x = max(0, min(rel_x, window_largeur))
    rel_y = max(0, min(rel_y, window_hauteur))


    virtual_w, virtual_h = bluestacks_resolution

    px_brd = 36.4 # pixels border to ignore the title bar and borders

    bs_x = round((rel_x * virtual_w / window_largeur), 2)
    bs_y = round( (rel_y - px_brd)* virtual_h / (window_hauteur - px_brd),2)
    bs_y = max(0,bs_y)  # ignore title bar area

    return bs_x , bs_y

def on_click(x, y, button, pressed):
        """
        Handles mouse click events and converts the coordinates to BlueStacks coordinates.

        :param x: Global X coordinate of the mouse click.
        :param y: Global Y coordinate of the mouse click.
        :param button: The mouse button that was clicked.
        :param pressed: Boolean indicating whether the button was pressed.
        """
        if pressed:
            new_x, new_y = convert_to_bluestacks_coords(x, y,bluestacks_resolution=(960, 540))
            print(f"bluestacks_x: {new_x}, bluestacks_y: {new_y}")

def on_key(key):
        """
        Handles keyboard key press events and updates the current card if a specific key is pressed.

        :param key: The key that was pressed.
        """
        try:
            print(f"Key {key.char} pressed")
            # Check if the key is one of the predefined keys and update the current card
            if key == keyboard.Key.f1 or keyboard.Key.f2 or keyboard.Key.f3 or keyboard.Key.f4:
                CurrentCard(key.char)
        except AttributeError:
            print(f"Special key {key} pressed")

    # Create mouse and keyboard listeners
mouse_listener = mouse.Listener(on_click=on_click)
keyboard_listener = keyboard.Listener(on_press=on_key)

print("Démarrage des écouteurs de souris et de clavier...")
mouse_listener.start()
keyboard_listener.start()

print("Écoute en cours... Appuyez sur Echap pour quitter le clavier.")

# --- 3. Maintien du programme en vie ---
try:
    # On demande au programme principal d'attendre que les listeners finissent.
    # Si on_press retourne False (touche Echap), k_listener s'arrête.
    keyboard_listener.join()
    mouse_listener.join()
except KeyboardInterrupt:
    # Pour gérer le Ctrl+C dans le terminal proprement
    print("\nStop the listner.")
    keyboard_listener.stop()
    mouse_listener.stop()









