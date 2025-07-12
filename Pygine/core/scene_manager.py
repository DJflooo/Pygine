
class SceneManager:
    _instance = None # To hold the singleton instance

    def __new__(cls):
        """
        Implements the Singleton pattern: ensures only one instance of SceneManager exists.
        """
        if cls._instance is None:
            cls._instance = super(SceneManager, cls).__new__(cls)
            cls._instance._objects = [] # Internal list to store game objects
        return cls._instance

    def add_object(self, game_object):
        """
        Adds a game object (an instance of obj) to the current scene.
        """
        if game_object not in self._objects:
            self._objects.append(game_object)
            # print(f"Added object: {game_object.name} to scene.") # Optional: for debugging
        else:
            print(f"Warning: Object {game_object.name} already in scene.")

    def remove_object(self, game_object):
        """
        Removes a game object from the current scene.
        """
        if game_object in self._objects:
            self._objects.remove(game_object)
            # print(f"Removed object: {game_object.name} from scene.") # Optional: for debugging
        else:
            print(f"Warning: Object {game_object.name} not found in scene.")

    def get_all_objects(self):
        """
        Returns a list of all active game objects in the scene.
        """
        return list(self._objects) # Return a copy to prevent external modification issues

    def clear_scene(self):
        """
        Clears all objects from the current scene.
        Useful when switching levels or resetting.
        """
        self._objects.clear()
        print("Scene cleared.")

Scene = SceneManager()