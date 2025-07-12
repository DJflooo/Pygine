# Pygine: A Simple Python 3D Game Engine (v0.1)

![Pygine Logo Placeholder](https://via.placeholder.com/600x300?text=Pygine+Engine+Screenshot+or+Logo)
*(Placeholder to be replaced)*

Pygine is a lightweight and accessible 3D game engine built entirely with Python, leveraging `moderngl` and `glfw` for GPU acceleration and window management. Designed for simplicity and ease of use, Pygine allows developers and aspiring game creators to quickly prototype and build 3D applications and games using familiar Python syntax.

## ✨ Features

* **Python-Native Development:** Create entire 3D scenes and game logic using only Python and Pygine-custom classes.
* **Automatic Script Loading:** Simply place your `.pyg` script files in the `scripts/` directory, and Pygine will automatically load and execute them.
* **Core 3D Primitives:** Built-in support for rendering cubes, spheres, and pyramids.
* **Custom Model Loading:** Load your own 3D models from `.obj` files.
* **Texture Support:** Apply textures to your models for enhanced visual detail.
* **Integrated Input System:** Easy-to-use API for keyboard and mouse input.
* **Flexible Scene Management:** Add, remove, and organize game objects within a centralized `Scene`.
* **Physics & Collision Detection:**
    * Basic gravity simulation.
    * Axis-Aligned Bounding Box (AABB) colliders, automatically converted to Oriented Bounding Boxes (OBB) for accurate world-space collision.
    * Configurable collision layers for precise interaction rules (e.g., player collides with ground, but not with triggers).
    * Collision callbacks (`on_collision_enter`, `on_collision_stay`, `on_collision_exit`) and trigger callbacks (`on_trigger_enter`) for event-driven game logic.
* **Hierarchical Object System:** Parent-child relationships for more complex object hierarchies.
* **Camera System:**
    * Switchable main and scene cameras.
    * Configurable third-person follow camera (demonstrated in tutorial).
    * First-person camera controls for scene exploration.
* **Delta Time Integration:** Physics and movement are frame-rate independent.

## ⚠️ Security Warning Regarding `exec()` ⚠️

**Pygine automatically executes all `.pyg` files found in the `scripts/` directory using Python's `exec()` function.**

While this provides great flexibility, it also means that **any code in these `.pyg` files will be run with full permissions** on your system.

* **Always check the contents of `.pyg` files** from untrusted sources before placing them in your `scripts/` folder and running Pygine.
* Malicious `.pyg` scripts could potentially perform harmful operations (e.g., delete files, access personal data, install unwanted software).
* The `exec()` function is also used within Pygine's `engine.py` (or similar core file) to load these scripts.

**Exercise extreme caution when downloading or running Pygine scripts from third parties.**


## 🚀 Getting Started

To get Pygine up and running, choose one of the following methods:

### Option 1: Download the Project (Recommended for Beginners)

This is the simplest way to get Pygine if you just want to run the engine and experiment with it, without needing Git installed.

1.  **Go to the Pygine GitHub repository:** Navigate to `https://github.com/DJflooo/Pygine`.
2.  **Click the green "Code" button** on the repository page.
3.  From the dropdown menu, select **"Download ZIP"**.
4.  Once downloaded, extract the `.zip` file to a location on your computer.

### Option 2: Clone the Repository (For Developers & Contributors)

If you plan to contribute to Pygine, keep your local copy updated easily, or explore the project's version history, cloning with Git is the recommended approach. You will need [Git installed](https://git-scm.com/downloads) on your system.

1.  **Open your terminal or command prompt.**

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/DJflooo/Pygine.git
    cd Pygine
    ```

3.  **Install Dependencies:**
    Pygine relies on `moderngl`, `glfw`, and `numpy`. You can install them using pip:
    ```bash
    pip install moderngl glfw numpy
    ```

4.  **Run Pygine:** (when you made your first script)
    Execute the `main.py` file from the project root:
    ```bash
    python main.py
    ```
    *(or you can use VS Code to quickly execute and make changes to your scripts (strongly recommend it). Just know that if nothing happens, it means you didn't have anything in scripts/. In that case, you should go read TUTORIAL.txt)*
    Pygine will automatically look for and load all `.pyg` files in the `scripts/` directory.

## ✍️ Creating Your First Pygine Game

Pygine uses `.pyg` files (standard Python scripts with a custom extension) to define your game logic, objects, and scene setup.

**Simple Tutorials.**

For a detailed, step-by-step tutorial on how to build a basic game with Pygine, including object creation, input handling, physics, and camera control, please refer to the comprehensive `TUTORIAL.txt` file located in the project root.

### Core Concepts You'll Encounter:

* **`obj`**: The fundamental building block for all interactive elements in your game world.
* **`Scene`**: Your game's central manager where you add and manage all game objects.
* **`Input`**: A utility for checking keyboard and mouse states.
* **`float3`, `float4`**: Custom vector types for positions, rotations, scales, and colors.
* **Collision Layers**: Define which objects can interact physically.
* **`update` functions**: Logic that runs every frame for your objects.
* **Collision Callbacks**: Functions that fire when objects collide or trigger.

## 📚 Documentation & Examples

* **`TUTORIAL.txt`**: Your go-to guide for a comprehensive (and normally simple) introduction to Pygine.

## 💡 Contributing

Pygine is an open-source project, and contributions are welcome! If you have ideas for new features, find a bug, or want to improve the codebase, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeatureName`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeatureName`).
6.  Open a Pull Request.

## 📄 License

Pygine is released under the [MIT License](LICENSE.md).

---

Made with ❤️ by DJflooo.
Hope you enjoy :D
