import moderngl
import glfw
import os
import struct
import numpy as np
from math import cos, sin, radians, tan, pi
from core.objects import obj # Ensure this is imported correctly
from core.datatypes import float3, float4, AABB
from core.model import Model
from core import input
from core.scene_manager import Scene
from core.camera import Camera
from core.collision_manager import CollisionManager
from core.renderer import Renderer
from core.model_loader import load_obj_model

_TOGGLE_CAMERA_KEY = "P"
main_camera = None
scene_camera = None
active_camera = None # This will be the actual camera instance used

_last_frame_time = 0.0 # Stores the time at the end of the previous frame
_delta_time = 0.0      # Stores the time passed since the last frame

GRAVITY = float3(0, -15, 0) # Adjust for your game's scale

# --- Hardcoded Model Generation Functions (unchanged) ---
def generate_sphere_data(radius: float = 1.0, segments: int = 32) -> tuple[list[float], list[int]]:
    vertices = []
    indices = []
    for i in range(segments + 1):
        lat = pi * (-0.5 + float(i) / segments)
        sin_lat = sin(lat)
        cos_lat = cos(lat)
        for j in range(segments + 1):
            lon = 2 * pi * float(j) / segments
            sin_lon = sin(lon)
            cos_lon = cos(lon)
            x = radius * cos_lon * cos_lat
            y = radius * sin_lat
            z = radius * sin_lon * cos_lat
            norm_x, norm_y, norm_z = x / radius, y / radius, z / radius
            u = float(j) / segments
            v = float(i) / segments
            vertices.extend([x, y, z, norm_x, norm_y, norm_z, u, v])
    for i in range(segments):
        for j in range(segments):
            first_row = i * (segments + 1)
            second_row = (i + 1) * (segments + 1)
            indices.append(first_row + j)
            indices.append(second_row + j)
            indices.append(first_row + j + 1)
            indices.append(first_row + j + 1)
            indices.append(second_row + j)
            indices.append(second_row + j + 1)
    return vertices, indices

def generate_pyramid_data(base_size: float = 2.0, height: float = 2.0) -> tuple[list[float], list[int]]:
    half_base = base_size / 2.0
    tip_y = height / 2.0
    base_y = -height / 2.0
    vertices = []
    indices = []
    tip_pos = [0.0, tip_y, 0.0]
    brf = [half_base, base_y, half_base]
    blf = [-half_base, base_y, half_base]
    blb = [-half_base, base_y, -half_base]
    brb = [half_base, base_y, -half_base]
    norm_bottom = np.array([0.0, -1.0, 0.0], dtype='f4')
    def calculate_normal(v1, v2, v3):
        vec1 = np.array(v2) - np.array(v1)
        vec2 = np.array(v3) - np.array(v1)
        normal = np.cross(vec1, vec2)
        return list(normal / np.linalg.norm(normal))
    v_idx_start = len(vertices) // 8
    vertices.extend(tip_pos + calculate_normal(tip_pos, blf, brf) + [0.5, 1.0])
    vertices.extend(blf + calculate_normal(tip_pos, blf, brf) + [0.0, 0.0])
    vertices.extend(brf + calculate_normal(tip_pos, blf, brf) + [1.0, 0.0])
    indices.extend([v_idx_start, v_idx_start + 1, v_idx_start + 2])
    v_idx_start = len(vertices) // 8
    vertices.extend(tip_pos + calculate_normal(tip_pos, brb, brf) + [0.5, 1.0])
    vertices.extend(brf + calculate_normal(tip_pos, brb, brf) + [0.0, 0.0])
    vertices.extend(brb + calculate_normal(tip_pos, brb, brf) + [1.0, 0.0])
    indices.extend([v_idx_start, v_idx_start + 1, v_idx_start + 2])
    v_idx_start = len(vertices) // 8
    vertices.extend(tip_pos + calculate_normal(tip_pos, blb, brb) + [0.5, 1.0])
    vertices.extend(brb + calculate_normal(tip_pos, blb, brb) + [0.0, 0.0])
    vertices.extend(blb + calculate_normal(tip_pos, blb, brb) + [1.0, 0.0])
    indices.extend([v_idx_start, v_idx_start + 1, v_idx_start + 2])
    v_idx_start = len(vertices) // 8
    vertices.extend(tip_pos + calculate_normal(tip_pos, blf, blb) + [0.5, 1.0])
    vertices.extend(blb + calculate_normal(tip_pos, blf, blb) + [0.0, 0.0])
    vertices.extend(blf + calculate_normal(tip_pos, blf, blb) + [1.0, 0.0])
    indices.extend([v_idx_start, v_idx_start + 1, v_idx_start + 2])
    v_idx_start = len(vertices) // 8
    vertices.extend(brf + list(norm_bottom) + [1.0, 1.0])
    vertices.extend(blf + list(norm_bottom) + [0.0, 1.0])
    vertices.extend(blb + list(norm_bottom) + [0.0, 0.0])
    vertices.extend(brb + list(norm_bottom) + [1.0, 0.0])
    indices.extend([v_idx_start, v_idx_start + 1, v_idx_start + 2,
                    v_idx_start, v_idx_start + 2, v_idx_start + 3])
    return vertices, indices

def run_engine():
    global main_camera, scene_camera, active_camera
    global _last_frame_time, _delta_time
    
    if not glfw.init():
        raise Exception("glfw can't be initialized")

    window = glfw.create_window(800, 600, "Pygine", None, None)

    if not window:
        glfw.terminate()
        raise Exception("glfw window can't be created")

    glfw.make_context_current(window)
    ctx = moderngl.create_context()
    
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.CULL_FACE)
    
    glfw.set_key_callback(window, input._key_callback)
    glfw.set_mouse_button_callback(window, input._mouse_button_callback)
    glfw.set_cursor_pos_callback(window, input._cursor_pos_callback)
    glfw.set_scroll_callback(window, input._scroll_callback)
    
    # Define cube vertices (unchanged)
    cube_vertices = [
        # Front face (Z+ normal: 0,0,1) - Indices 0-3
        #   Pos             Normal        UV
         1.0,  1.0,  1.0,   0.0, 0.0, 1.0,   1.0, 1.0, # v0 (Top Right Front)
        -1.0,  1.0,  1.0,   0.0, 0.0, 1.0,   0.0, 1.0, # v1 (Top Left Front)
        -1.0, -1.0,  1.0,   0.0, 0.0, 1.0,   0.0, 0.0, # v2 (Bottom Left Front)
         1.0, -1.0,  1.0,   0.0, 0.0, 1.0,   1.0, 0.0, # v3 (Bottom Right Front)

        # Back face (Z- normal: 0,0,-1) - Indices 4-7
        #   Pos             Normal        UV
         1.0, -1.0, -1.0,   0.0, 0.0, -1.0,  0.0, 0.0, # v4 (Bottom Right Back)
         1.0,  1.0, -1.0,   0.0, 0.0, -1.0,  0.0, 1.0, # v5 (Top Right Back)
        -1.0,  1.0, -1.0,   0.0, 0.0, -1.0,  1.0, 1.0, # v6 (Top Left Back)
        -1.0, -1.0, -1.0,   0.0, 0.0, -1.0,  1.0, 0.0, # v7 (Bottom Left Back)

        # Right face (X+ normal: 1,0,0) - Indices 8-11
        #   Pos             Normal        UV
         1.0,  1.0,  1.0,   1.0, 0.0, 0.0,   0.0, 1.0, # v8 (Top Right Front)
         1.0, -1.0,  1.0,   1.0, 0.0, 0.0,   0.0, 0.0, # v9 (Bottom Right Front)
         1.0, -1.0, -1.0,   1.0, 0.0, 0.0,   1.0, 0.0, # v10 (Bottom Right Back)
         1.0,  1.0, -1.0,   1.0, 0.0, 0.0,   1.0, 1.0, # v11 (Top Right Back)

        # Left face (X- normal: -1,0,0) - Indices 12-15
        #   Pos             Normal        UV
        -1.0,  1.0,  1.0,  -1.0, 0.0, 0.0,   1.0, 1.0, # v12 (Top Left Front)
        -1.0, -1.0,  1.0,  -1.0, 0.0, 0.0,   1.0, 0.0, # v13 (Bottom Left Front)
        -1.0, -1.0, -1.0,  -1.0, 0.0, 0.0,   0.0, 0.0, # v14 (Bottom Left Back)
        -1.0,  1.0, -1.0,  -1.0, 0.0, 0.0,   0.0, 1.0, # v15 (Top Left Back)

        # Top face (Y+ normal: 0,1,0) - Indices 16-19
        #   Pos             Normal        UV
         1.0,  1.0,  1.0,   0.0, 1.0, 0.0,   1.0, 0.0, # v16 (Top Right Front)
         1.0,  1.0, -1.0,   0.0, 1.0, 0.0,   1.0, 1.0, # v17 (Top Right Back)
        -1.0,  1.0, -1.0,   0.0, 1.0, 0.0,   0.0, 1.0, # v18 (Top Left Back)
        -1.0,  1.0,  1.0,   0.0, 1.0, 0.0,   0.0, 0.0, # v19 (Top Left Front)

        # Bottom face (Y- normal: 0,-1,0) - Indices 20-23
        #   Pos             Normal        UV
         1.0, -1.0,  1.0,   0.0, -1.0, 0.0,  1.0, 1.0, # v20 (Bottom Right Front)
        -1.0, -1.0,  1.0,   0.0, -1.0, 0.0,  0.0, 1.0, # v21 (Bottom Left Front)
        -1.0, -1.0, -1.0,   0.0, -1.0, 0.0,  0.0, 0.0, # v22 (Bottom Left Back)
         1.0, -1.0, -1.0,   0.0, -1.0, 0.0,  1.0, 0.0, # v23 (Bottom Right Back)
    ]
    cube_indices = [
        0, 1, 2,   0, 2, 3,  
        4, 7, 6,   4, 6, 5,  
        8, 9, 10,  8, 10, 11,
        12, 15, 14, 12, 14, 13,
        16, 17, 18, 16, 18, 19,
        20, 21, 22, 20, 22, 23,
    ]
    
    renderer = Renderer(ctx)
    cube_model = Model(ctx, cube_vertices, cube_indices, renderer.main_program)

    sphere_vertices, sphere_indices = generate_sphere_data(radius=1.0, segments=32)
    sphere_model = Model(ctx, sphere_vertices, sphere_indices, renderer.main_program)

    pyramid_vertices, pyramid_indices = generate_pyramid_data(base_size=2.0, height=2.0)
    pyramid_model = Model(ctx, pyramid_vertices, pyramid_indices, renderer.main_program)
    
    main_camera = Camera()
    main_camera.position.z = 5.0

    scene_camera = Camera()
    scene_camera.position.z = 10.0
    scene_camera.position.y = 5.0
    scene_camera.rotation.x = -20.0

    active_camera = main_camera
    
    # Calculate the project's root directory (Pygine/)
    project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define the directory where user scripts are located
    scripts_dir = os.path.join(project_root_dir, "scripts")

    # Custom global scope for user scripts
    # These are the variables and functions that user scripts will have direct access to.
    script_globals = {
        "obj": obj,
        "Input": input,
        "cube_model": cube_model,
        "sphere_model": sphere_model,
        "pyramid_model": pyramid_model,
        "Scene": Scene,
        "Camera": Camera,
        "main_camera": main_camera,
        "active_camera": lambda: active_camera, # Provides the currently active camera instance
        "get_delta_time": lambda: _delta_time,   # Provides the time since the last frame
        "AABB": AABB,
        "float3": float3,
        "float4": float4,
        "load_texture_func": renderer.load_texture,
        "PROJECT_ROOT_DIR": project_root_dir, 
        "load_model_func": lambda path: Model(ctx, *load_obj_model(path), renderer.main_program),
        # Collision Manager specific imports needed for user scripts
        "CollisionManager": CollisionManager, 
        "LAYER_DEFAULT": 1, # Expose collision layer constants to user scripts
        "LAYER_PLAYER": 2,
        "LAYER_GROUND": 4,
        "LAYER_ENEMY": 8,
    }

    # --- New Logic: Load and Execute all .pyg files in the scripts directory ---
    loaded_scripts_count = 0
    if os.path.exists(scripts_dir) and os.path.isdir(scripts_dir):
        for filename in os.listdir(scripts_dir):
            if filename.endswith(".pyg"):
                script_path = os.path.join(scripts_dir, filename)
                print(f"Loading script: {filename}")
                try:
                    # 'exec' executes the code from the file within the 'script_globals' scope.
                    # This means variables defined in the user's .pyg script will be available
                    # in the shared Scene and other engine components.
                    exec(open(script_path).read(), script_globals)
                    loaded_scripts_count += 1
                except Exception as e:
                    print(f"Error loading script {filename}: {e}")
    else:
        print(f"Warning: Scripts directory not found at {scripts_dir}. No user scripts will be loaded.")

    if loaded_scripts_count == 0:
        print("No .pyg scripts were found or loaded. Ensure your scripts are in the 'scripts/' directory and end with '.pyg'.")
    
    if not Scene.get_all_objects():
        # This check is now even more important as multiple scripts might contribute to the scene.
        raise Exception("No game objects found in the scene after loading all scripts. Please ensure your scripts create objects and add them using Scene.add_object().")
    
    width, height = glfw.get_framebuffer_size(window)
    aspect_ratio = width / height
    
    input._last_mouse_x, input._last_mouse_y = glfw.get_cursor_pos(window)
    _last_frame_time = glfw.get_time()
    
    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        _delta_time = current_time - _last_frame_time
        _last_frame_time = current_time
        
        input._reset_frame_states()
        glfw.poll_events()
        
        if input.is_key_pressed(_TOGGLE_CAMERA_KEY):
            if active_camera is main_camera:
                active_camera = scene_camera
                print("Switched to Scene Camera")
            else:
                active_camera = main_camera
                print("Switched to Main Camera")
        
        # --- Camera Control Logic (unchanged for now) ---
        if active_camera is scene_camera:
            camera_speed = 3.0
            rotation_speed = 100.0
            mouse_sensitivity = 0.1
            scroll_speed = 750.0

            if input.is_key_down("W"):
                scene_camera.Move(0, 0, -camera_speed * _delta_time)
            if input.is_key_down("S"):
                scene_camera.Move(0, 0, camera_speed * _delta_time)
            if input.is_key_down("A"):
                scene_camera.Move(-camera_speed * _delta_time, 0, 0)
            if input.is_key_down("D"):
                scene_camera.Move(camera_speed * _delta_time, 0, 0)
            if input.is_key_down("E"):
                scene_camera.Move(0, camera_speed * _delta_time, 0)
            if input.is_key_down("Q"):
                scene_camera.Move(0, -camera_speed * _delta_time, 0)

            if input.is_key_down("UP"):
                scene_camera.Turn(rotation_speed * _delta_time, 0, 0)
            if input.is_key_down("DOWN"):
                scene_camera.Turn(-rotation_speed * _delta_time, 0, 0)
            if input.is_key_down("LEFT"):
                scene_camera.Turn(0, rotation_speed * _delta_time, 0)
            if input.is_key_down("RIGHT"):
                scene_camera.Turn(0, -rotation_speed * _delta_time, 0)

            if input.is_mouse_button_down("RIGHT"):
                dx, dy = input.get_mouse_delta()
                if dx != 0 or dy != 0:
                    scene_camera.Turn(-dy * mouse_sensitivity, -dx * mouse_sensitivity, 0)

                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
                center_x, center_y = width / 2, height / 2
                glfw.set_cursor_pos(window, center_x, center_y)
                input._last_mouse_x, input._last_mouse_y = center_x, center_y
            else:
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

            scroll_y = input.get_scroll_offset()
            if scroll_y != 0:
                scene_camera.Move(0, 0, -scroll_y * scroll_speed * _delta_time)
        else:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        
        for game_obj in Scene.get_all_objects():
            # Always call the object's update method if it exists, regardless of parentage.
            # This allows scripts to control local transformations or other logic for children.
            if game_obj.update:
                game_obj.update(active_camera, _delta_time)
            
            # Apply physics integration ONLY if the object is a root object (no parent).
            if game_obj.parent is None:
                if game_obj.apply_gravity is True and game_obj.is_grounded is False:
                    game_obj.acceleration += GRAVITY * game_obj.gravity_multiplier
                
                game_obj.velocity += game_obj.acceleration * _delta_time
                
                current_world_pos = game_obj.get_world_position()
                new_world_pos = current_world_pos + (game_obj.velocity * _delta_time)
                
                game_obj.MoveGlobal(new_world_pos)

                game_obj.acceleration = float3(0, 0, 0)
            # Children's world positions are implicitly updated when their parent's world position changes
            # and get_model_matrix() is called during rendering.
        
        CollisionManager.process_collisions(Scene.get_all_objects())

        renderer.render_scene(active_camera, Scene.get_all_objects(), aspect_ratio)
        
        glfw.swap_buffers(window)
    
    # Release resources when the game loop ends
    cube_model.release()
    sphere_model.release()
    pyramid_model.release()
    renderer.release()
    glfw.terminate()