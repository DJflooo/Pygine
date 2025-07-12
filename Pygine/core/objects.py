from typing import Union, List, Tuple
from core.datatypes import float3, float4, AABB, OBB, matrix_from_euler_angles
from core.model import Model
import numpy as np # Import numpy for vector-matrix multiplication


class obj:
    """
    Contains all the informations about an object.
    Supports hierarchical parent-child relationships for scene graph.
    """
    _next_id = 0 # Class variable to generate unique IDs

    def __init__(self, name="Unnamed", model=None, collider: Union[AABB, OBB] = None, is_trigger: bool = False, color: float4 = None, tag: str = "Default", layer: int = 1, is_static: bool = False):
        self.id = obj._next_id # Assigns a unique ID to this instance
        obj._next_id += 1       # Increments the counter for the next instance

        self.name = name
        self.position = float3() # Local position relative to parent
        self.rotation = float3() # Local Euler angles in degrees relative to parent
        self.scale = float3(1.0, 1.0, 1.0) # Local scale relative to parent

        self.desired_rotation = self.rotation # This might be used for smooth rotation, currently not actively used for direct movement

        self.update = None
        self.model = model # Stores the Model instance here

        # Store the collider. It can be an AABB (local space) or an OBB (world space)
        # For now, we assume AABB is always local, and we convert to OBB in get_world_collider.
        self.collider = collider
        self.is_trigger = is_trigger

        self.on_collision_enter = None
        self.on_collision_stay = None
        self.on_collision_exit = None

        self.on_trigger_enter = None
        self.on_trigger_exit = None

        self._colliding_with = set()
        self._triggering_with = set()

        self.is_static = is_static

        self.velocity = float3(0, 0, 0) # World-space velocity
        self.acceleration = float3(0, 0, 0) # World-space acceleration
        self.apply_gravity = True
        self.gravity_multiplier = 1.0
        self.is_grounded = False

        self.color = color if color is not None else float4(0.8, 0.2, 0.4, 1.0) # Default to opaque
        self.texture = None

        self.tag = tag
        self.layer = layer # An integer representing the object's layer (e.g., 1, 2, 4, 8, so only powers of 2)

        # --- Scene Graph / Parent-Child Attributes ---
        self._parent: 'obj' = None
        self._children: List['obj'] = []

    @property
    def parent(self) -> 'obj':
        return self._parent

    @parent.setter
    def parent(self, new_parent: 'obj'):
        # Only change parent if it's actually different
        if self._parent is new_parent:
            return

        # If already has a parent, remove self from old parent's children list
        if self._parent:
            self._parent._children.remove(self) # Directly modify _children list

        # Set new parent and add self to new parent's children list
        self._parent = new_parent
        if new_parent:
            new_parent._children.append(self) # Directly modify _children list
            # When parenting, reset child's world-space physics state
            # as its movement will now be relative to the parent.
            self.velocity = float3(0,0,0)
            self.acceleration = float3(0,0,0)
            self.is_grounded = False # Re-evaluate grounded state after parenting

    def add_child(self, child_obj: 'obj'):
        # Use the setter to ensure proper parent/child relationship management
        child_obj.parent = self

    def remove_child(self, child_obj: 'obj'):
        # Use the setter to ensure proper parent/child relationship management
        if child_obj.parent is self:
            child_obj.parent = None # This will trigger the setter to remove from self._children

    def get_world_position(self) -> float3:
        """Calculates the object's world-space position."""
        model_matrix = self.get_model_matrix()
        # The world position is the translation component of the model matrix
        return float3(model_matrix[0, 3], model_matrix[1, 3], model_matrix[2, 3])

    # Helper method to process input args into a float3
    def _process_vector3_input(self, x_or_vector, y=None, z=None, method_name: str = "method"):
        if isinstance(x_or_vector, float3):
            return x_or_vector
        elif isinstance(x_or_vector, (tuple, list)) and len(x_or_vector) == 3:
            return float3(x_or_vector[0], x_or_vector[1], x_or_vector[2])
        elif isinstance(x_or_vector, (int, float)) and y is not None and z is not None:
            return float3(x_or_vector, y, z)
        else:
            # Enhanced error message to include types of provided arguments
            error_msg = (
                f"{method_name} expects a 'float3' object, a 3-element 'tuple' or 'list', "
                f"or three separate 'float' arguments (x, y, z)."
                f"\nReceived: x_or_vector (type: {type(x_or_vector).__name__}, value: {x_or_vector}), "
            )
            if y is not None:
                error_msg += f"y (type: {type(y).__name__}, value: {y}), "
            if z is not None:
                error_msg += f"z (type: {type(z).__name__}, value: {z})"
            else:
                error_msg = error_msg.rstrip(', ')
            
            raise ValueError(error_msg)

    # Helper method to process input args into a float4
    def _process_vector4_input(self, x_or_vector, y=None, z=None, w=None, method_name: str = "method"):
        if isinstance(x_or_vector, float4):
            return x_or_vector
        elif isinstance(x_or_vector, (tuple, list)) and len(x_or_vector) == 4:
            return float4(x_or_vector[0], x_or_vector[1], x_or_vector[2], x_or_vector[3])
        elif isinstance(x_or_vector, (int, float)) and y is not None and z is not None and w is not None:
            return float4(x_or_vector, y, z, w)
        else:
            error_msg = (
                f"{method_name} expects a 'float4' object, a 4-element 'tuple' or 'list', "
                f"or four separate 'float' arguments (x, y, z, w)."
                f"\nReceived: x_or_vector (type: {type(x_or_vector).__name__}, value: {x_or_vector}), "
            )
            if y is not None:
                error_msg += f"y (type: {type(y).__name__}, value: {y}), "
            if z is not None:
                error_msg += f"z (type: {type(z).__name__}, value: {z}), "
            if w is not None:
                error_msg += f"w (type: {type(w).__name__}, value: {w})"
            else:
                error_msg = error_msg.rstrip(', ')
            
            raise ValueError(error_msg)

    def MoveGlobal(self, x_or_vector: Union[float, float3, List[float], Tuple[float, float, float]], y: float = None, z: float = None):
        """
        Moves the object to the specified GLOBAL position (World-space).
        This method directly sets the global position by adjusting its local position
        relative to its parent.
        """
        target_world_pos = self._process_vector3_input(x_or_vector, y, z, "MoveGlobal")
        
        if self.parent:
            # Get parent's full world model matrix
            parent_world_matrix = self.parent.get_model_matrix()

            # The goal is to find 'self.position' (local) such that:
            # parent_world_matrix @ local_transform_matrix_for_self.position = target_world_pos_matrix
            # So, local_transform_matrix_for_self.position = inv(parent_world_matrix) @ target_world_pos_matrix

            # Convert target_world_pos to homogeneous coordinates (vec4)
            target_world_pos_vec4 = np.array([target_world_pos.x, target_world_pos.y, target_world_pos.z, 1.0], dtype='f4')
            
            try:
                # Get the inverse of the parent's world matrix
                inv_parent_world_matrix = np.linalg.inv(parent_world_matrix)
                
                # Transform the target world position by the inverse parent matrix
                # This gives us the new local position
                new_local_pos_vec4 = inv_parent_world_matrix @ target_world_pos_vec4
                
                # Assign the new local position
                self.position = float3(new_local_pos_vec4[0], new_local_pos_vec4[1], new_local_pos_vec4[2])
            except np.linalg.LinAlgError:
                print(f"Warning: Parent matrix for {self.name} is singular and cannot be inverted. Position not updated.")
                # If inversion fails, keep the current local position to prevent jumping to origin
                pass 
            
        else:
            # If no parent, local position IS world position
            self.position = target_world_pos

    def TranslateGlobal(self, x_or_vector: Union[float, float3, List[float], Tuple[float, float, float]], y: float = None, z: float = None):
        """
        Translates the object by a given delta in WORLD space.
        This adds to the current global position by adjusting its local position.
        """
        delta_world_vector = self._process_vector3_input(x_or_vector, y, z, "TranslateGlobal")
        
        # Get current world position
        current_world_pos = self.get_world_position()
        # Calculate target world position
        target_world_pos = current_world_pos + delta_world_vector
        
        # Now use MoveGlobal to set the new world position, which handles parent transformation
        self.MoveGlobal(target_world_pos)


    def AddVelocityLocal(self, x_or_vector: Union[float, float3, List[float], Tuple[float, float, float]], y: float = None, z: float = None):
        """
        Adds a velocity vector specified in the object's local coordinate space
        to the object's current world-space velocity.
        """
        local_velocity_delta = self._process_vector3_input(x_or_vector, y, z, "AddVelocityLocal")

        # Get the 3x3 rotation matrix for this object's current world orientation.
        # This is derived from its full model matrix, which includes parent transforms.
        # We need the 3x3 rotation part of the world model matrix.
        world_rotation_matrix_3x3 = self.get_model_matrix()[:3, :3]
        
        # Transform the local velocity vector (float3 converted to numpy array) to world space
        world_velocity_delta_np = world_rotation_matrix_3x3 @ np.array(local_velocity_delta.to_list(), dtype='f4')

        # Add the transformed velocity to the object's current world-space velocity
        self.velocity.x += world_velocity_delta_np[0]
        self.velocity.y += world_velocity_delta_np[1]
        self.velocity.z += world_velocity_delta_np[2]
    
    def AddVelocityRelativeToRotation(self, 
                                      local_x_delta: float, 
                                      local_y_delta: float, 
                                      local_z_delta: float, 
                                      reference_rotation_y_degrees: float):
        """
        Adds a velocity vector to the object's world-space velocity,
        where the local X and Z components are rotated by a given Y-axis (yaw) rotation,
        and the local Y component is applied directly to world Y.
        This is useful for character movement relative to camera's horizontal view.

        :param local_x_delta: Movement along the local X-axis (right/left).
        :param local_y_delta: Movement along the local Y-axis (up/down, directly applied to world Y).
        :param local_z_delta: Movement along the local Z-axis (forward/backward).
        :param reference_rotation_y_degrees: The yaw (Y-axis rotation) in degrees to use for
                                             transforming the local X and Z movement.
        """
        # Convert Y-rotation to radians
        rot_y_rad = np.radians(reference_rotation_y_degrees)

        # Create a 3x3 rotation matrix using ONLY the Y-axis rotation
        # This matrix will transform the local XZ plane movements into world XZ plane movements
        # aligned with the reference_rotation_y_degrees.
        # We use a standard rotation matrix for Y-axis (yaw)
        # Assuming +X is right, +Y is up, -Z is forward for your local space convention
        rotation_matrix_yaw_only = np.array([
            [np.cos(rot_y_rad), 0.0, np.sin(rot_y_rad)],  # New X-axis
            [0.0, 1.0, 0.0],                              # Y-axis remains world Y
            [-np.sin(rot_y_rad), 0.0, np.cos(rot_y_rad)]  # New Z-axis
        ], dtype='f4')

        # Create the local movement vector for horizontal components
        local_horizontal_movement = np.array([local_x_delta, 0.0, local_z_delta], dtype='f4')
        
        # Transform the horizontal movement by the yaw-only rotation matrix
        transformed_horizontal_movement = rotation_matrix_yaw_only @ local_horizontal_movement

        # Add the transformed horizontal components to the world velocity
        self.velocity.x += transformed_horizontal_movement[0]
        self.velocity.z += transformed_horizontal_movement[2]

        # Add the vertical (Y) component directly to world velocity
        self.velocity.y += local_y_delta


    def Turn(self, x_or_vector: Union[float, float3, List[float], Tuple[float, float, float]], y: float = None, z: float = None):
        """
        Turns the object by the given delta degrees on its LOCAL x, y, and z axes.
        """
        delta_rotation = self._process_vector3_input(x_or_vector, y, z, "Turn")
        self.rotation += delta_rotation # float3's __add__ handles this correctly

    def Scale(self, x_or_float3: Union[float, float3, List[float], Tuple[float, float, float]], y: float = None, z: float = None):
        """
        Scales the object on its LOCAL x, y and z axes.
        """
        new_scale = self._process_vector3_input(x_or_float3, y, z, "Scale")
        self.scale = new_scale

    def SetVelocity(self, x_or_vector: Union[float, float3, List[float], Tuple[float, float, float]], y: float = None, z: float = None):
        """
        Sets the velocity of the object to a specific world-space vector.
        """
        new_velocity = self._process_vector3_input(x_or_vector, y, z, "SetVelocity")
        self.velocity = new_velocity

    def SetColor(self, x_or_vector: Union[float, float4, List[float], Tuple[float, float, float, float]], y: float = None, z: float = None, w: float = None):
        """
        Sets the color of the object to a specific RGBA float4.
        """
        new_color = self._process_vector4_input(x_or_vector, y, z, w, "SetColor")
        self.color = new_color

    def GetTag(self):
        """
        Returns the tag of the object.
        """
        return self.tag

    def GetLayer(self):
        """
        Returns the layer of the object.
        """
        return self.layer

    def get_world_collider(self) -> Union[AABB, OBB]:
        """
        Calculates and returns the world-space collider (AABB or OBB) for the object.
        If the object has a local AABB collider, it will be transformed into a world-space OBB
        considering the object's world position, rotation, and scale (derived from its model matrix).
        """
        if self.collider is None:
            return None 

        if isinstance(self.collider, OBB):
            # If it's already an OBB, assume it's world-space and return it directly.
            # (Though for a full engine, you might want to re-calculate it if the object moved)
            # A more robust approach might be to store OBBs in local space too,
            # and transform them here, similar to AABB. But for now, let's assume
            # OBBs are created directly in world space if they are not AABBs.
            # If you *always* want to derive from model matrix, make all colliders local AABB/OBB.
            # For now, we'll keep the current behavior for pre-defined OBBs.
            return self.collider
        
        elif isinstance(self.collider, AABB):
            # Get the object's full world model matrix
            world_model_matrix = self.get_model_matrix()

            # Extract world position (translation component)
            world_pos = float3(world_model_matrix[0, 3], world_model_matrix[1, 3], world_model_matrix[2, 3])

            # Extract world scale (assuming non-skewed transformations)
            # This is correct: norm of the basis vectors gives scale.
            world_scale_x = np.linalg.norm(world_model_matrix[:3, 0])
            world_scale_y = np.linalg.norm(world_model_matrix[:3, 1])
            world_scale_z = np.linalg.norm(world_model_matrix[:3, 2])
            world_scale = float3(world_scale_x, world_scale_y, world_scale_z)

            # Extract world rotation matrix (remove scaling from the 3x3 part)
            # This is the crucial part: directly get the world rotation matrix
            # from the model matrix, *after* normalizing by scale.
            world_rotation_matrix_3x3 = world_model_matrix[:3, :3].copy()
            if world_scale_x != 0: world_rotation_matrix_3x3[:, 0] /= world_scale_x
            if world_scale_y != 0: world_rotation_matrix_3x3[:, 1] /= world_scale_y
            if world_scale_z != 0: world_rotation_matrix_3x3[:, 2] /= world_scale_z

            # Create the world-space OBB using the extracted world_pos, world_scale,
            # and the *direct* 3x3 world rotation matrix.
            return OBB.from_aabb_and_transform(self.collider, world_pos, world_rotation_matrix_3x3, world_scale)
        return None
    
    def get_model_matrix(self):
        """
        Generates the 4x4 model matrix for the object based on its LOCAL position, rotation, and scale.
        If the object has a parent, its local matrix is multiplied by the parent's world matrix.
        Order of operations for local matrix: Scale -> Rotate -> Translate.
        """
        # Calculate local scale matrix
        local_scale_matrix = np.array([
            [self.scale.x, 0.0, 0.0, 0.0],
            [0.0, self.scale.y, 0.0, 0.0],
            [0.0, 0.0, self.scale.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype='f4')

        # Calculate local rotation matrix from Euler angles (YXZ order assumed)
        local_rot_3x3 = matrix_from_euler_angles(self.rotation.x, self.rotation.y, self.rotation.z)
        local_rotation_matrix = np.identity(4, dtype='f4')
        local_rotation_matrix[:3, :3] = local_rot_3x3

        # Calculate local translation matrix
        local_translation_matrix = np.array([
            [1.0, 0.0, 0.0, self.position.x],
            [0.0, 1.0, 0.0, self.position.y],
            [0.0, 0.0, 1.0, self.position.z],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype='f4')

        # Combine local transformations: Translate * Rotate * Scale
        local_model_matrix = local_translation_matrix @ local_rotation_matrix @ local_scale_matrix

        # DEBUGGING PRINT: See the local matrix
        # print(f"DEBUG: {self.name} local_model_matrix:\n{local_model_matrix}")

        # If there's a parent, multiply by the parent's world matrix
        if self.parent:
            parent_world_matrix = self.parent.get_model_matrix()
            world_model_matrix = parent_world_matrix @ local_model_matrix
            # DEBUGGING PRINT: See the final world matrix for a child
            # print(f"DEBUG: {self.name} (child of {self.parent.name}) world_model_matrix:\n{world_model_matrix}")
            return world_model_matrix
        else:
            # DEBUGGING PRINT: See the final world matrix for a root object
            # print(f"DEBUG: {self.name} (ROOT) world_model_matrix:\n{local_model_matrix}")
            return local_model_matrix

