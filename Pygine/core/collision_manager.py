from core.objects import obj
from core.datatypes import AABB, OBB, float3
import numpy as np

class CollisionManager:
    """
    Manages collision and trigger detection for all objects in the scene using OBBs.
    Implements the Separating Axis Theorem (SAT) for OBB-OBB intersection and MTV calculation.
    Includes collision layer filtering and passes collision normal to callbacks.
    """

    _collision_layer_rules = {} 

    @staticmethod
    def initialize_layer_collision_rules(known_layers: list[int]):
        CollisionManager._collision_layer_rules = {}
        for layer1 in known_layers:
            for layer2 in known_layers:
                if layer1 <= layer2:
                    CollisionManager._collision_layer_rules[(layer1, layer2)] = True
                else:
                    CollisionManager._collision_layer_rules[(layer2, layer1)] = True

    @staticmethod
    def set_layer_collision(layer1_mask: int, layer2_mask: int, enabled: bool):
        if layer1_mask > layer2_mask:
            layer1_mask, layer2_mask = layer2_mask, layer1_mask
        CollisionManager._collision_layer_rules[(layer1_mask, layer2_mask)] = enabled

    @staticmethod
    def _should_layers_collide(layer1: int, layer2: int) -> bool:
        if layer1 > layer2:
            layer1, layer2 = layer2, layer1
        return CollisionManager._collision_layer_rules.get((layer1, layer2), True)

    @staticmethod
    def _project_obb_on_axis(obb: OBB, axis: float3) -> tuple[float, float]:
        vertices = obb.get_vertices()
        projections = [v.dot(axis) for v in vertices]
        return min(projections), max(projections)

    @staticmethod
    def _overlap_intervals(min1: float, max1: float, min2: float, max2: float) -> tuple[bool, float]:
        if max1 < min2 or max2 < min1:
            return False, 0.0
        penetration = min(max1, max2) - max(min1, min2)
        return True, penetration
    
    @staticmethod
    def _check_obb_collision_sat(obb1: OBB, obb2: OBB) -> tuple[bool, float3, float3]:
        axes_obb1 = obb1.get_axes()
        axes_obb2 = obb2.get_axes()

        candidate_axes = []
        for ax1 in axes_obb1:
            candidate_axes.append(ax1.normalize())
        for ax2 in axes_obb2:
            candidate_axes.append(ax2.normalize())
        
        for ax1 in axes_obb1:
            for ax2 in axes_obb2:
                cross_product = ax1.cross(ax2)
                if cross_product.length_sq() > 1e-6: 
                    candidate_axes.append(cross_product.normalize())
        
        min_overlap = float('inf')
        mtv_axis = float3(0, 0, 0) 
        
        for axis in candidate_axes:
            if axis.length_sq() < 1e-6:
                continue

            min1, max1 = CollisionManager._project_obb_on_axis(obb1, axis)
            min2, max2 = CollisionManager._project_obb_on_axis(obb2, axis)
            
            overlaps, penetration = CollisionManager._overlap_intervals(min1, max1, min2, max2)
            
            if not overlaps:
                return False, float3(0, 0, 0), float3(0, 0, 0) 
            
            if penetration < min_overlap:
                min_overlap = penetration
                mtv_axis = axis 
                
        if mtv_axis.length_sq() < 1e-6:
             return False, float3(0,0,0), float3(0,0,0)

        d = obb1.center - obb2.center
        if d.dot(mtv_axis) < 0:
            mtv_axis = -mtv_axis 

        mtv = mtv_axis * min_overlap
        normal = mtv_axis.normalize() 

        return True, mtv, normal

    @staticmethod
    def process_collisions(all_objects: list):
        collidable_objects = [o for o in all_objects if o.collider is not None]
        num_objects = len(collidable_objects)

        current_frame_overlaps = {obj.id: set() for obj in collidable_objects}
        
        # NEW: detailed_contacts will store all contacts detected in Phase 1.
        # It will NOT be cleared during resolution iterations.
        # Each entry will be {obj.id: [(other_obj, initial_mtv, initial_normal), ...]}
        # We store initial MTV/normal because they are based on positions *before* resolution.
        initial_detailed_contacts = {obj.id: [] for obj in collidable_objects}

        if not hasattr(CollisionManager, '_previous_frame_overlaps'):
            CollisionManager._previous_frame_overlaps = {obj.id: set() for obj in collidable_objects}

        NUM_RESOLUTION_ITERATIONS = 9
        RESOLUTION_EPSILON_SQ = 1e-6 
        SLOPE_LIMIT_COS = 0.8 # cos(45 degrees) for ground detection
        CORRECTION_FACTOR = 0.65 # Apply 65% of the calculated MTV per iteration

        # --- PHASE 0: Reset is_grounded for all objects at the start of the frame ---
        # This is CRUCIAL for reliable grounded state.
        for game_obj in collidable_objects:
            game_obj.is_grounded = False

        # --- PHASE 1: DETECT COLLISIONS AND GATHER CONTACTS (ONCE PER FRAME) ---
        for i in range(num_objects):
            obj1 = collidable_objects[i]
            world_collider1 = obj1.get_world_collider() 
            if world_collider1 is None: 
                continue

            for j in range(i + 1, num_objects): 
                obj2 = collidable_objects[j]
                world_collider2 = obj2.get_world_collider() 
                if world_collider2 is None:
                    continue

                if obj1.is_static and obj2.is_static:
                    continue

                if not CollisionManager._should_layers_collide(obj1.layer, obj2.layer):
                    continue

                is_colliding, mtv_for_obj1, collision_normal = \
                    CollisionManager._check_obb_collision_sat(world_collider1, world_collider2)

                if is_colliding:
                    current_frame_overlaps[obj1.id].add(obj2.id)
                    current_frame_overlaps[obj2.id].add(obj1.id)

                    if not obj1.is_trigger and not obj2.is_trigger:
                        if not obj1.is_static: 
                            initial_detailed_contacts[obj1.id].append((obj2, mtv_for_obj1, collision_normal))
                        if not obj2.is_static: 
                            initial_detailed_contacts[obj2.id].append((obj1, -mtv_for_obj1, -collision_normal))


         # --- PHASE 2: ITERATIVE COLLISION RESOLUTION ---
        for iteration in range(NUM_RESOLUTION_ITERATIONS):
            has_resolved_this_iteration = False 

            for obj_to_resolve_id, contacts in initial_detailed_contacts.items():
                obj_to_move = next((o for o in collidable_objects if o.id == obj_to_resolve_id), None)
                
                if not obj_to_move or obj_to_move.is_trigger or obj_to_move.is_static: 
                    continue

                total_mtv_this_iteration = float3(0, 0, 0)
                
                for other_obj, _, _ in contacts: 
                    current_world_collider_self = obj_to_move.get_world_collider()
                    current_world_collider_other = other_obj.get_world_collider()

                    if current_world_collider_self is None or current_world_collider_other is None:
                        continue
                    
                    is_colliding_current, mtv_current, normal_current = \
                        CollisionManager._check_obb_collision_sat(current_world_collider_self, current_world_collider_other)
                    
                    if is_colliding_current:
                        total_mtv_this_iteration += mtv_current
                        
                        if other_obj.tag == "Ground" and normal_current.y > SLOPE_LIMIT_COS:
                            obj_to_move.is_grounded = True

                if total_mtv_this_iteration.length_sq() > RESOLUTION_EPSILON_SQ:
                    obj_to_move.position += total_mtv_this_iteration * CORRECTION_FACTOR
                    has_resolved_this_iteration = True
                    
                    if obj_to_move.apply_gravity:
                        if obj_to_move.is_grounded:
                            # When grounded, prevent vertical movement due to gravity
                            obj_to_move.velocity.y = 0 # <--- CHANGE THIS LINE
                            obj_to_move.acceleration.y = 0 # <--- ADD/KEEP THIS LINE
                            # Apply friction to horizontal velocity
                            friction_factor = 0.85 
                            obj_to_move.velocity.x *= friction_factor
                            obj_to_move.velocity.z *= friction_factor
                            
                            # Snap total small velocities to zero to prevent creeping.
                            # This is usually fine as it's a very low threshold.
                            if obj_to_move.velocity.length_sq() < 0.1**2: 
                                obj_to_move.velocity = float3(0,0,0)
            
            if not has_resolved_this_iteration:
                break


        # --- PHASE 3: EVENT DISPATCHING ---
        # This phase remains largely the same. `is_grounded` is already set in Phase 2.
        for obj in collidable_objects:
            obj_id = obj.id
            if obj_id not in CollisionManager._previous_frame_overlaps:
                CollisionManager._previous_frame_overlaps[obj_id] = set()

            current_ids = current_frame_overlaps.get(obj_id, set())
            previous_ids = CollisionManager._previous_frame_overlaps.get(obj_id, set())

            all_involved_ids = current_ids.union(previous_ids)

            for other_id in all_involved_ids:
                other_obj_instance = next((o for o in collidable_objects if o.id == other_id), None)
                if not other_obj_instance: 
                    continue
                
                if not CollisionManager._should_layers_collide(obj.layer, other_obj_instance.layer):
                    continue

                world_collider_self = obj.get_world_collider()
                world_collider_other = other_obj_instance.get_world_collider()

                if world_collider_self is None or world_collider_other is None:
                    continue 

                is_colliding_now, _, event_collision_normal = \
                    CollisionManager._check_obb_collision_sat(world_collider_self, world_collider_other)
                
                # Refined handling for event_collision_normal
                if not is_colliding_now or event_collision_normal.length_sq() < 1e-6:
                    # If not currently colliding, or if SAT returned a degenerate normal for some reason,
                    # the normal for events is ambiguous.
                    # For ENTER/STAY events, we ideally want a valid normal.
                    # For EXIT events, the normal is often less critical as objects are separating.
                    
                    # Provide a default normal (e.g., upward) if it's supposed to be colliding but the normal is bad.
                    # Otherwise, a zero normal implies no clear contact direction.
                    if is_colliding_now: # It *should* be colliding, but the normal is bad
                        event_collision_normal = float3(0, 1, 0) # Default to an upward normal
                    else: # It's not colliding, so no meaningful collision normal exists
                        event_collision_normal = float3(0, 0, 0) 

                # --- OnCollision/TriggerExit ---
                if other_id in previous_ids and other_id not in current_ids:
                    if obj.is_trigger or other_obj_instance.is_trigger:
                        if obj.on_trigger_exit:
                            obj.on_trigger_exit(obj, other_obj_instance, event_collision_normal) 
                    else: 
                        if obj.on_collision_exit:
                            obj.on_collision_exit(obj, other_obj_instance, event_collision_normal)

                # --- OnCollision/TriggerEnter ---
                elif other_id in current_ids and other_id not in previous_ids:
                    if obj.is_trigger or other_obj_instance.is_trigger:
                        if obj.on_trigger_enter:
                            obj.on_trigger_enter(obj, other_obj_instance, event_collision_normal)
                    else: 
                        if obj.on_collision_enter:
                            obj.on_collision_enter(obj, other_obj_instance, event_collision_normal)

                # --- OnCollision/TriggerStay ---
                elif other_id in current_ids and other_id in previous_ids:
                    if obj.is_trigger or other_obj_instance.is_trigger:
                        if obj.on_trigger_stay:
                            obj.on_trigger_stay(obj, other_obj_instance, event_collision_normal)
                    else: 
                        if obj.on_collision_stay:
                            obj.on_collision_stay(obj, other_obj_instance, event_collision_normal)
                        
        # --- End of Frame Cleanup ---
        # Update previous_frame_overlaps for the next frame.
        # This must happen *after* all events have been dispatched.
        CollisionManager._previous_frame_overlaps = current_frame_overlaps
