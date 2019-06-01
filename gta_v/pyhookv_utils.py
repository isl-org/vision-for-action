import logging
import math
import random

import pyhookv as h
import constants
import controls


######################################################################
# Namespace                                                          #
######################################################################
Controls = controls.DrivingControls
BasicControls = controls.BasicControls

is_looking_at_point = controls.is_looking_at_point
is_looking_at_angle = controls.is_looking_at_angle
get_relative_yaw_to_point = controls.get_relative_yaw_to_point

######################################################################
# Miscellaneous helpers                                              #
######################################################################
def random_point(x, lo, hi=None):
    if hi is None:
        return x + random.uniform(-lo, lo)

    return x + random.uniform(lo, hi)


######################################################################
# Higher-level PyhookV interface                                     #
######################################################################
class Vehicle(object):
    TYPES = ['UNKNOWN', 'CAR', 'BIKE', 'QUADBIKE', 'BICYCLE']
    MAX_SPEED = {
            'UNKNOWN': 0,
            'CAR': 10,
            'BIKE': 8,
            'CAR': 15,
            'BIKE': 25,
            'QUADBIKE': 30,
            'BICYCLE': 25,
            }
    MAPPING = {
            'CAR': h.Vehicle.is_this_model_a_car,
            'BIKE': h.Vehicle.is_this_model_a_bike,
            'QUADBIKE': h.Vehicle.is_this_model_a_quadbike,
            'BICYCLE': h.Vehicle.is_this_model_a_bicycle,
            }

    def __init__(self, code):
        self.code = code
        self.hash = h.Hash(code)
        self.type = 'UNKNOWN'

        # Set type.
        for vehicle_type, bool_func in Vehicle.MAPPING.items():
            if bool_func(self.hash):
                self.type = vehicle_type

        self.max_speed = Vehicle.MAX_SPEED[self.type]


class VehiclePoolMetaclass(type):
    @property
    def vehicles(cls):
        if cls._vehicles is None:
            cls._vehicles = [Vehicle(code) for code in constants.CODES_ALL_VEHICLES]

        return cls._vehicles

    @property
    def typed_vehicles(cls):
        if cls._typed_vehicles is None:
            cls._typed_vehicles = dict()

            for vehicle in cls.vehicles:
                if vehicle.type not in cls._typed_vehicles:
                    cls._typed_vehicles[vehicle.type] = list()

                cls._typed_vehicles[vehicle.type].append(vehicle)

        return cls._typed_vehicles


class VehiclePool(object, metaclass=VehiclePoolMetaclass):
    _vehicles = None
    _typed_vehicles = None

    @classmethod
    def random_vehicle(cls, type_mask=None):
        mask = list(set(type_mask) & set(cls.typed_vehicles)) or list(cls.typed_vehicles)

        return random.choice(cls.typed_vehicles[random.choice(mask)])


def pause(ticks):
    """
    No user inputs during pause.
    """
    for _ in range(ticks):
        toggle_controls(False)
        h.wait(0)


def toggle_controls(should_enable):
    """
    Must be called every frame(?).
    """
    if should_enable:
        control_func = getattr(h.Controls, 'enable_control_action')
    else:
        control_func = getattr(h.Controls, 'disable_control_action')

    for prefix in ['look', 'move']:
        for direction in ['left_right', 'up_down', 'up_only', 'down_only', 'right_only']:
            enum = getattr(h.eControl, 'control_%s_%s' % (prefix, direction))

            control_func(0, enum, True)

    control_func(0, h.eControl.control_next_camera, True)


def bad_stuff():
    player = h.Player.id()
    ped = h.Player.ped_id()
    if not ped.does_exist: return True
    if ped.is_dead: return True
    if not player.is_control_on: return True
    if player.is_being_arrested(True): return True
    return False


def get_player_ped_id(n=100):
    for _ in range(n):
        ped = h.Player.ped_id()

        if ped.does_exist():
            return ped

        h.wait(0)

    logging.warn('Cannot get pyhookv Player ped_id.')

    return None


def get_coords(ped_id):
    return ped_id.get_coords(1)


def get_xyz_from_coord(coord):
    return coord.x, coord.y, coord.z


# TODO(bradyz): still occasionally buggy.
def get_coord_near_road(x, y):
    coord = h.Pathfind.save_coord_for_ped(x, y, 0, False, 0)

    if coord is not None:
        return coord

    road_coord, _ = h.Pathfind.closest_vehicle_node_with_heading(x, y, 0, 0, 0, 0)
    road_coord_safe = h.Pathfind.save_coord_for_ped(road_coord.x, road_coord.y, 0, False, 0)

    if road_coord_safe is None:
        return road_coord

    return road_coord_safe


def get_road_heading(x, y):
    _, heading = h.Pathfind.closest_vehicle_node_with_heading(x, y, 0, 0, 0, 0)

    return heading


def get_distance_between_entities(u, v):
    a = u.get_coords(1)
    b = v.get_coords(1)

    return h.Gameplay.get_distance_between_coords(a.x, a.y, a.z, b.x, b.y, b.z, 1)


def get_distance(u, v):
    return h.Gameplay.get_distance_between_coords(u.x, u.y, 0.0, v.y, v.y, 0.0, 1)


def get_ray_cast_hit(u, v):
    """
    Must use my shitty fork of PyhookV.
    """
    a = u.get_coords(1)
    b = v.get_coords(1)

    hit_entity = h.Entity(0)

    h.Worldprobe.get_raycast_result(
            h.Worldprobe.cast_ray_point_to_point(a.x, a.y, a.z, b.x, b.y, b.z, -1, u, 7),
            0, h.Vector3(0, 0, 0), h.Vector3(0, 0, 0), hit_entity)

    return hit_entity


def get_vehicle_speed(vehicle):
    v = vehicle.velocity

    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def set_vehicle_view_mode(view_mode):
    if h.Cam.get_follow_vehicle_view_mode() != view_mode:
        h.Cam.set_follow_vehicle_view_mode(view_mode)


def set_vehicle_invincible(vehicle):
    vehicle.tyres_can_burst = False
    vehicle.set_wheels_can_break(False)
    vehicle.set_has_strong_axles(True)
    vehicle.set_can_be_visibly_damaged(False)
    vehicle.set_invincible(True)
    vehicle.set_proofs(1, 1, 1, 1, 1, 1, 1, 1)


def set_camera_zeroed():
    """
    Doesn't actually make pitch 0...
    """
    return
    # h.Cam.set_gameplay_relative_heading(0.0)
    # h.Cam.set_gameplay_relative_pitch(0.5, 0x3F800000)
    # h.Cam.clamp_gameplay_pitch(0.0, 0.0)


def set_camera_view_mode(view_mode, for_ped=False, for_vehicle=False):
    # h.Cam.set_first_person_pitch_range(0.0, 0.0)
    # h.Cam.set_gameplay_raw_pitch(0.0)
    # h.Cam.set_gameplay_relative_pitch(0.0, 0.0)
    # h.Cam.clamp_gameplay_pitch(0.0, 0.0)

    if for_ped and h.Cam.get_follow_ped_view_mode() != view_mode:
        h.Cam.set_follow_ped_view_mode(view_mode)

    if for_vehicle and h.Cam.get_follow_vehicle_view_mode() != view_mode:
        h.Cam.set_follow_vehicle_view_mode(view_mode)


def set_ped_position(ped_id, x, y=None, z=None):
    """
    Usage:
    - set_ped_position(ped_id, coord)
    - set_ped_position(ped_id, x, y)
    - set_ped_position(ped_id, x, y, z)

    HACK: set z to None for automatically finding the z.
    """
    if y is None and z is None:
        x, y, z = get_xyz_from_coord(x)
    elif z is None:
        ped_id.set_coords(x, y, constants.MAX_Z, 0, 0, 0, 1)
        z = constants.MAX_Z - ped_id.height_above_ground + 1.0

    ped_id.set_coords(x, y, z, 0, 0, 0, 1)


def set_ped_heading(ped_id, heading):
    ped_id.heading = heading


def set_ped_single_weapon(ped_id, weapon_code, infinite_ammo=False):
    h.Weapon.remove_all_ped_weapons(ped_id, 1)
    h.Weapon.give_delayed_to_ped(ped_id, h.Hash(weapon_code), 1, True)

    if infinite_ammo:
        h.Weapon.set_ped_infinite_ammo_clip(ped_id, True)


def set_ped_into_vehicle(ped_id, vehicle):
    if vehicle is not None and vehicle.does_exist():
        ped_id.set_into_vehicle(vehicle, -1)


def make_ped_never_run_away(ped_id):
    ped_id.set_blocking_of_non_temporary_events(1)
    ped_id.set_flee_attributes(0, 0)
    ped_id.set_combat_attributes(17, 1)
    ped_id.set_can_be_targetted_by_player(h.Player.id(), 1)


def make_ped_wander(ped_id, x=None, y=None, z=None, radius=10.0, length=0.1, pause_duration=0.1):
    """
    Usage:
    - make_ped_wander(ped_id)
    - make_ped_wander(ped_id, coord)
    - make_ped_wander(ped_id, x, y, z)
    """
    if x is None and y is None and z is None:
        x, y, z = get_xyz_from_coord(ped_id.get_coords(1))
    elif y is None and z is None:
        x, y, z = get_xyz_from_coord(x)

    ped_id.set_keep_task(False)

    h.Ai.clear_ped_tasks_immediately(ped_id)
    h.Ai.task_wander_in_area(ped_id, x, y, z, radius, length, pause_duration)

    ped_id.set_keep_task(True)


def make_ped_shoot_at(ped_id, x=None, y=None, z=None):
    """
    Usage:
    - make_ped_shoot_at(ped_id, coord)
    - make_ped_shoot_at(ped_id, x, y, z)
    """
    if x is not None and y is None and z is None:
        coord = x

        x = coord.x
        y = coord.y
        z = coord.z

    # ped_id.set_shoots_at_coord(x, y, z, 1)
    Controls.shoot()


def spawn_ped(ped_code, ped_type, x, y, z=None):
    """
    h.Streaming.request_model MUST be called before spawning.
    """
    if z is not None:
        return h.Ped.create_ped(ped_type, h.Hash(ped_code), x, y, z, 0.0, False, False)

    ped_id = spawn_ped(ped_code, ped_type, x, y, constants.MAX_Z)

    set_ped_position(ped_id, x, y)

    return ped_id


def spawn_vehicle(vehicle_metadata, pos, heading, set_invincible=True, n_retry=100):
    for _ in range(n_retry):
        vehicle = h.Vehicle.create_vehicle(
                vehicle_metadata.hash,
                pos.x, pos.y, pos.z, heading, False, False)

        if vehicle.does_exist():
            if set_invincible:
                set_vehicle_invincible(vehicle)

            vehicle.set_on_ground_properly()

            return vehicle

        h.wait(100)

    logging.warn(
            'Cannot spawn vehicle. Type: %s, Code: %s' % (
                vehicle_metadata.code, vehicle_metadata.type))

    return None


def spawn_entity(entity_code, spawn_func):
    """
    entity_code is a single code.
    """
    return spawn_entities([entity_code], spawn_func)


def spawn_entities(entity_codes, spawn_func):
    """
    entity_codes is a list of codes.
    """
    entity_hashes = [h.Hash(code) for code in entity_codes]

    for entity_hash in entity_hashes:
        h.Streaming.request_model(entity_hash)

    result = spawn_func()

    for entity_hash in entity_hashes:
        h.Streaming.set_model_as_no_longer_needed(entity_hash)

    return result
