import logging
import random

import pyhookv as h
import pyhookv_utils as h_utils

import constants
import utils


INVISIBLE = False


class BaseScenario(object):
    """
    Inherit from this to create a custom scenario.
    """
    def __init__(
            self, x=None, y=None, weather=None, weather_list=None, in_game_time=None,
            view_mode=None, timeout=30, safety_timeout=1.0, seed=None, **kwargs):
        self.x = x or h_utils.random_point(0.0, constants.MIN_X, constants.MAX_X)
        self.y = y or h_utils.random_point(0.0, constants.MIN_Y, constants.MAX_Y)
        self.weather = weather or random.choice(weather_list or constants.WEATHER_LIST)
        self.in_game_time = in_game_time or random.randint(0, 24 * 60)

        if view_mode is not None:
            self.view_mode = view_mode
        else:
            self.view_mode = constants.ViewMode.FIRST_PERSON

        self.timeout_timer = utils.Timer(timeout)
        self.make_safe_timer = utils.Timer(safety_timeout)
        self.seed = seed or random.randint(0, 1000000)
        self.should_keep_running = True

    def start(self, autopilot=None):
        logging.info('##################################################')
        logging.info('Starting scenario.\nParams:\n%s' % utils.pretty_print_dict(self.params()))

        while h_utils.bad_stuff():
            h.wait(0)

        # Basic map setup.
        h.Gameplay.set_random_seed(self.seed)

        while not h.Pathfind.load_all_path_nodes():
            h.wait(0)

        # Setup weather and time.
        h.Gameplay.set_weather_type_now_persist(self.weather)
        h.Time.set_clock_time(self.in_game_time // 60, self.in_game_time % 60, 0)

        self.timeout_timer.reset()
        self.make_safe_timer.reset()

        # NOTE: method should be overriden.
        self.setup(autopilot)

        self._make_player_safe()
        self._remove_cameras()
        self._set_camera_view()

        # set_camera_zeroed()
        h_utils.toggle_controls(False)

    def setup(self, autopilot=None):
        """
        NOTE: Override this method.
        This method is in charge of spawning, setting up autopilot, etc.
        """
        pass

    def stop(self):
        logging.info('Scenario stopped.')

        self._remove_cameras()

        h_utils.toggle_controls(True)

    def update(self):
        """
        Returns True if everything is going okay.
        """
        if self.make_safe_timer.is_time():
            self._make_player_safe()

        self._set_camera_view()
        # h_utils.toggle_controls(False)

        # TODO(bradyz): change h_utils.bad_stuff into a class method.
        if h_utils.bad_stuff() or self.timeout_timer.is_time():
            self.should_keep_running = False

        return self.should_keep_running

    def params(self):
        """
        Dict of scenario start settings.
        """
        return dict(
                x=self.x,
                y=self.y,
                weather=self.weather,
                start_time=self.in_game_time,
                view_mode=self.view_mode)

    def state(self):
        """
        Dict of current scenario/player state.
        """
        coord = h_utils.get_coords(h_utils.get_player_ped_id())

        return dict(position=(coord.x,coord.y,coord.z), reward=0)

    def get_info(self):
        """
        Must override for TaskQueue to work properly.
        """
        return dict(player_ped_id=h_utils.get_player_ped_id())

    def _remove_cameras(self):
        h.Cam.destroy_all_cams(True)
        h.Cam.render_script_cams(0, 0, 0, 0, 0)

    def _set_camera_view(self):
        h_utils.set_camera_view_mode(self.view_mode, for_ped=True, for_vehicle=True)

    def _make_player_safe(self):
        player = h.Player.id()
        player.set_can_be_hassled_by_gangs(False)
        player.set_dispatch_cops_for_player(False)
        player.set_everyone_ignore_player(True)
        player.set_police_ignore_player(True)
        player.clear_wanted_level()
        player.invincible = True
        player.remove_helmet(1)

        ped = h.Player.ped_id()
        ped.set_helmet(False)
        ped.set_config_flag(32, False)
        ped.set_driver_ability(100)
        ped.set_driver_aggressiveness(50)


class WalkingScenario(BaseScenario):
    def setup(self, autopilot):
        """
        Teleport to and wander around a random road.
        """
        pos = get_coord_near_road(self.x, self.y)
        heading = get_road_heading(pos.x, pos.y)

        player_ped_id = h_utils.get_player_ped_id()

        h_utils.set_ped_position(player_ped_id, pos)
        h_utils.set_ped_heading(player_ped_id, heading)

        h_utils.pause(50)


class DrivingScenario(BaseScenario):
    def __init__(
            self, vehicle_types=None, driving_style=None,
            max_speed=None, random_respawn=None, **kwargs):
        super().__init__(**kwargs)

        self.vehicle_metadata = h_utils.VehiclePool.random_vehicle(vehicle_types or h_utils.Vehicle.TYPES)
        self.max_speed = max_speed or self.vehicle_metadata.max_speed
        self.driving_style = driving_style or constants.DrivingStyle.Normal
        self.random_respawn = random_respawn

        self.vehicle = None
        self.last_control = None

    def setup(self, autopilot):
        coord = h_utils.get_coords(h_utils.get_player_ped_id())
        h.Gameplay.clear_area(coord.x, coord.y, coord.z, 30.0, 0, 0, 0, 0)

        self._remove_all_vehicles()
        self._respawn()
        self._ready_vehicle()

        self.last_control = self.vehicle.control

        # Give autopilot information about vehicle, etc.
        autopilot.init(self)

    def update(self):
        if not super().update():
            return False
        elif self.vehicle is None or not self.vehicle.does_exist():
            logging.warn(
                    'Bad vehicle, Hash: %s, Display name: %s.' % (
                        self.vehicle_metadata.hash,
                        h.Vehicle.get_display_name_from_model(self.vehicle_metadata.hash)))

            return False

        self.last_control = self.vehicle.control

        return self.should_keep_running

    def stop(self):
        super().stop()

        self.vehicle = None
        self.last_control = None

    def params(self):
        result = super().params()
        result.update(dict(
            vehicle_code=self.vehicle_metadata.code,
            max_speed=self.max_speed,
            driving_style=self.driving_style))

        return result

    def state(self):
        result = super().state()
        result.update(dict(last_control=self.last_control))

        return result

    def _set_camera_view(self):
        h_utils.set_camera_view_mode(self.view_mode, for_ped=True, for_vehicle=True)
        h_utils.set_camera_zeroed()

    def _ready_vehicle(self):
        player_ped_id = h_utils.get_player_ped_id()
        pos = player_ped_id.get_coords(1)
        heading = player_ped_id.heading

        h_utils.set_ped_position(player_ped_id, pos)
        h_utils.set_ped_heading(player_ped_id, heading)

        h.wait(500)
        self.vehicle = h_utils.spawn_entity(
                self.vehicle_metadata.code,
                lambda: h_utils.spawn_vehicle(self.vehicle_metadata, pos, heading))

        h.wait(500)
        player_ped_id.set_into_vehicle(self.vehicle, -1)

        if INVISIBLE:
            self.vehicle.set_alpha(0, 0)
            player_ped_id.set_alpha(0, 0)

    def _respawn(self):
        player_ped_id = h_utils.get_player_ped_id()

        if self.random_respawn:
            pos = h_utils.get_coord_near_road(self.x, self.y)
            heading = h_utils.get_road_heading(pos.x, pos.y)
        else:
            pos = h_utils.get_coords(player_ped_id)
            heading = player_ped_id.heading

        h_utils.set_ped_position(player_ped_id, pos)
        h_utils.set_ped_heading(player_ped_id, heading)

    def _remove_all_vehicles(self):
        for c in h.Vehicle.list():
            h.Vehicle.delete(c)

    def get_info(self):
        result = super().get_info()
        result.update(dict(
            vehicle=self.vehicle,
            control=self.last_control,
            ))

        return result


class DrivingPointGoal(DrivingScenario):
    def __init__(self, ints=None, floats=None, **kwargs):
        super().__init__(**kwargs)

        if floats is None:
            floats = [0] * 7
            floats[0] = h_utils.get_coords(h_utils.get_player_ped_id()).x
            floats[1] = h_utils.get_coords(h_utils.get_player_ped_id()).y
            floats[2] = h_utils.get_coords(h_utils.get_player_ped_id()).z
            floats[3] = 0.0
            floats[4] = h_utils.random_point(h_utils.get_coords(h_utils.get_player_ped_id()).x, 1000)
            floats[5] = h_utils.random_point(h_utils.get_coords(h_utils.get_player_ped_id()).y, 1000)
            floats[6] = h_utils.random_point(h_utils.get_coords(h_utils.get_player_ped_id()).z, 1000)

        # Normal
        self.start_pos = h.Vector3(floats[0], floats[1], floats[2])
        self.start_heading = floats[3]
        self.end_pos = h.Vector3(floats[4], floats[5], floats[6])

        pos = h_utils.get_coords(h_utils.get_player_ped_id())

        self.last_distance = utils.dist(pos, self.end_pos)
        self.last_distance_curve = h.Pathfind.calculate_travel_distance_between_points(
                pos.x, pos.y, pos.z,
                self.end_pos.x, self.end_pos.y, self.end_pos.z)

    def _respawn(self):
        player_ped_id = h_utils.get_player_ped_id()

        h_utils.set_ped_position(player_ped_id, self.start_pos)
        h_utils.set_ped_heading(player_ped_id, self.start_heading)

    def update(self):
        if not super().update():
            return False

        self.last_distance = utils.dist(
                h_utils.get_coords(h_utils.get_player_ped_id()), self.end_pos)

        pos = h_utils.get_coords(h_utils.get_player_ped_id())
        x1, y1, z1 = pos.x, pos.y, pos.z
        x2, y2, z2 = self.end_pos.x, self.end_pos.y, self.end_pos.z

        self.last_distance_curve = h.Pathfind.calculate_travel_distance_between_points(
                x1, y1, z1, x2, y2, z1)

        return self.should_keep_running

    def state(self):
        result = super().state()
        result.update(dict(reward=self.last_distance))
        result.update(dict(reward_curve=self.last_distance_curve))

        return result
