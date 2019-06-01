import constants


class Walking(object):
    SCENE_1 = {
            'weather_list': constants.HARD_WEATHER,
            'timeout': 30,
            }


class Driving(object):
    SCENE_1 = {
            'weather_list': constants.WEATHER_LIST,
            'timeout': 100000,
            'vehicle_types': ['CAR'],
            'driving_style': constants.DrivingStyle.Normal,
            'random_respawn': True,
            }

    SCENE_2 = {
            'weather_list': ['EXTRASUNNY'],
            'timeout': 100000000,
            'vehicle_types': ['BIKE'],
            'driving_style': constants.DrivingStyle.Rushed,
            'in_game_time': 8 * 60 + 30,
            'random_respawn': False,
            }
