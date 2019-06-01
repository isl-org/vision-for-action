# Keyboard input.
F6 = 0x75
F7 = 0x76
F8 = 0x77
F9 = 0x78
F10 = 0x79
F11 = 0x7A

# Positions.
MIN_X = -2500.0
MAX_X = 2500.0

MIN_Y = -2000.0
MAX_Y = 6000.0

MIN_Z = 0.0
MAX_Z = 10000.0

# Vehicle codes.
CODES_BICYCLES = [0x43779c54, 0x1aba13b5, 0xce23d3bf, 0xf4e1aa15, 0x4339cd69, 0xb67597ec, 0xe823fb48]
# CODES_MOTORCYCLES = [0x63abade7, 0x806b9cc3, 0xf9300cc5, 0xcadd5d2d, 0xabb0c0, 0x77934cee, 0x9c669788, 0x6882fa73]
CODES_MOTORCYCLES = [0x2EF89E46]
CODES_QUADS = [0x8125bcf9, 0xfd231729, 0xb44f0582]
# CODES_CARS = [0xb779a091, 0x4c80eb0e, 0x5d0aac8f, 0x2db8d1aa, 0x45d56ada, 0x94204d89, 0x9441d8d5, 0x8e9254fb]
# CODES_CARS = [0xb779a091]
CODES_CARS = [1348744438]
CODES_ALL_VEHICLES = CODES_BICYCLES + CODES_MOTORCYCLES + CODES_QUADS + CODES_CARS

# Animal codes.
CODE_RABBIT = 0xDFB55C81
CODE_DEER = 0xd86b5a95
CODE_BOAR = 0xce5ff074
CODE_COYOTE = 0x644ac75e
CODES_ALL_ANIMALS = [CODE_DEER, CODE_BOAR, CODE_COYOTE, CODE_RABBIT]
ANIMAL_PED_TYPE = 28

# Weapons.
CODE_AK47 = 0xbfefff6d
CODE_AR = 0x83bf0278
CODE_SINGLE_SHOT = 0x5d60e4e0
CODE_FULL_AUTO = 0xc6ee6b4c

# Weather modes.
EASY_WEATHER = ['CLEAR', 'EXTRASUNNY', 'CLOUDS', 'OVERCAST']
HARD_WEATHER = ['RAIN', 'CLEARING', 'THUNDER', 'SMOG']
WEATHER_LIST = EASY_WEATHER + HARD_WEATHER

# Mmap paths.
CLIENT_MMAP_PATH = 'C:\\Users\\Brady\\Documents\\code\\vision_tasks\\gta_v\\client.txt'
SERVER_MMAP_PATH = 'C:\\Users\\Brady\\Documents\\code\\vision_tasks\\gta_v\\server.txt'

CLIENT_MMAP_PATH_LINUX = '/mnt/c/Users/Brady/Documents/code/vision_tasks/gta_v/client.txt'
SERVER_MMAP_PATH_LINUX = '/mnt/c/Users/Brady/Documents/code/vision_tasks/gta_v/server.txt'

TARGETS = ['final', 'disparity', 'object_id', 'texture_id', 'flow', 'albedo', 'velocity']


class DrivingStyle(object):
    AvoidTraffic = 786468
    AvoidTrafficExtremely = 6
    IgnoreLights = 2883621
    Normal = 786603
    Rushed = 1074528293
    SometimesOvertakeTraffic = 5


class ViewMode(object):
    """
    Skips 3 for some reason.
    """
    THIRD_PERSON_CLOSE = 0
    THIRD_PERSON_MEDIUM = 1
    THIRD_PERSON_FAR = 2
    FIRST_PERSON = 4


class AgentStatus(object):
    NOT_STARTED = 0
    RUNNING = 1
    NOISE = 2
    TERMINATED = 3


class AgentType(object):
    NONE = 0
    WALKING_AUTOPILOT = 1
    DRIVING_AUTOPILOT = 2
    MANUAL = 3


class EnvType(object):
    NONE = 0
    WALKING_AUTOPILOT = 1
    DRIVING_AUTOPILOT = 2
    MANUAL = 3
