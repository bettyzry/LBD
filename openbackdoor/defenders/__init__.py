from .defender import Defender
from .strip_defender import STRIPDefender
from .rap_defender import RAPDefender
from .onion_defender import ONIONDefender
from .bki_defender import BKIDefender
from .cube_defender import CUBEDefender
from .loss_defender import LOSSDefender
from .lossin_defender import LossInDefender
from .z_defender import ZDefender
from .muscle_defender import MuscleDefender
from .badacts_defender import BadActs_Defender
from .att_defender import ATTDefender
from .gant_defender import GanTDefender

DEFENDERS = {
    "base": Defender,
    'strip': STRIPDefender,
    'rap': RAPDefender,
    'onion': ONIONDefender,
    'bki':  BKIDefender,
    'cube': CUBEDefender,
    'loss': LOSSDefender,
    'lossin': LossInDefender,
    'zdefence': ZDefender,
    'muscle': MuscleDefender,
    'badacts': BadActs_Defender,
    'att': ATTDefender,
    'gant': GanTDefender
}

def load_defender(config):
    if config['name'] == 'none':
        return None
    else:
        return DEFENDERS[config["name"].lower()](**config)