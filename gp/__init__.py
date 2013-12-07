from . import ext
__all__ = ["ext"]

from .gp import GP
__all__.append("GP")

from .kernels import *
__all__.extend(kernels.__all__)

import logging
FORMAT = '%(levelname)s -- %(processName)s/%(filename)s -- %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("gp")
logger.setLevel("INFO")
