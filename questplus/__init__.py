from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("questplus")
except PackageNotFoundError:
    # package is not installed
    pass

from .qp import QuestPlus, QuestPlusWeibull, QuestPlusThurstone  # noqa: F401
