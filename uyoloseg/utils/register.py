import logging
import importlib

logger = logging.getLogger('uyoloseg')

HUB_MODULES = ["bisenetv2", "ddrnet", "pidnet", "rtformer"]
LOSSES_MODULES = ["cross_entropy_loss", "dice_loss", "focal_loss", "lovasz_softmax_loss", "ohem_cross_entropy_loss", "compose_loss"]
EVALUATOR_MODULES = ['segment', 'compose']

ALL_MODULES = [
    ("uyoloseg.models.hub", HUB_MODULES),
    # ("uyoloseg.models.backbones", None),
    # ("uyoloseg.models.heads", None),
    ("uyoloseg.models.losses", LOSSES_MODULES),
    ("uyoloseg.core.evaluator", EVALUATOR_MODULES)
]


def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    if not errors:
        return
    for name, err in errors:
        logger.warning("Module {} import failed: {}".format(name, err))


def import_all_modules_for_register(custom_module_paths=None):
    """Import all modules for register."""
    all_modules = []
    for base_dir, modules in ALL_MODULES:
        for name in modules:
            full_name = base_dir + "." + name
            all_modules.append(full_name)
    if isinstance(custom_module_paths, list):
        all_modules += custom_module_paths
    errors = []
    for module in all_modules:
        try:
            importlib.import_module(module)
        except ImportError as error:
            errors.append((module, error))
    _handle_errors(errors)

class Register:

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logger.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()
    
class registers():  # pylint: disable=invalid-name, too-few-public-methods
  """All module registers."""

  def __init__(self):
    raise RuntimeError("Registries is not intended to be instantiated")

  model_hub = Register('model_hub')
  losses = Register('losses')
  evaluator = Register('evaluator')