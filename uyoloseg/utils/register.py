import logging
import importlib

HUB_MODULES = ["BiSeNetV2", "DDRNet", "PIDNet", "RTFormer"]

ALL_MODULES = [
    ("uyoloseg.models.hub", HUB_MODULES)
]


def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    if not errors:
        return
    for name, err in errors:
        logging.warning("Module {} import failed: {}".format(name, err))


def import_all_modules_for_register(custom_module_paths=None):
    """Import all modules for register."""
    all_modules = []
    for base_dir, modules in ALL_MODULES:
        for name in modules:
            full_name = base_dir + "." + name
            print(full_name)
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
            logging.warning("Key %s already in registry %s." % (key, self._name))
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
    
if __name__ == "__main__":
    print("Register.model._dict before: ", registers.model_hub._dict)
    import_all_modules_for_register()
    print("Register.model._dict after: ", registers.model_hub._dict)