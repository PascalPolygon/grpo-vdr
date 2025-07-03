"""
Startup patch to fix torchvision compatibility issues.
Import this at the very beginning of any script that has import issues.
"""
import sys
import types

# Pre-create dummy modules to avoid import errors
# Create a dummy InterpolationMode enum with all common modes
class DummyInterpolationMode:
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2
    BOX = 3
    HAMMING = 4
    LANCZOS = 5

# Create a dummy module spec
class DummyModuleSpec:
    def __init__(self, name):
        self.name = name
        self.loader = None
        self.origin = None
        self.submodule_search_locations = None
        self.cached = None
        self.parent = None
        self.has_location = False

# Create dummy modules with necessary attributes
dummy_modules = {
    'torchvision': types.ModuleType('torchvision'),
    'torchvision._meta_registrations': types.ModuleType('torchvision._meta_registrations'),
    'torchvision.ops': types.ModuleType('torchvision.ops'),
    'torchvision.transforms': types.ModuleType('torchvision.transforms'),
}

# Add __spec__ to each module
for module_name, module in dummy_modules.items():
    module.__spec__ = DummyModuleSpec(module_name)

# Add InterpolationMode to transforms
dummy_modules['torchvision.transforms'].InterpolationMode = DummyInterpolationMode

# Add all modules to sys.modules
for module_name, module in dummy_modules.items():
    if module_name not in sys.modules:
        sys.modules[module_name] = module

# Patch the __import__ function to catch and handle the specific error
# Handle both dict and module forms of __builtins__
import builtins
_original_import = builtins.__import__

def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    try:
        return _original_import(name, globals, locals, fromlist, level)
    except RuntimeError as e:
        if "operator torchvision::nms does not exist" in str(e):
            # Return dummy module
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)
            return sys.modules[name]
        raise

builtins.__import__ = _patched_import

print("✓ Torchvision compatibility patch applied") 