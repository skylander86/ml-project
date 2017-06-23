from importlib import import_module


__all__ = ['get_class_from_module_path']


def get_class_from_module_path(path):
    try: module_path, class_name = path.rsplit('.', 1)
    except ValueError: raise ImportError('{} is not a valid module path. You need to specify the full Python dotted path to the module.')

    module = import_module(module_path)
    klass = getattr(module, class_name)

    return klass
#end def
