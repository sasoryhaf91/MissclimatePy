
REGISTRY = {}
def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco
