class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        if key not in self.keys():
            return None
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value

    def __setattr__(self, key, value):
        self[key] = value
        