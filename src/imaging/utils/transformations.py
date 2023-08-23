from monai.transforms import MapTransform
import monai.transforms as mt


class CustomRandAffine(MapTransform):
    def __init__(self, keys, *args, **kwargs):
        super().__init__(keys)
        self.transform = mt.RandAffine(*args, **kwargs)

    def __call__(self, data):
        for key in self.keys:
            data[key] = self.transform(data[key])
        return data


class CustomRand3DElastic(MapTransform):
    def __init__(self, keys, *args, **kwargs):
        super().__init__(keys)
        self.transform = mt.Rand3DElastic(*args, **kwargs)

    def __call__(self, data):
        for key in self.keys:
            data[key] = self.transform(data[key])
        return data


class CustomRandGaussianNoise(MapTransform):
    def __init__(self, keys, *args, **kwargs):
        super().__init__(keys)
        self.transform = mt.RandGaussianNoise(*args, **kwargs)

    def __call__(self, data):
        for key in self.keys:
            data[key] = self.transform(data[key])
        return data
