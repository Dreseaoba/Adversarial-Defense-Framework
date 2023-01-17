DEFENSE_NAMES = [
    'SpatialSmoothing',
    'JPEG',
    'BitDepth',
    'PIN',
    'DAE',
    'DUNET',
    'MultiFilter'
]

def get_defense(model_name:str, *args, **kwargs):
    if model_name == DEFENSE_NAMES[0]:
        from .SpatialSmoothing import SpatialSmoothingTorch
        # from .SpatialSmoothing import SpatialSmoothing
        return SpatialSmoothingTorch(*args, **kwargs)
    elif model_name == DEFENSE_NAMES[1]:
        from .JPEG import Jpeg_compresssion
        return Jpeg_compresssion(*args, **kwargs)
    elif model_name == DEFENSE_NAMES[2]:
        from .BitDepth import BitDepthReduction
        return BitDepthReduction(*args, **kwargs)
    elif model_name == DEFENSE_NAMES[3]:
        from .PerturbationInactivation import PerturbationInactivation
        return PerturbationInactivation(*args, **kwargs)
    elif model_name == DEFENSE_NAMES[4]:
        from .DenoisingAutoEncoder import DenoisingAutoEncoder
        return DenoisingAutoEncoder(*args, **kwargs)
    elif model_name == DEFENSE_NAMES[5]:
        from .DUNET import DUNet
        return DUNet(*args, **kwargs)
    elif model_name == DEFENSE_NAMES[6]:
        from .MultiFulter import MultiFulter
        return MultiFulter(*args, **kwargs)
    else:
        raise NotImplementedError(
            'No defense model named {}.\nAll available:\n{}'.format(model_name, DEFENSE_NAMES))
