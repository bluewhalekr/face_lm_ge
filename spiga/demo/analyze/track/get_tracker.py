# Demo libs
import spiga.demo.analyze.track.retinasort.zoo as zoo_rs

zoos = [zoo_rs]


def get_tracker(model_name,device):
    for zoo in zoos:
        model = zoo.get_tracker(model_name, device)
        if model is not None:
            return model

    raise NotImplementedError('Tracker name not available')