from .wrg import generate_input_feature, get_weighted_reccurrence_graph


class WrgData:

    WIDTH = 50
    IMAGE_TYPE = "wrg"
    EPS = None
    DELTA = None

    def transform(self, current, voltage):
        input_feature = generate_input_feature(current, voltage, self.IMAGE_TYPE, self.WIDTH, False)
        v_feature = generate_input_feature(voltage, voltage, self.IMAGE_TYPE, self.WIDTH, False)
        i_feature = get_weighted_reccurrence_graph(input_feature, self.EPS, self.DELTA)
        v_feature = get_weighted_reccurrence_graph(v_feature, self.EPS, self.DELTA)
        return i_feature, v_feature


class CoolData(WrgData):
    EPS = 1e3
    DELTA = 50


class WhitedData(WrgData):
    EPS = 1e3
    DELTA = 50


class PlaidData(WrgData):
    EPS = 1e1
    DELTA = 20