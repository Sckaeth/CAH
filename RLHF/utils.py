import numpy as np
from copy import deepcopy


# Defines the adaptive KL divergence.
class AdaptiveKLController:
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


def create_reference_model(model):
    # Deep copies the base model into a reference model.
    ref_model = deepcopy(model)

    # Freezes the copied model's parameters.
    parameters = [param for param, _ in model.named_parameters()]
    for param in parameters:
        param = ref_model.get_parameter(param)
        param.requires_grad = False

    # Sets the copied model into evaluation mode, off from training mode.
    return ref_model.eval()
