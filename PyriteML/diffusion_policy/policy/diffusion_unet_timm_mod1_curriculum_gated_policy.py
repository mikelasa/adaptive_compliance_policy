from diffusion_policy.policy.diffusion_unet_timm_mod1_policy import (
    DiffusionUnetTimmMod1Policy,
)


class DiffusionUnetTimmMod1CurriculumGatedPolicy(DiffusionUnetTimmMod1Policy):
    """
    DiffusionUnetTimmMod1Policy + contact-gate wrench-normalizer wiring for
    TimmObsEncoderWithForceCurriculumGated. Mirrors the getattr/hasattr-
    guarded wiring in DiffusionTransformerTimmMod1Policy (RACP's policy) --
    see set_wrench_normalizer() on the obs_encoder for why a live reference
    is stored rather than a snapshot (survives model_io.py's whole
    state_dict reload at inference, which never calls set_normalizer()).
    Kept as a separate subclass rather than editing
    DiffusionUnetTimmMod1Policy in place, since that class is shared by
    several other active UNet baseline configs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.obs_encoder, "contact_gate_enabled", False) and hasattr(
            self.obs_encoder, "set_wrench_normalizer"
        ):
            wrench_keys = getattr(self.obs_encoder, "wrench_keys", [])
            if wrench_keys:
                self.obs_encoder.set_wrench_normalizer(self.sparse_normalizer, wrench_keys[0])
