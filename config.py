from super_image.configuration_utils import PretrainedConfig

from super_image.data.datasets import (
    DIV2K_RGB_MEAN,
    DIV2K_RGB_STD
)

class AlanConfig(PretrainedConfig):
    model_type = 'ALAN'

    def __init__(self, scale: int = None, n_resblocks=16, n_feats=64, n_colors=3, rgb_range=255,
                 rgb_mean=DIV2K_RGB_MEAN, rgb_std=DIV2K_RGB_STD, no_upsampling=False,
                 res_scale=1, data_parallel=False, batch_norm=False, n_stage=2,
                 n_up_acb=1, upsampling_type='nearest', **kwargs):
        """
        Args:
            scale (int): Scale for the model to train an upscaler/super-res model.
            n_resblocks (int): Number of residual blocks.
            n_feats (int): Number of filters.
            n_colors (int):
                Number of color channels.
            rgb_range (int):
                Range of RGB as a multiplier to the MeanShift.
            res_scale (int):
                The res scale multiplier.
            rgb_mean (tuple):
                The RGB mean of the train dataset.
                You can use `~super_image.utils.metrics.calculate_mean_std` to calculate it.
            rgb_std (tuple):
                The RGB standard deviation of the train dataset.
                You can use `~super_image.utils.metrics.calculate_mean_std` to calculate it.
            no_upsampling (bool):
                Option to turn off upsampling.
            data_parallel (bool):
                Option to use multiple GPUs for training.
        """
        super().__init__(**kwargs)
        self.scale = scale
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.n_colors = n_colors
        self.rgb_range = rgb_range
        self.res_scale = res_scale
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.no_upsampling = no_upsampling
        self.data_parallel = data_parallel

        self.use_batch_norm = batch_norm
        self.n_stage = n_stage
        self.n_up_acb = n_up_acb
        self.upsampling_type = upsampling_type
