import torch
import torch.nn as nn
from config.base_config import Config


class LinearCosRadius(nn.Module):
    '''
    Define the radius (R) as the linear function of the cos-similarity (t, v)
    '''
    def __init__(self, config: Config):
        super(LinearCosRadius, self).__init__()
        self.num_frames = config.num_frames
        self.embed_dim = config.embed_dim

        self.linear_proj = nn.Linear(self.num_frames, self.embed_dim)
        self.learnable_scalar = nn.Parameter(torch.Tensor(1))

        self._init_parameters()
        self.config = config

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        # normalization
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        video_embeds = video_embeds / video_embeds.norm(dim=-1, keepdim=True)

        # sim computation
        text_embeds = text_embeds.unsqueeze(1).repeat(1, self.config.num_frames, 1)
        sims = torch.matmul(text_embeds, video_embeds.permute(0,2,1))
        sims = torch.mean(sims, dim=1)

        # linear proj
        sims_out = self.linear_proj(sims)

        return sims_out


class StochasticText(nn.Module):
    def __init__(self, config: Config):
        super(StochasticText, self).__init__()

        self.config = config

        self.std_branch = LinearCosRadius(config)

    def forward(self, text_features, video_features):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_texts x embed_dim
        """
        # @WJM: re-parameterization for text (independent of the text pool video)
        text_mean = text_features

        # radius
        log_var = self.std_branch(text_features, video_features)
        text_std = torch.exp(log_var) # log var -> var

        # randomness
        if self.config.stochastic_prior == 'uniform01':
            sigma = torch.rand_like(text_features)
        elif self.config.stochastic_prior == 'normal':
            sigma = torch.normal(mean=0., std=self.config.stochastic_prior_std, size=text_features.shape).to(text_std.device)
        else:
            raise NotImplementedError

        # re-parameterization
        text_features = text_mean + sigma * text_std

        return text_features, text_mean, log_var


