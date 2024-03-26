import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer
from modules.stochastic_module import StochasticText

class CLIPStochastic(nn.Module):
    def __init__(self, config: Config):
        super(CLIPStochastic, self).__init__()
        self.config = config
        
        from transformers import CLIPModel
        if config.clip_arch == 'ViT-B/32':
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        elif config.clip_arch == 'ViT-B/16':
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        else:
            raise ValueError


        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)
        self.stochastic = StochasticText(config)


    def forward(self, data, return_all_frames=False, is_train=True):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)

        if is_train:

            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)



            video_features = video_features.reshape(batch_size, self.config.num_frames, -1) # [bs, #F, 512]

            video_features_pooled = self.pool_frames(text_features, video_features)

            # @WJM: perform stochastic text
            text_features_stochstic, text_mean, log_var = self.stochastic(text_features, video_features)


            if return_all_frames:
                return text_features, video_features, video_features_pooled, text_features_stochstic, text_mean, log_var

            return text_features, video_features_pooled,  text_features_stochstic, text_mean, log_var

        else:

            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)



            video_features = video_features.reshape(batch_size, self.config.num_frames, -1)
            video_features_pooled = self.pool_frames(text_features, video_features)

            # @WJM: re-parameterization for text (independent of the text-cond pooling)
            text_features_stochstic, _, _ = self.stochastic(text_features, video_features)


            if return_all_frames:
                return text_features, video_features, video_features_pooled, text_features_stochstic

            return text_features, video_features_pooled, text_features_stochstic
