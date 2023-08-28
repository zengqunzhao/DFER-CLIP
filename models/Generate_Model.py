from torch import nn
from models.Temporal_Model import *
from models.Prompt_Learner import *


class GenerateModel(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.input_text = input_text
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)

        self.clip_model_ = clip_model

    def forward(self, image):
        
        ################# Visual Part #################
        n, t, c, h, w = image.shape
        image = image.contiguous().view(-1, c, h, w)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features.contiguous().view(n, t, -1)
        video_features = self.temporal_net(image_features)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        ###############################################
        
        ################## Text Part ##################
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts) 
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
        ###############################################

        output = video_features @ text_features.t() / 0.01

        return output
