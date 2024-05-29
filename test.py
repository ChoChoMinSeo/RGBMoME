import torch
from model.rgb_mome import RGB_MoME
from utils.count_parameter import count_parameters, count_trainable_parameters, cal_size

model = RGB_MoME(
    num_channels = 3,
    img_size = 32,
    patch_size = 4,
    emb_dim = 768,
    num_head = 12,
    sep_ffn_dim = 1024,
    comb_ffn_dim = 3072,
    sep_channel_depth = 2,
    combine_channel_depth = 2,
    num_classes = 100,
    ffn_dropout = 0.1,
    dropout = 0.1,
    activation_fn = 'gelu',
    task_type = 'classification',
    include_channel_embedding = True,
    relative_pos=True,
    multiway_patch_embedding=True,
    auxilary_output = False,
)

img = torch.randn((1,3,32,32)) # B,C,H,W
print(model(img))

print('# Trainable Parameters:',format(count_trainable_parameters(model),','))
print('# Parameters:',format(count_parameters(model),','))
print('Model Size {:.2f} MB'.format(cal_size(count_trainable_parameters(model),'float32')))