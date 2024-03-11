
from torch import nn 
from torch.nn import functional as F
import torch
from einops import rearrange
import torch
import torch.nn as nn
###########Resnet Block############
def normalization(planes, norm = 'instance'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'instance':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError("Does not support this kind of norm.")
    return m
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, norm = 'instance'):
        super().__init__()
        self.resnetblock = nn.Sequential(
            normalization(in_channels, norm = norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size = 3, padding = 1),
            normalization(in_channels, norm = norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size = 3, padding = 1)
        )

    def forward(self, x):
        y = self.resnetblock(x)
        return y + x


##############VAE###############
def calculate_total_dimension(a):
    res = 1
    for x in a:
        res *= x
    return res

class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim, num_channels):
        super().__init__()
        self.input_shape = input_shape
        self.in_channels = input_shape[1]  #input_shape[0] is batch size
        self.latent_dim = latent_dim
        self.encoder_channels = self.in_channels // 16

        #Encoder
        self.VAE_reshape = nn.Conv3d(self.in_channels, self.encoder_channels,
                     kernel_size = 3, stride = 2, padding=1)
        # self.VAE_reshape = nn.Sequential(
        #     nn.GroupNorm(8, self.in_channels),
        #     nn.ReLU(),
        #     nn.Conv3d(self.in_channels, self.encoder_channels,
        #              kernel_size = 3, stride = 2, padding=1),
        # )

        flatten_input_shape =  calculate_total_dimension(input_shape)
        flatten_input_shape_after_vae_reshape = \
            flatten_input_shape * self.encoder_channels // (8 * self.in_channels)

        #Convert from total dimension to latent space
        self.to_latent_space = nn.Linear(
            flatten_input_shape_after_vae_reshape // self.in_channels, 1)

        self.mean = nn.Linear(self.in_channels, self.latent_dim)
        self.logvar = nn.Linear(self.in_channels, self.latent_dim)
#         self.epsilon = nn.Parameter(torch.randn(1, latent_dim))

        #Decoder
        self.to_original_dimension = nn.Linear(self.latent_dim, flatten_input_shape_after_vae_reshape)
        self.Reconstruct = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                self.encoder_channels, self.in_channels,
                stride = 1, kernel_size = 1),
            nn.Upsample(scale_factor=2, mode = 'nearest'),

            nn.Conv3d(
                self.in_channels, self.in_channels // 2,
                stride = 1, kernel_size = 1),
            nn.Upsample(scale_factor=2, mode = 'nearest'),
            ResNetBlock(self.in_channels // 2),

            nn.Conv3d(
                self.in_channels // 2, self.in_channels // 4,
                stride = 1, kernel_size = 1),
            nn.Upsample(scale_factor=2, mode = 'nearest'),
            ResNetBlock(self.in_channels // 4),

            nn.Conv3d(
                self.in_channels // 4, self.in_channels // 8,
                stride = 1, kernel_size = 1),
            nn.Upsample(scale_factor=2, mode = 'nearest'),
            ResNetBlock(self.in_channels // 8),

            nn.InstanceNorm3d(self.in_channels // 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                self.in_channels // 8, num_channels,
                kernel_size = 3, padding = 1),
#             nn.Sigmoid()
        )


    def forward(self, x):   #x has shape = input_shape
        #Encoder
        # print(x.shape)
        x = self.VAE_reshape(x)
        shape = x.shape

        x = x.view(self.in_channels, -1)
        x = self.to_latent_space(x)
        x = x.view(1, self.in_channels)

        mean = self.mean(x)
        logvar = self.logvar(x)
#         sigma = torch.exp(0.5 * logvar)
        # Reparameter
        epsilon = torch.randn_like(logvar)
        sample = mean + epsilon * torch.exp(0.5*logvar)

        #Decoder
        y = self.to_original_dimension(sample)
        y = y.view(*shape)
        return self.Reconstruct(y), mean, logvar
    def total_params(self):
        total = sum(p.numel() for p in self.parameters())
        return format(total, ',')

    def total_trainable_params(self):
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return format(total_trainable, ',')


# x = torch.rand((1, 256, 16, 16, 16))
# vae = VAE(input_shape = x.shape, latent_dim = 256, num_channels = 4)
# y = vae(x)
# print(y[0].shape, y[1].shape, y[2].shape)
# print(vae.total_trainable_params())


### Decoder ####



class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size = 1)
        self.deconv = nn.ConvTranspose3d(out_channel, out_channel, kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv3d(out_channel * 2, out_channel, kernel_size = 1)

    def forward(self, prev, x):
        x = self.deconv(self.conv1(x))
        y = torch.cat((prev, x), dim = 1)
        return self.conv2(y)

class FinalConv(nn.Module): # Input channels are equal to output channels
    def __init__(self, in_channels, out_channels=32, norm="instance"):
        super(FinalConv, self).__init__()
        if norm == "batch":
            norm_layer = nn.BatchNorm3d(num_features=in_channels)
        elif norm == "group":
            norm_layer = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm3d(in_channels)

        self.layer = nn.Sequential(
            norm_layer,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            )
    def forward(self, x):
        return self.layer(x)

class Decoder(nn.Module):
    def __init__(self, img_dim, patch_dim, embedding_dim, num_classes = 3):
        super().__init__()
        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.embedding_dim = embedding_dim

        self.decoder_upsample_1 = Upsample(128, 64)
        self.decoder_block_1 = ResNetBlock(64)

        self.decoder_upsample_2 = Upsample(64, 32)
        self.decoder_block_2 = ResNetBlock(32)

        self.decoder_upsample_3 = Upsample(32, 16)
        self.decoder_block_3 = ResNetBlock(16)

        self.endconv = FinalConv(16, num_classes)
        # self.normalize = nn.Sigmoid()

    def forward(self, x1, x2, x3, x):
        x = self.decoder_upsample_1(x3, x)
        x = self.decoder_block_1(x)

        x = self.decoder_upsample_2(x2, x)
        x = self.decoder_block_2(x)

        x = self.decoder_upsample_3(x1, x)
        x = self.decoder_block_3(x)

        y = self.endconv(x)
        return y



###############Encoder##############
class InitConv(nn.Module):
    def __init__(self, in_channels = 4, out_channels = 16, dropout = 0.2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.Dropout3d(dropout)
        )
    def forward(self, x):
        y = self.layer(x)
        return y


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)
    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels, dropout = 0.2):
        super().__init__()

        self.init_conv = InitConv(in_channels, base_channels, dropout = dropout)
        self.encoder_block1 = ResNetBlock(in_channels = base_channels)
        self.encoder_down1 = DownSample(base_channels, base_channels * 2)

        self.encoder_block2_1 = ResNetBlock(base_channels * 2)
        self.encoder_block2_2 = ResNetBlock(base_channels * 2)
        self.encoder_down2 = DownSample(base_channels * 2, base_channels * 4)

        self.encoder_block3_1 = ResNetBlock(base_channels * 4)
        self.encoder_block3_2 = ResNetBlock(base_channels * 4)
        self.encoder_down3 = DownSample(base_channels * 4, base_channels * 8)

        self.encoder_block4_1 = ResNetBlock(base_channels * 8)
        self.encoder_block4_2 = ResNetBlock(base_channels * 8)
        self.encoder_block4_3 = ResNetBlock(base_channels * 8)
        self.encoder_block4_4 = ResNetBlock(base_channels * 8)
        # self.encoder_down3 = EncoderDown(base_channels * 8, base_channels * 16)
    def forward(self, x):
        x = self.init_conv(x) #(1, 16, 128, 128, 128)

        x1 = self.encoder_block1(x)
        x1_down = self.encoder_down1(x1)  #(1, 32, 64, 64, 64)

        x2 = self.encoder_block2_2(self.encoder_block2_1(x1_down))
        x2_down = self.encoder_down2(x2) #(1, 64, 32, 32, 32)

        x3 = self.encoder_block3_2(self.encoder_block3_1(x2_down))
        x3_down = self.encoder_down3(x3) #(1, 128, 16, 16, 16)

        output = self.encoder_block4_4(
                            self.encoder_block4_3(
                            self.encoder_block4_2(
                            self.encoder_block4_1(x3_down))))  #(1, 256, 16, 16, 16)
        return x1, x2, x3, output

# x = torch.rand((1, 4, 128, 128, 128))
# Enc = Encoder(4, 32)
# _, _, _, y = Enc(x)
# print(y.shape) (1,256,16,16)


###############FeatureMapping###############

class FeatureMapping(nn.Module):
    def __init__(self, in_channel, out_channel, norm = 'instance'):
        super().__init__()
        if norm == 'bn':
            norm_layer_1 = nn.BatchNorm3d(out_channel)
            norm_layer_2 = nn.BatchNorm3d(out_channel)
        elif norm == 'gn':
            norm_layer_1 = nn.GroupNorm(8, out_channel)
            norm_layer_2 = nn.GroupNorm(8, out_channel)
        elif norm == 'instance':
            norm_layer_1 = nn.InstanceNorm3d(out_channel)
            norm_layer_2 = nn.InstanceNorm3d(out_channel)
        self.feature_mapping = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size = 3, padding = 1),
            norm_layer_1,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channel, out_channel, kernel_size = 3, padding = 1),
            norm_layer_2,
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.feature_mapping(x)


class FeatureMapping1(nn.Module):
    def __init__(self, in_channel, norm = 'instance'):
        super().__init__()
        if norm == 'bn':
            norm_layer_1 = nn.BatchNorm3d(in_channel)
            norm_layer_2 = nn.BatchNorm3d(in_channel)
        elif norm == 'gn':
            norm_layer_1 = nn.GroupNorm(8, in_channel)
            norm_layer_2 = nn.GroupNorm(8, in_channel)
        elif norm == 'instance':
            norm_layer_1 = nn.InstanceNorm3d(in_channel)
            norm_layer_2 = nn.InstanceNorm3d(in_channel)
        self.feature_mapping1 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size = 3, padding = 1),
            norm_layer_1,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channel, in_channel, kernel_size = 3, padding = 1),
            norm_layer_2,
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        y = self.feature_mapping1(x)
        return x + y #Resnet Like

################Transformer#######################


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, function):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.function = function

    def forward(self, x):
        return self.function(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.0):
        super().__init__()
        all_head_size = heads * dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.softmax = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, all_head_size * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(all_head_size, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        #(batch, heads * dim_head) -> (batch, all_head_size)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots =  torch.matmul(q, k.transpose(-1, -2)) * self.scale

        atten = self.softmax(dots)

        out = torch.matmul(atten, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    def forward(self, x):
        for attention, feedforward in self.layers:
            x = attention(x) + x
            x = feedforward(x) + x
        return x

class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=768):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.seq_length = seq_length
        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim)) #8x

    def forward(self, x, position_ids=None):
        position_embeddings = self.position_embeddings
#         print(x.shape, self.position_embeddings.shape)
        return x + position_embeddings





###############Main model#################

class SegTransVAE(nn.Module):
    def __init__(self, img_dim, patch_dim, num_channels, num_classes,
                embedding_dim, num_heads, num_layers, hidden_dim, in_channels_vae,
                dropout = 0.0, attention_dropout = 0.0,
                conv_patch_representation = True, positional_encoding = 'learned',
                use_VAE = False):

        super().__init__()
        assert embedding_dim % num_heads == 0
        assert img_dim[0] % patch_dim == 0 and img_dim[1] % patch_dim == 0 and img_dim[2] % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.in_channels_vae = in_channels_vae
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.conv_patch_representation = conv_patch_representation
        self.use_VAE = use_VAE

        self.num_patches = int((img_dim[0] // patch_dim) * (img_dim[1] // patch_dim) * (img_dim[2] // patch_dim))
        self.seq_length = self.num_patches
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.embedding_dim, self.seq_length
            )
        elif positional_encoding == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        self.pe_dropout = nn.Dropout(self.dropout)

        self.transformer = Transformer(
            embedding_dim, num_layers, num_heads, embedding_dim // num_heads,  hidden_dim, dropout
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:
            self.conv_x = nn.Conv3d(128, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.encoder = Encoder(self.num_channels, 16)
        self.bn = nn.InstanceNorm3d(128)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.FeatureMapping = FeatureMapping(in_channel = self.embedding_dim, out_channel= self.in_channels_vae)
        self.FeatureMapping1 = FeatureMapping1(in_channel = self.in_channels_vae)
        self.decoder = Decoder(self.img_dim, self.patch_dim, self.embedding_dim, num_classes)

        self.vae_input = (1, self.in_channels_vae, img_dim[0] // 8, img_dim[1] // 8, img_dim[2] // 8)
        if use_VAE:
            self.vae = VAE(input_shape = self.vae_input , latent_dim= 256, num_channels= self.num_channels)
    def encode(self, x):
        if self.conv_patch_representation:
            x1, x2, x3, x = self.encoder(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)
        x = self.transformer(x)
        x = self.pre_head_ln(x)

        return x1, x2, x3, x

    def decode(self, x1, x2, x3, x):
        #x: (1, 4096, 512) -> (1, 16, 16, 16, 512)
#         print("In decode...")
#         print(" x1: {} \n x2: {} \n x3: {} \n x: {}".format( x1.shape, x2.shape, x3.shape, x.shape))
#         break
        return self.decoder(x1, x2, x3, x)

    def forward(self, x, is_validation = True):
        x1, x2, x3, x = self.encode(x)
        x = x.view( x.size(0),
                    self.img_dim[0] // self.patch_dim,
                    self.img_dim[1] // self.patch_dim,
                    self.img_dim[2] // self.patch_dim,
                    self.embedding_dim)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.FeatureMapping(x)
        x = self.FeatureMapping1(x)
        if self.use_VAE and not is_validation:
            vae_out, mu, sigma = self.vae(x)
        y = self.decode(x1, x2, x3, x)
        if self.use_VAE and not is_validation:
            return y, vae_out, mu, sigma
        else:
            return y


