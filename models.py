
from pyexpat import model
from re import L
import torch
import os
from cgitb import enable
from distutils.command.config import config
from doctest import OutputChecker
from unittest import skip
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import imgviz


#from inplace_abn import InPlaceABN
import math
import yaml

#from visualisation.tensorboard_vis import TFVisualizer

# import from local
from utils.image_utils import *
from utils.utils import homo_warp
from renderer import run_network_mvs

# code refer to mvs nerf models

#=========
def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        self.freq_bands = freq_bands.reshape(1,-1,1).cuda()

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        repeat = inputs.dim()-1
        inputs_scaled = (inputs.unsqueeze(-2) * self.freq_bands.view(*[1]*repeat,-1,1)).reshape(*inputs.shape[:-1],-1)
        inputs_scaled = torch.cat((inputs, torch.sin(inputs_scaled), torch.cos(inputs_scaled)),dim=-1)
        return inputs_scaled

def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.ReLU(out_f)
    )

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

#========================================
class semmvs_Renderer(nn.Module):
    def __init__(self,num_semantic_class, D=8, W=256, in_ch_pts=3, in_ch_views=3, out_ch_pts=5, in_ch_feats=8, skips=[4], use_viewdirs=False, enable_semantic = True):
        """
        """
        super(semmvs_Renderer, self).__init__()
        
        self.D = D
        self.W = W
        self.in_ch_pts = in_ch_pts
        self.in_ch_views = in_ch_views
        self.in_ch_feats= in_ch_feats 
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.enable_semantic = enable_semantic
        self.num_semantic_class = num_semantic_class
        
        # build the encoder
        # pts linear and bias
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.in_ch_pts, W,bias = True)] \
            + [nn.Linear(W, W, bias = True) if i not in self.skips else nn.Linear(W + self.in_ch_pts, W) for i in range(D-1)] 
        )
        self.pts_bias = nn.ModuleList(
            [nn.Linear(self.in_ch_feats, W)]
        )
        # view linear
        self.views_linears = nn.ModuleList(
            [nn.Linear(self.in_ch_views+ W, W//2)]
        )

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            if enable_semantic:
                self.semantic_linear = nn.Sequential(fc_block(W, W // 2), nn.Linear(W // 2, num_semantic_class))
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            output_linear = nn.ModuleList(
                nn.Linear(W, out_ch_pts)
            )

        # 初始化 sem没有但是mvs有 要回来确认sem的初始化写在哪里
        self.pts_linears.apply(weights_init)
        self.views_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.semantic_linear.apply(weights_init)
        self.rgb_linear.apply(weights_init)

    def forward_alpha(self, x): # alpha only

        dim = x.shape[-1]
        in_ch_feats = dim-self.in_ch_pts
        input_pts, input_feats = torch.split(x, [self.in_ch_pts, in_ch_feats], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = torch.relu(self.alpha_linear(h))
        return alpha

    def forward(self, x, show_endpoint = False):   #同时预测alpha以及rgb的MLP
        """
        Encodes input (xyz+dir) to rgb+sigma+semantics raw output
        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of 3D xyz position and viewing direction
        """
        dim = x.shape[-1]
        in_ch_feats = dim-self.in_ch_pts-self.in_ch_views
        input_pts, input_feats, input_views = torch.split(x, [self.in_ch_pts, in_ch_feats, self.in_ch_views], dim=-1)

        # pts的linear
        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # view的linear
        if self.use_viewdirs:
            alpha = torch.relu(self.alpha_linear(h))
            if self.enable_semantic:
                sem_logits = self.semantic_linear(h)
            feature = self.feature_linear(h)

            # h加入input_views的信息
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            if show_endpoint:
                endpoint_feat = h
            rgb = torch.sigmoid(self.rgb_linear(h)) # sem写的是rgb = self.rgb_linear(h)
            
            if self.enable_semantic:
                outputs = torch.cat([rgb,alpha,sem_logits],-1)
            else:
                outputs = torch.cat([rgb,alpha],-1)
            # else是直接不需要rgb的MLP
        else:
            outputs = self.output_linear(h)

        if show_endpoint is False:  #sem的部分 但是还不是很清楚endpoint都低在干嘛
            return outputs
        else:
            return torch.cat([outputs,endpoint_feat],-1)


class semmvsNeRF(nn.Module):
    def __init__(self, enable_semantic, num_semantic_class, D=8, W=256, in_ch_pts=3, in_ch_views=3, in_ch_feats=8, skips=[4]):
        super(semmvsNeRF, self).__init__()
        self.in_ch_pts, self.in_ch_views, self.in_ch_feats = in_ch_pts, in_ch_views, in_ch_feats
        self.enable_semantic = enable_semantic
        self.nerf = semmvs_Renderer(    num_semantic_class,
                                        D=D, W=W,
                                        in_ch_feats=in_ch_feats,
                                        in_ch_pts=in_ch_pts, 
                                        out_ch_pts=5, 
                                        skips=skips,
                                        in_ch_views=in_ch_views,
                                        use_viewdirs=True)

    def forward_alpha(self, x):
        return self.nerf.forward_alpha()

    def forward(self, x):
        RGBAS = self.nerf(x) #调用nn.Module的时候会自动调用forward
        return RGBAS

#=========================================
def create_semmvs(args, use_mvs=False,dir_embedder=True, pts_embedder=True, num_valid_semantic_class = 40):
    """Instantiate mvs NeRF's MLP model.
    """
    #============================ embedding ============================                                                              
    if pts_embedder:    
        embed_fn, in_ch_pts = get_embedder(args.multires, args.i_embed, input_dims=3) # get_embedder: get_embedder(multires, i=0, input_dims=3) --> return embed, embedder_obj.out_dim
    else:
        embed_fn, in_ch_pts = None, 3

    if dir_embedder:
        embeddirs_fn, in_ch_views = get_embedder(args.multires_views, args.i_embed, input_dims=3)
    else:
        embeddirs_fn, in_ch_views = None, 3

    #============================ model ============================
    skips = [4]
    model = semmvsNeRF(     
                            enable_semantic = True,
                            num_semantic_class = 0, 
                            D = 8, W = 256,
                            in_ch_pts = in_ch_pts, 
                            in_ch_views = in_ch_views, 
                            in_ch_feats = 8, 
                            skips = skips
                    )     
        
    grad_vars = []
    grad_vars+=list(model.parameters())  
           

    #============================ fine  model   ============================
    model_fine = None
    if args.N_importance > 0:   
        print("-------------- N_importance = 0 .. fine model  ------------")
        model_fine = semmvsNeRF(
                                    enable_semantic = True, 
                                    num_semantic_class = num_valid_semantic_class,
                                    D = 8, W = 256, 
                                    in_ch_pts = in_ch_pts, 
                                    in_ch_views = in_ch_views, 
                                    in_ch_feats = 8, 
                                    skips = skips
                                )

        grad_vars += list(model_fine.parameters())

    #=========================== network function ==========================
    #model             #semmvs_Renderer
    #model_fine
            

    # 2.3
    #============================ Create optimizer ============================
    #optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate)
    
        
        
        
    # 3
    #============================ network_query_fn : 得到经过embed之后的output（包括是否加入viewdirs）
        # note that the input from : renderer ---> run_network_mvs  # renderer.py
    network_query_fn = lambda num_semantic_class, pts, viewdirs, rays_feats, network_fn: run_network_mvs(
                                                                                                            num_semantic_class, pts, viewdirs, rays_feats, network_fn,
                                                                                                            embed_fn=embed_fn,
                                                                                                            embeddirs_fn=embeddirs_fn,
                                                                                                            netchunk=1024
                                                                                                        )
                                                                        # raw = network_query_fn(rays_ndc, angle, input_feat, network_fn)
    
    # 4 encodingnet
    EncodingNet = None
    EncodingNet = MVSNet().to(device)
    grad_vars += list(EncodingNet.parameters())
    start = 0

    # 5 load checkpoints
    ckpts = []
    if args.ckpt is not None and args.ckpt != 'None':
        ckpts = [args.ckpt]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 :
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # Load model
        if use_mvs:
            state_dict = ckpt['network_mvs_state_dict']
            EncodingNet.load_state_dict(state_dict) 

        model.load_state_dict(ckpt['network_fn_state_dict'])

    # 6 kwargs
    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'network_mvs': EncodingNet,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }


    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars
            


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################################     MVS Net models        ################################################
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=nn.BatchNorm2d):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)


    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=nn.BatchNorm3d):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        # self.bn = nn.ReLU()

    def forward(self, x):
        return self.bn(self.conv(x))

###################################  feature net  ######################################
# Class MVSNet(): self.feature = FeatureNet()
class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=nn.BatchNorm3d):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv0(x) # (B, 8, H, W)
        x = self.conv1(x) # (B, 16, H//2, W//2)
        x = self.conv2(x) # (B, 32, H//4, W//4)
        x = self.toplayer(x) # (B, 32, H//4, W//4)

        return x

# cost正则化网络模块
class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=nn.BatchNorm3d):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        # x = self.conv12(x)
        return x

# create nerf mvs的encoding net调用这个class
class MVSNet(nn.Module):
    def __init__(self,
                 num_groups=1,
                 norm_act=nn.BatchNorm3d,
                 levels=1):
        super(MVSNet, self).__init__()
        self.levels = levels  # 3 depth levels
        self.n_depths = [128,32,8]  #D = 128, f = 32(channels), 
        self.G = num_groups  # number of groups in groupwise correlation
        self.feature = FeatureNet() ##return feature map(B, 32, H//4, W//4)

        self.N_importance = 0   
        self.chunk = 1024

        self.cost_reg_2 = CostRegNet(32+9, norm_act)

    ##获得cost volume（里面已经包含了warping--调用home_warp函数）
    def build_volume_costvar(self, feats, proj_mats, depth_values, pad=0):
        # feats: (B, V, C, H, W)    #V view？D应该是ref map frustum上面取的深度数量
        # proj_mats: (B, V, 3, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        # volume_sum [B, G, D, h, w]
        # prob_volume [B D H W]
        # volume_feature [B C D H W]

        B, V, C, H, W = feats.shape # batch view channel height weight
        D = depth_values.shape[1]   # depth_value = B D H W--> batch, depth for ref. frustum, H, W

        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)   #permute就是变换维度
        proj_mats = proj_mats[:, 1:]
        proj_mats = proj_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)

        if pad > 0:
            ref_feats = F.pad(ref_feats, (pad, pad, pad, pad), "constant", 0)

################## 跟下面的不一样的地方
        #专门处理ref_volume（猜测是因为ref_volume不用去做warping）
        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, h, w) 在第二个维度重复D次（复制tensor数据）

        volume_sum = ref_volume #volume_sum = ref_volume + warped volume（所以在下面的部分获得warped volume）
        volume_sq_sum = ref_volume ** 2

        del ref_feats

        #warping and get warped src volume
        in_masks = torch.ones((B, 1, D, H + pad * 2, W + pad * 2), device=volume_sum.device)
        for i, (src_feat, proj_mat) in enumerate(zip(src_feats, proj_mats)):    #zip 打包成tuple(a,b)   # 对每一个src feature map及其proj mat都做homegraphic warping
            warped_volume, grid = homo_warp(src_feat, proj_mat, depth_values, pad=pad)  #返回的是B, C, Ndepth, H, W（每个batch的ref map，在每个深度Ndepth都有一张HW的feature map）
            # warped_volume(B, C, D, H, W) grid(B ,D, H_pad, W_pad, 2)
####################

            #？？？？？？？？？in_mask这里不太懂oAo
            grid = grid.view(B, 1, D, H + pad * 2, W + pad * 2, 2)  # 变换shape但是数据其实是一样的 ## dim with pad
            in_mask = ((grid > -1.0) * (grid < 1.0))
            in_mask = (in_mask[..., 0] * in_mask[..., 1])
            in_masks += in_mask.float()

            if self.training:   # 如果是training的话就不inplace warped_volume的平方
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume 
                volume_sq_sum += warped_volume.pow_(2)  ## warped volume**2 inplace ## for testing (var based evaluation at test)

            del warped_volume, src_feat, proj_mat
        del src_feats, proj_mats

        count = 1.0 / in_masks
        img_feat = volume_sq_sum * count - (volume_sum * count) ** 2    # volume sum是所有warped volume加起来 volume_sq_sum是所有的warped_volume的平方加起来
        del volume_sq_sum, volume_sum, count

        return img_feat, in_masks

    ## 其实还是不太懂这里是在干嘛oAo
    def build_volume_costvar_img(self, imgs, feats, proj_mats, depth_values, pad=0):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V, 3, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        # volume_sum [B, G, D, h, w]
        # prob_volume [B D H W]
        # volume_feature [B C D H W]

        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]
        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        proj_mats = proj_mats[:, 1:]
        proj_mats = proj_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)

        if pad > 0:
            ref_feats = F.pad(ref_feats, (pad, pad, pad, pad), "constant", 0)

################ 
        img_feat = torch.empty((B, 9 + 32, D, *ref_feats.shape[-2:]), device=feats.device, dtype=torch.float)
        imgs = F.interpolate(imgs.view(B * V, *imgs.shape[2:]), (H, W), mode='bilinear', align_corners=False).view(B, V,-1,H,W).permute(1, 0, 2, 3, 4)
        img_feat[:, :3, :, pad:H + pad, pad:W + pad] = imgs[0].unsqueeze(2).expand(-1, -1, D, -1, -1)

        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, h, w)

        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        del ref_feats
###################

        in_masks = torch.ones((B, V, D, H + pad * 2, W + pad * 2), device=volume_sum.device)
        for i, (src_img, src_feat, proj_mat) in enumerate(zip(imgs[1:], src_feats, proj_mats)):
            warped_volume, grid = homo_warp(src_feat, proj_mat, depth_values, pad=pad)
            img_feat[:, (i + 1) * 3:(i + 2) * 3], _ = homo_warp(src_img, proj_mat, depth_values, src_grid=grid, pad=pad)

            grid = grid.view(B, 1, D, H + pad * 2, W + pad * 2, 2)
            in_mask = ((grid > -1.0) * (grid < 1.0))
            in_mask = (in_mask[..., 0] * in_mask[..., 1])
            in_masks[:, i + 1] = in_mask.float()

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)

            del warped_volume, src_feat, proj_mat
        del src_feats, proj_mats

        count = 1.0 / torch.sum(in_masks, dim=1, keepdim=True)
        img_feat[:, -32:] = volume_sq_sum * count - (volume_sum * count) ** 2   # view之间的var
        del volume_sq_sum, volume_sum, count

        return img_feat, in_masks

    def forward(self, imgs, proj_mats, near_far, pad=0,  return_color=False, lindisp=False):
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V, 3, 4) from fine to coarse
        # init_depth_min, depth_interval: (B) or float
        # near_far (B, V, 2)

        B, V, _, H, W = imgs.shape

        imgs = imgs.reshape(B * V, 3, H, W)
        feats = self.feature(imgs)  # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)

        imgs = imgs.view(B, V, 3, H, W)


        feats_l = feats  # (B*V, C, h, w)

        feats_l = feats_l.view(B, V, *feats_l.shape[1:])  # (B, V, C, h, w)


        D = 128
        t_vals = torch.linspace(0., 1., steps=D, device=imgs.device, dtype=imgs.dtype)  # (B, D)
        near, far = near_far  # assume batch size==1
        if not lindisp:
            depth_values = near * (1.-t_vals) + far * (t_vals)
        else:
            depth_values = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        depth_values = depth_values.unsqueeze(0)
        # volume_feat, in_masks = self.build_volume_costvar(feats_l, proj_mats, depth_values, pad=pad)
        volume_feat, in_masks = self.build_volume_costvar_img(imgs, feats_l, proj_mats, depth_values, pad=pad)
        if return_color:
            feats_l = torch.cat((volume_feat[:,:V*3].view(B, V, 3, *volume_feat.shape[2:]),in_masks.unsqueeze(2)),dim=2)


        volume_feat = self.cost_reg_2(volume_feat)  # (B, 1, D, h, w)
        volume_feat = volume_feat.reshape(1,-1,*volume_feat.shape[2:])

        return volume_feat, feats_l, depth_values


class RefVolume(nn.Module):
    def __init__(self, volume):
        super(RefVolume, self).__init__()

        self.feat_volume = nn.Parameter(volume)

    def forward(self, ray_coordinate_ref):
        '''coordinate: [N, 3]
            z,x,y
        '''

        device = self.feat_volume.device
        H, W = ray_coordinate_ref.shape[-3:-1]
        grid = ray_coordinate_ref.view(-1, 1, H, W, 3).to(device) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
        features = F.grid_sample(self.feat_volume, grid, align_corners=True, mode='bilinear')[:, :, 0].permute(2, 3, 0,1).squeeze()
        return features


