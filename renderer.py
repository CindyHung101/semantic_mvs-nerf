import torch
import torch.nn.functional as F
from utils.utils import normal_vect, index_point_feature, build_color_volume

# 这里的
def depth2dist(z_vals, cos_angle):
    # z_vals: [N_ray N_sample]
    device = z_vals.device
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * cos_angle.unsqueeze(-1)
    return dists

def ndc2dist(ndc_pts, cos_angle):
    dists = torch.norm(ndc_pts[:, 1:] - ndc_pts[:, :-1], dim=-1)
    dists = torch.cat([dists, 1e10*cos_angle.unsqueeze(-1)], -1)  # [N_rays, N_samples]
    return dists

# raw to output called
def raw2alpha(sigma, dist):

    alpha_softmax = F.softmax(sigma, 1)

    alpha = 1. - torch.exp(-sigma)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * T  # [N_rays, N_samples]
    return alpha, weights, alpha_softmax

#run network mvs called
#ok
def batchify(num_semantic_class, fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(num_semantic_class, inputs, alpha_only):
        if alpha_only:
            return torch.cat([fn.forward_alpha(num_semantic_class, inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        else:
            return torch.cat([fn(num_semantic_class, inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret

# models.py called
#ok
def run_network_mvs(num_semantic_class, pts, viewdirs, alpha_feat, fn, embed_fn, embeddirs_fn, netchunk=1024):
    """Prepares inputs and applies network 'fn'.getting raw = network_query_fn(rays_ndc, angle, input_feat, network_fn)
        1)embed the feature extracted (see rendering)

        2.1)adding alpha feature or not
        2.2)adding semantic feature or not
        2.3)using directions or not

        3) apply functions fn

        note:  pts, viewdirs, alpha_feat === rays_ndc, angle, input_feat (extracted features)
        
    """
    
    #=================================
    # 1
    if embed_fn is not None:
        pts = embed_fn(pts)

    # 2.1)
    if alpha_feat is not None:
        pts = torch.cat((pts,alpha_feat), dim=-1)
    # 2.2)
    #### semantic

    # 2.3)
    if viewdirs is not None:
        if viewdirs.dim()!=3:
            viewdirs = viewdirs[:, None].expand(-1,pts.shape[1],-1)

        if embeddirs_fn is not None:
            viewdirs = embeddirs_fn(viewdirs)
        pts = torch.cat([pts, viewdirs], -1)

    alpha_only = viewdirs is None

    #===================== apply function on pts, getting raw =====================
    outputs_flat = batchify(fn, netchunk)(num_semantic_class, pts, alpha_only)  # ouput  (whether predict alpha or not, whether to add viewdirs or not)
    outputs = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

# rendering called
#ok
def raw2outputs(raw, z_vals, dists, white_bkgd=False, enable_semantic = True, num_semantic_class=0 ,endpoint_feat = False,raw_noise_std = 0):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
        sem_map: [num_rays, num_semantic_class]. Estimated label of a ray.
    """

    device = z_vals.device

    rgb = raw[..., :3] # [N_rays, N_samples, 3]

    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
        noise = noise.cuda()
    else:
        noise = 0.

    alpha, weights, alpha_softmax = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]

    #------ rgb map ---------
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    #--------- semantic map -----------
    if enable_semantic:
        assert num_semantic_class>0
        # https://discuss.pytorch.org/t/multi-class-cross-entropy-loss-and-softmax-in-pytorch/24920/2
        sem_logits = raw[..., 4:4+num_semantic_class]  # [N_rays, N_samples, num_class]
        sem_map = torch.sum(weights[..., None] * sem_logits, -2)  # [N_rays, num_class]
    else:
        sem_map = torch.tensor(0)

    # --------- endpoint_feat : for semantic nerf's fine net
    # if endpoint_feat:
    #     feat = raw[..., -128:] # [N_rays, N_samples, feat_dim] take the last 128 dim from predictions
    #     feat_map = torch.sum(weights[..., None] * feat, -2)  # [N_rays, feat_dim]
    # else:
    #     feat_map = torch.tensor(0)
    
    #----------- depth, disp, acc map -------------
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
        if enable_semantic:
            sem_map = sem_map + (1.-acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map, sem_map, alpha

#ok
def gen_angle_feature(c2ws, rays_pts, rays_dir):
    """
    Inputs:
        c2ws: [1,v,4,4]
        rays_pts: [N_rays, N_samples, 3]
        rays_dir: [N_rays, 3]

    Returns:

    """
    N_rays, N_samples = rays_pts.shape[:2]
    dirs = normal_vect(rays_pts.unsqueeze(2) - c2ws[:3, :3, 3][None, None])  # [N_rays, N_samples, v, 3]
    angle = torch.sum(dirs[:, :, :3] * rays_dir.reshape(N_rays,1,1,3), dim=-1, keepdim=True).reshape(N_rays, N_samples, -1)
    return angle

#ok
def gen_dir_feature(w2c_ref, rays_dir):
    """
    Inputs:
        c2ws: [1,v,4,4]
        rays_pts: [N_rays, N_samples, 3]
        rays_dir: [N_rays, 3]

    Returns:

    """
    dirs = rays_dir @ w2c_ref[:3,:3].t() # [N_rays, 3]
    return dirs

# ok
def gen_pts_feats(imgs, volume_feature, rays_pts, pose_ref, rays_ndc, feat_dim, img_feat=None, img_downscale=1.0, use_color_volume=False, net_type='v0'):
    N_rays, N_samples = rays_pts.shape[:2]
    if img_feat is not None:
        feat_dim += img_feat.shape[1]*img_feat.shape[2]

    if not use_color_volume:
        input_feat = torch.empty((N_rays, N_samples, feat_dim), device=imgs.device, dtype=torch.float)
        ray_feats = index_point_feature(volume_feature, rays_ndc) if torch.is_tensor(volume_feature) else volume_feature(rays_ndc)
        input_feat[..., :8] = ray_feats
        input_feat[..., 8:] = build_color_volume(rays_pts, pose_ref, imgs, img_feat, with_mask=True, downscale=img_downscale)
    else:
        input_feat = index_point_feature(volume_feature, rays_ndc) if torch.is_tensor(volume_feature) else volume_feature(rays_ndc)
    return input_feat

def rendering(args, pose_ref, rays_pts, rays_ndc, depth_candidates, rays_o, rays_dir,
              volume_feature=None, imgs=None, network_fn=None, img_feat=None, network_query_fn=None, num_semantic_class = 0, white_bkgd=False, **kwargs ):
    
    #================ get view dir feature map angle(unwarped) ==================================
    
    cos_angle = torch.norm(rays_dir, dim=-1)    # rays angle

    if pose_ref is not None:   
        # use_viewdirs = true, using direction
        angle = gen_dir_feature(pose_ref['w2cs'][0], rays_dir/cos_angle.unsqueeze(-1))  # view dir feature
    else:   
        # use_viewdirs = false, compute unit directions corresponding to rays_dir
        angle = rays_dir/cos_angle.unsqueeze(-1)    


    #======================== get rays pts feature map (unwarped) ================================
    input_feat = gen_pts_feats(imgs, volume_feature, rays_pts, pose_ref, rays_ndc, args.feat_dim, \
                               img_feat, args.img_downscale, args.use_color_volume, args.net_type)

    


    #==========================将光线上的每个点投入到 MLP 网络 network_fn 中 : 把feature map聚合成raw，之後作為network fn的input ==================================
    # rays_ndc = rays_ndc * 2 - 1.0
    # network_query_fn --> run_network_mvs() --> actual function: self.semmvs_model_coarse 
    network_query_fn = kwargs["network_query_fn"]
    raw = network_query_fn(num_semantic_class, rays_ndc, angle, input_feat, network_fn)  # semmvs_model_coarse get RGBAS
    if raw.shape[-1]>4:
        input_feat = torch.cat((input_feat[...,:8],raw[...,4:]), dim=-1)

#ok
    dists = depth2dist(depth_candidates, cos_angle)
    # dists = ndc2dist(rays_ndc)

# get output
    rgb_map, disp_map, acc_map, weights, depth_map, sem_map, alpha = raw2outputs(raw, depth_candidates, dists, white_bkgd,args.net_type, num_semantic_class, endpoint_feature =False)
    ret = {}

    return rgb_map, input_feat, weights, depth_map, sem_map, alpha, ret
    

def render_density(network_fn, rays_pts, density_feature,  network_query_fn, chunk=1024 * 5):
    densities = []
    device = density_feature.device
    for i in range(0, rays_pts.shape[0], chunk):

        input_feat = rays_pts[i:i + chunk].to(device)

        density = network_query_fn(input_feat, None, density_feature[i:i + chunk], network_fn)
        densities.append(density)

    return torch.cat(densities)