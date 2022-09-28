import torch
from opt import config_parser
import configargparse
import yaml
from tqdm import trange
import time
import imageio

from utils.utils import *
from torch.utils.data import DataLoader

# import from local
from renderer import *
from models import *
from models import semmvsNeRF, semmvs_Renderer

from data.scannet3 import ScanNet_Dataset


# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, depth_pred, depth_gt, mask=None):
        if None == mask:
            mask = depth_gt > 0
        loss = self.loss(depth_pred[mask], depth_gt[mask]) * 2 ** (1 - 2)
        return loss

class MVSSystem(LightningModule):
    def __init__(self, args):
        super(MVSSystem, self).__init__()
        self.args = args
        self.args.feat_dim = 8+3*4
        self.idx = 0

        self.loss = SL1Loss()
        self.learning_rate = args.lrate

        # Create nerf model
        ## create_nerf_mvs() : return render_kwargs_train, render_kwargs_test, start, grad_vars
        
        self.render_kwargs_train, self.render_kwargs_test, start, self.grad_vars = create_semmvs(args = args, use_mvs=True, dir_embedder=False, pts_embedder=True, num_valid_semantic_class = 20)
        ########### filter_keys(self.render_kwargs_train)   # dict.pop('N_samples')('ndc')('lindisp')三个key-value值

        # Create mvs model
        self.MVSNet = self.render_kwargs_train['network_mvs']   # encodingnet
        self.render_kwargs_train.pop('network_mvs')
        self.render_kwargs_train['NDC_local'] = False

        self.eval_metric = [0.01,0.05, 0.1]


    def decode_batch(self, batch, idx=list(torch.arange(4))):

        data_mvs = sub_selete_data(batch, device, idx, filtKey=[])  # return batch dictionary with essential key-value
        pose_ref = {'w2cs': data_mvs['w2cs'].squeeze(), 'intrinsics': data_mvs['intrinsics'].squeeze(),
                    'c2ws': data_mvs['c2ws'].squeeze(),'near_fars':data_mvs['near_fars'].squeeze()}

        return data_mvs, pose_ref

    def unpreprocess(self, data, shape=(1,1,3,1,1)):
        # to unnormalize image for visualization
        # data N V C H W
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std

    def forward(self):
        return

    def prepare_data(self):
        dataset = ScanNet_Dataset
        root_dir = self.args.datadir
        print("root dir for data : ",str(root_dir)) #C:/Users/cindyhung/Desktop/semantic_mvs nerf5/data/scans
        self.train_dataset = dataset(split = 'train', img_h = self.args.img_h, img_w = self.args.img_w, root_dir=root_dir , max_len=-1)
        self.val_dataset   = dataset(split = 'val', img_h = self.args.img_h, img_w = self.args.img_w, root_dir=root_dir , max_len=-1)#

    def configure_optimizers(self):
        eps = 1e-7
        self.optimizer = torch.optim.Adam(self.grad_vars, lr=self.learning_rate, betas=(0.9, 0.999))
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.num_epochs, eta_min=eps)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=0,
                          batch_size=1,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=0,
                          batch_size=1,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):

        if 'scan' in batch.keys():  
            batch.pop('scan')

        

        #============ get data and pose ==================
        # get batch key-value including poses
        data_mvs, pose_ref = self.decode_batch(batch)   
        imgs      = data_mvs['images']      # imgs      (V, H, W, 3)
        proj_mats = data_mvs['proj_mats']   
        near_fars = data_mvs['near_fars']
        depths_h  = data_mvs['depths_h']    # depth_h   (V, H, W)
        num_semantic_class = data_mvs['num_semantic_class']
        semantic_remap = data_mvs['semantic_remap']




        #============= create semantic mvs nerf model ============
        # see __init__ create mvs model
        # see .model 
        volume_feature, img_feat, depth_values = self.MVSNet(imgs[:, :3], proj_mats[:, :3], near_fars[0,0],pad=args.pad)
        imgs = self.unpreprocess(imgs)  # to unnormalize image for visualization

        #============== build rays for one views (having N_rays sampled rays at diff depth)=================================
        N_rays, N_samples = args.batch_size, args.N_samples
        c2ws, w2cs, intrinsics = pose_ref['c2ws'], pose_ref['w2cs'], pose_ref['intrinsics'] # pose_ref: from self.decode_batch(batch)在上面几行定义的
        rays_pts, rays_dir, target_s, rays_NDC, depth_candidates, rays_o, rays_depth, ndc_parameters, rays_semantic = \
            build_rays(semantic_remap, imgs, depths_h, pose_ref, w2cs, c2ws, intrinsics, near_fars, N_rays, N_samples, pad=args.pad)

        #=============== render ==============================
        rgb, disp, acc, depth_pred, sem_pred, alpha, ret = rendering(args, pose_ref, rays_pts, rays_NDC, depth_candidates, rays_o, rays_dir,
                                                       volume_feature, imgs[:, :-1], img_feat=None, num_semantic_class= num_semantic_class, **self.render_kwargs_train)
        
        #=============== logging ==============================
        log, total_loss = {},0
        if self.args.with_depth:
            mask = rays_depth > 0
            if self.args.with_depth_loss:
                total_loss += self.loss(depth_pred, rays_depth, mask)

            self.log(f'train/acc_l_{self.eval_metric[0]}mm', acc_threshold(depth_pred, rays_depth, mask, self.eval_metric[0]).mean(), prog_bar=False)
            self.log(f'train/acc_l_{self.eval_metric[1]}mm', acc_threshold(depth_pred, rays_depth, mask, self.eval_metric[1]).mean(), prog_bar=False)
            self.log(f'train/acc_l_{self.eval_metric[2]}mm', acc_threshold(depth_pred, rays_depth, mask, self.eval_metric[2]).mean(), prog_bar=False)

            abs_err = abs_error(depth_pred, rays_depth, mask).mean()
            self.log('train/abs_err', abs_err, prog_bar=True)

        ##################  rendering   #####################
        # sample rays to query and optimise

        img_loss = img2mse(rgb, target_s)
        total_loss = total_loss + img_loss
        if 'rgb0' in ret:
            img_loss_coarse = img2mse(ret['rgb0'], target_s)
            psnr = mse2psnr2(img_loss_coarse.item())
            self.log('train/PSNR_coarse', psnr.item(), prog_bar=True)
            total_loss = total_loss + img_loss_coarse
        # semantic loss
        if self.args.enable_semantic:
            sem_loss_coarse = crossentropy_loss(sem_pred, rays_semantic)
        else:
            sem_loss_coarse = torch.tensor(0)


        wgt_sem_loss = float(self.config["train"]["wgt_sem"])
        total_loss = total_loss + sem_loss_coarse*wgt_sem_loss


        if args.with_depth:
            psnr = mse2psnr(img2mse(rgb.cpu()[mask], target_s.cpu()[mask]))
            psnr_out = mse2psnr(img2mse(rgb.cpu()[~mask], target_s.cpu()[~mask]))
            self.log('train/PSNR_out', psnr_out.item(), prog_bar=True)
        else:
            psnr = mse2psnr2(img_loss.item())

        with torch.no_grad():
            self.log('train/loss', total_loss, prog_bar=True)
            self.log('train/img_mse_loss', img_loss.item(), prog_bar=False)
            self.log('train/PSNR', psnr.item(), prog_bar=True)

        if self.global_step % 20000==19999:
            self.save_ckpt(f'{self.global_step}')


        return  {'loss':total_loss}

    def validation_step(self, batch, batch_nb):

        if 'scan' in batch.keys():
            batch.pop('scan')

        log = {}

        #============ get data and pose ==================
        # get batch key-value including poses
        data_mvs, pose_ref = self.decode_batch(batch)   
        imgs      = data_mvs['images']      # imgs      (V C H W)
        proj_mats = data_mvs['proj_mats']   
        near_fars = data_mvs['near_fars']
        depths_h  = data_mvs['depths_h']    # depth_h   (V, H, W)
        num_semantic_class = data_mvs['num_semantic_class']
        semantic_remap = data_mvs['semantic_remap']

        #============== create encoding net ================
        self.MVSNet.train() # encoding net
        H, W = imgs.shape[-2:]
        H, W = int(H), int(W)


        ##################  rendering #####################
        keys = ['val_psnr', 'val_depth_loss_r', 'val_abs_err', 'mask_sum'] + [f'val_acc_{i}mm' for i in self.eval_metric]
        log = init_log(log, keys)

        with torch.no_grad():

            args.img_downscale = torch.rand((1,)) * 0.75 + 0.25  # for super resolution
            world_to_ref = pose_ref['w2cs'][0]
            tgt_to_world, intrinsic = pose_ref['c2ws'][-1], pose_ref['intrinsics'][-1]
            volume_feature, img_feat, _ = self.MVSNet(imgs[:, :3], proj_mats[:, :3], near_fars[0], pad=args.pad)
            imgs = self.unpreprocess(imgs)
            rgbs, depth_preds, semantic_preds = [],[], []

            #========================= render rays chunk ===============================
            # process in chunk = 1024(number of rays processed in parallel, decrease if running out of memory)
            for chunk_idx in range(H*W//args.chunk + int(H*W%args.chunk>0)):


                rays_pts, rays_dir, rays_NDC, depth_candidates, rays_o, ndc_parameters = \
                    build_rays_test(H, W, tgt_to_world, world_to_ref, intrinsic, near_fars, near_fars[-1], args.N_samples, pad=args.pad, chunk=args.chunk, idx=chunk_idx)
                    # diff from build_rays : no color, rays_depth, semantic_rays


                #------ render, get prediction of one chunk ------
                rgb, disp, acc, depth_pred, sem_pred, density_ray, ret = rendering(args, pose_ref, rays_pts, rays_NDC, depth_candidates, rays_o, rays_dir,
                                                       volume_feature, imgs[:, :-1], img_feat=None, num_semantic_class = num_semantic_class, **self.render_kwargs_train)
                #------- append the prediction of one chunk together ------
                rgbs.append(rgb.cpu())
                depth_preds.append(depth_pred.cpu())
                # semnatic 
                logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
                semantic_preds_label = logits_2_label(sem_pred)
                semantic_preds_label.append(sem_pred.cpu())


            #=========================== rendered rays chunk  ========
            imgs = imgs.cpu()
            rgb = torch.clamp(torch.cat(rgbs).reshape(H, W, 3).permute(2,0,1),0,1)  # (3, H, W)
            depth_r = torch.cat(depth_preds).reshape(H, W)
            semantic_r = torch.cat(semantic_preds_label).reshape(H, W)

            


            #========================== logging ==========================
            img_err_abs = (rgb - imgs[0,-1]).abs()  # imgs[0][-1]   # 为什么是最后一个image才是gt

            if self.args.with_depth:
                depth_gt_render = depths_h[0, -1].cpu() 
                mask = depth_gt_render > 0
                log['val_psnr'] = mse2psnr(torch.mean(img_err_abs[:,mask] ** 2))
            else:
                log['val_psnr'] = mse2psnr(torch.mean(img_err_abs**2))
            # assume enable semantic = true
            semantic_gt_render = semantic_remap[0,-1].cpu()


            #============== visualize depth ================================
            if self.args.with_depth:

                log['val_depth_loss_r'] = self.loss(depth_r, depth_gt_render, mask)
                
                #------- visualize depth -----------
                minmax = [2.0,6.0]  # ？？？？？？？
                depth_gt_render_vis,_ = visualize_depth(depth_gt_render,minmax)
                depth_pred_r_, _ = visualize_depth(depth_r, minmax)
                depth_err_, _ = visualize_depth(torch.abs(depth_r-depth_gt_render)*5, minmax)

                
                img_vis_depth = torch.stack((depth_gt_render_vis, depth_pred_r_, depth_err_))
                self.logger.experiment.add_images('val/depth_gt_pred_err', img_vis_depth, self.global_step)
                
                log['val_abs_err'] = abs_error(depth_r, depth_gt_render, mask).sum()
                log[f'val_acc_{self.eval_metric[0]}mm'] = acc_threshold(depth_r, depth_gt_render, mask, self.eval_metric[0]).sum()
                log[f'val_acc_{self.eval_metric[1]}mm'] = acc_threshold(depth_r, depth_gt_render, mask, self.eval_metric[1]).sum()
                log[f'val_acc_{self.eval_metric[2]}mm'] = acc_threshold(depth_r, depth_gt_render, mask, self.eval_metric[2]).sum()
                log['mask_sum'] = mask.float().sum()
                
            else:
                minmax = [2.0, 6.0]
                depth_pred_r_, _ = visualize_depth(depth_r, minmax)
                self.logger.experiment.add_images('val/depth_gt_pred_err', depth_pred_r_[None], self.global_step)
                


            #============ visualize semantic =====================================
            log['val_semantic_loss_r'] = self.loss(semantic_r, semantic_gt_render)

            
            # assume enable semantic = true
            minmax = [2.0,6.0]
            semantic_gt_render_vis,_ = visualize_semantic(semantic_gt_render, minmax)
            semantic_pred_r_,_ = visualize_semantic(semantic_r, minmax)
            semantic_err_, _ = visualize_semantic(torch,abs(semantic_r - semantic_gt_render)*5, minmax)

            img_vis_semantic = torch.stack((semantic_gt_render_vis, semantic_pred_r_, semantic_err_))
            self.logger.experiment.add_images('val/zemantic_gt_pred_err', img_vis_semantic, self.global_step)
           


            #============= images ===============================================
            imgs = imgs[0]
            img_vis = torch.cat((imgs, torch.stack((rgb, img_err_abs.cpu()*5))), dim=0) # N 3 H W
            self.logger.experiment.add_images('val/rgb_pred_err', img_vis, self.global_step)



            #===========================================================================================
            os.makedirs(f'runs_new/{self.args.expname}/{self.args.expname}/',exist_ok=True)     
            img_vis = torch.cat((img_vis,depth_pred_r_[None], semantic_pred_r_[None]),dim=0).permute(2,0,3,1).reshape(img_vis.shape[2],-1,3).numpy()

            imageio.imwrite(f'runs_new/{self.args.expname}/{self.args.expname}/{self.global_step:08d}_{self.idx:02d}.png', (img_vis*255).astype('uint8'))
            self.idx += 1

        del rays_NDC, rays_dir, rays_pts, volume_feature
        return log

    def validation_epoch_end(self, outputs):


        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        mask_sum = torch.stack([x['mask_sum'] for x in outputs]).sum()
        mean_d_loss_r = torch.stack([x['val_depth_loss_r'] for x in outputs]).mean()
        mean_abs_err = torch.stack([x['val_abs_err'] for x in outputs]).sum() / mask_sum
        mean_acc_1mm = torch.stack([x[f'val_acc_{self.eval_metric[0]}mm'] for x in outputs]).sum() / mask_sum
        mean_acc_2mm = torch.stack([x[f'val_acc_{self.eval_metric[1]}mm'] for x in outputs]).sum() / mask_sum
        mean_acc_4mm = torch.stack([x[f'val_acc_{self.eval_metric[2]}mm'] for x in outputs]).sum() / mask_sum

        self.log('val/d_loss_r', mean_d_loss_r, prog_bar=False)
        self.log('val/PSNR', mean_psnr, prog_bar=False)
        self.log('val/abs_err', mean_abs_err, prog_bar=False)
        self.log(f'val/acc_{self.eval_metric[0]}mm', mean_acc_1mm, prog_bar=False)
        self.log(f'val/acc_{self.eval_metric[1]}mm', mean_acc_2mm, prog_bar=False)
        self.log(f'val/acc_{self.eval_metric[2]}mm', mean_acc_4mm, prog_bar=False)

        return


    def save_ckpt(self, name='latest'):
        save_dir = f'runs_new/{self.args.expname}/ckpts/'
        os.makedirs(save_dir, exist_ok=True)
        path = f'{save_dir}/{name}.tar'
        ckpt = {
            'global_step': self.global_step,
            'network_fn_state_dict': self.render_kwargs_train['network_fn'].state_dict(),
            'network_mvs_state_dict': self.MVSNet.state_dict()}
        if self.render_kwargs_train['network_fine'] is not None:
            ckpt['network_fine_state_dict'] = self.render_kwargs_train['network_fine'].state_dict()
        torch.save(ckpt, path)
        print('Saved checkpoints at', path)

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    args = config_parser()
    system = MVSSystem(args)
    checkpoint_callback = ModelCheckpoint(os.path.join(f'runs_new/{args.expname}/ckpts/','{epoch:02d}'),
                                          monitor='val/PSNR',
                                          mode='max',
                                          save_top_k=0)

    logger = loggers.TensorBoardLogger(
        # save_dir="runs_new",
        # name=args.expname,
        # debug=False,
        # create_git_tag=False
        save_dir="runs_new",
        name=args.expname


    )

    args.num_gpus, args.use_amp = 1, False
    trainer = Trainer(max_epochs=args.num_epochs,
                      #checkpoint_callback=checkpoint_callback,
                      callbacks=checkpoint_callback,
                      logger=logger,
                      #weights_summary=None,
                      #progress_bar_refresh_rate=1,
                      gpus=args.num_gpus,
                      #distributed_backend='ddp' if args.num_gpus > 1 else None,
                      num_sanity_val_steps=1,
                      check_val_every_n_epoch = max(system.args.num_epochs//system.args.N_vis,1),
                      benchmark=True,
                      precision=16 if args.use_amp else 32,
                      #amp_level='O1'
                      )

    trainer.fit(system)
    system.save_ckpt()
    torch.cuda.empty_cache()
