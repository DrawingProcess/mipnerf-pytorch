import os.path
import shutil
from config import get_config
from scheduler import MipLRDecay
from loss import NeRFLoss, mse_to_psnr
from model import MipNeRF
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
from os import path
from datasets import get_dataloader, cycle
import numpy as np
from tqdm import tqdm


def eval(config):
    model_save_path = path.join(config.log_dir, "model.pt")
    optimizer_save_path = path.join(config.log_dir, "optim.pt")

    data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="train", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device)))
    eval_data = None
    if config.do_eval:
        eval_data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="test", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device)))

    model = MipNeRF(
        use_viewdirs=config.use_viewdirs,
        randomized=config.randomized,
        ray_shape=config.ray_shape,
        white_bkgd=config.white_bkgd,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
        device=config.device,
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    if config.continue_training:
        model.load_state_dict(torch.load(model_save_path))
        optimizer.load_state_dict(torch.load(optimizer_save_path))

    scheduler = MipLRDecay(optimizer, lr_init=config.lr_init, lr_final=config.lr_final, max_steps=config.max_steps, lr_delay_steps=config.lr_delay_steps, lr_delay_mult=config.lr_delay_mult)
    loss_func = NeRFLoss(config.coarse_weight_decay)
    model.train()

    log_dir_origin = "./log/blender/lego"
    model_origin_save_path = path.join(log_dir_origin, "model.pt")
    optimizer_origin_save_path = path.join(log_dir_origin, "optim.pt")

    model_origin = MipNeRF(
        use_viewdirs=config.use_viewdirs,
        randomized=config.randomized,
        ray_shape=config.ray_shape,
        white_bkgd=config.white_bkgd,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
        device=config.device,
    )
    optimizer = optim.AdamW(model_origin.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    if config.continue_training:
        model_origin.load_state_dict(torch.load(model_origin_save_path))
        optimizer.load_state_dict(torch.load(optimizer_save_path))

    model_origin.train()

    os.makedirs(config.log_dir, exist_ok=True)
    shutil.rmtree(path.join(config.log_dir, 'eval'), ignore_errors=True)
    logger = tb.SummaryWriter(path.join(config.log_dir, 'eval'), flush_secs=1)

    psnr = eval_model(config, model, eval_data)
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = eval_depth_model(config, model, model_origin, data)
    psnr = psnr.detach().cpu().numpy()
    logger.add_scalar('eval/coarse_psnr', float(np.mean(psnr[:-1])))
    logger.add_scalar('eval/fine_psnr', float(psnr[-1]))
    logger.add_scalar('eval/avg_psnr', float(np.mean(psnr)))
    logger.add_scalar('eval/abs_rel', float(abs_rel))
    logger.add_scalar('eval/sq_rel', float(sq_rel))
    logger.add_scalar('eval/rmse', float(rmse))
    logger.add_scalar('eval/rmse_log', float(rmse_log))
    logger.add_scalar('eval/a1', float(a1))
    logger.add_scalar('eval/a2', float(a2))
    logger.add_scalar('eval/a3', float(a3))

    print('eval/coarse_psnr: ', float(np.mean(psnr[:-1])))
    print('eval/fine_psnr: ', float(psnr[-1]))
    print('eval/avg_psnr: ', float(np.mean(psnr)))
    print('eval/abs_rel: ', float(abs_rel))
    print('eval/sq_rel: ', float(sq_rel))
    print('eval/rmse: ', float(rmse))
    print('eval/rmse_log: ', float(rmse_log))
    print('eval/a1: ', float(a1))
    print('eval/a2: ', float(a2))
    print('eval/a3: ', float(a3))

    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)


def eval_model(config, model, data):
    model.eval()
    rays, pixels = next(data)
    with torch.no_grad():
        comp_rgb, _, _ = model(rays)
    pixels = pixels.to(config.device)
    model.train()
    return torch.tensor([mse_to_psnr(torch.mean((rgb - pixels[..., :3])**2)) for rgb in comp_rgb])

def eval_depth_model(config, model, model_origin, data):
    model.eval()
    rays, pixels = next(data)
    with torch.no_grad():
        comp_rgb, dist, _ = model(rays)
        comp_rgb_origin, dist_origin, _ = model_origin(rays)
    model.train()

    return compute_depth_errors(dist_origin, dist)

# https://gaussian37.github.io/vision-depth-metrics/
def compute_depth_errors(gt, pred):   
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)
    
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    delta = torch.max((gt / pred), (pred / gt))
    print("(pred): ", torch.max(pred), torch.min(pred))
    print("(gt): ", torch.max(gt), torch.min(gt))
    a1 = (delta < 1.25     ).float().mean()
    a2 = (delta < 1.25 ** 2).float().mean()
    a3 = (delta < 1.25 ** 3).float().mean()

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

if __name__ == "__main__":
    config = get_config()
    psnr = eval(config)
