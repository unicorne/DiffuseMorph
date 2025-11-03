import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
from math import *
import time
from util.visualizer import Visualizer
from PIL import Image
import numpy as np
from monai.metrics import DiceMetric
from model.deformation_net_2D import Dense2DSpatialTransformer
import monai
LNCC_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=2)
MI_loss = monai.losses.GlobalMutualInformationLoss()
SSIM_loss = monai.losses.ssim_loss.SSIMLoss(spatial_dims=2)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    visualizer = Visualizer(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase != opt['phase']: continue
        if opt['phase'] == 'train':
            train_set = Data.create_dataset_2D(dataset_opt, phase=phase, mri=True)
            batchSize = opt['datasets']['train']['batch_size']
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            training_iters = int(ceil(train_set.data_len / float(batchSize)))
        elif opt['phase'] == 'test':
            # RAFD_dataset needs to be modified to load masks
            test_set = Data.create_dataset_2D(dataset_opt, phase=phase, mri=True)
            test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    if opt['phase'] == 'train':
        current_step = diffusion.begin_step
        current_epoch = diffusion.begin_epoch
        n_epoch = opt['train']['n_epoch']
        if opt['path']['resume_state']:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

        while current_epoch < n_epoch:
            current_epoch += 1
            for istep, train_data in enumerate(train_loader):
                iter_start_time = time.time()
                current_step += 1

                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if (istep+1) % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    t = (time.time() - iter_start_time) / batchSize
                    visualizer.print_current_errors(current_epoch, istep+1, training_iters, logs, t, 'Train')
                    visualizer.plot_current_errors(current_epoch, (istep+1) / float(training_iters), logs)
                    visuals = diffusion.get_current_visuals_train()
                    visualizer.display_current_results(visuals, current_epoch, True)

                # validation
                if (istep+1) % opt['train']['val_freq'] == 0:
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
                    diffusion.test_generation(continuous=False)
                    diffusion.test_registration(continuous=False)
                    visuals = diffusion.get_current_visuals()
                    visualizer.display_current_results(visuals, current_epoch, True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')

            if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        initial_dice_scores = []
        registered_dice_scores = []
        initial_ssims = []
        registered_ssims = []
        initial_lnccs = []
        registered_lnccs = []
        initial_mis = []
        registered_mis = []
        stn = Dense2DSpatialTransformer()

        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for istep,  test_data in enumerate(test_loader):
            idx += 1
            fileInfo = test_data['P']
            dataXinfo, dataYinfo = fileInfo[0][0][:-4], fileInfo[1][0][:-4]

            data_origin = test_data['M'].squeeze().cpu().numpy()
            data_fixed = test_data['F'].squeeze().cpu().numpy()

            time1 = time.time()
            diffusion.feed_data(test_data)

            print('Generation from %s to %s' % (dataXinfo, dataYinfo))
            diffusion.test_generation(continuous=True)
            print('Registration from %s to %s' % (dataXinfo, dataYinfo))
            diffusion.test_registration(continuous=True)
            time2 = time.time()

            # Convert from [-1, 1] to [0, 255] for saving
            data_origin = (data_origin+1)/2. * 255
            data_fixed = (data_fixed + 1) / 2. * 255
            savePath = os.path.join(result_path, '%s_TO_%s_mov.png' % (dataXinfo, dataYinfo))
            save_image(data_origin, savePath)
            savePath = os.path.join(result_path, '%s_TO_%s_fix.png' % (dataXinfo, dataYinfo))
            save_image(data_fixed, savePath)

            visuals = diffusion.get_current_generation()
            sample_data = visuals['MF'].squeeze().numpy()
            for isamp in range(0, sample_data.shape[0], 6):
                savePath = os.path.join(result_path, '%s_TO_%s_sample_%d.png' % (dataXinfo, dataYinfo, isamp))
                synthetic_data = sample_data[isamp]
                synthetic_data -= synthetic_data.min()
                synthetic_data /= synthetic_data.max()
                synthetic_data = synthetic_data * 255
                save_image(synthetic_data, savePath)
            savePath = os.path.join(result_path, '%s_TO_%s_sample_last.png' % (dataXinfo, dataYinfo))
            synthetic_data = sample_data[-1]
            synthetic_data -= synthetic_data.min()
            synthetic_data /= synthetic_data.max()
            synthetic_data = synthetic_data * 255
            save_image(synthetic_data, savePath)
            
            # --- Dice Score Calculation ---
            visuals = diffusion.get_current_registration()
            flow = visuals['flow'].cuda()
            
            moving_mask = test_data['MM'].cuda()
            fixed_mask = test_data['FM'].cuda()
            
            # --- Before Registration ---
            # Binarize the masks
            moving_mask_bin_initial = (moving_mask > 0.5).float()
            fixed_mask_bin = (fixed_mask > 0.5).float()

            # Calculate Dice score before warping
            dice_metric(y_pred=moving_mask_bin_initial, y=fixed_mask_bin)
            initial_dice_score = dice_metric.aggregate().item()
            initial_dice_scores.append(initial_dice_score)

            # --- SSIM Calculation ---
            print(data_origin.shape, data_fixed.shape)
            print(data_origin.dtype, data_fixed.dtype)
            data_origin_torch = torch.from_numpy(data_origin).unsqueeze(0).unsqueeze(0).float().cuda()
            data_fixed_torch = torch.from_numpy(data_fixed).unsqueeze(0).unsqueeze(0).float().cuda()
            ssim_loss = SSIM_loss(data_origin_torch,
                                 data_fixed_torch)
            initial_ssims.append(ssim_loss.item())
            # --- LNCC Calculation ---
            lncc_loss = LNCC_loss(data_origin_torch,
                                 data_fixed_torch)
            initial_lnccs.append(lncc_loss.item())
            # --- MI Calculation ---
            mi_loss = MI_loss(data_origin_torch,
                             data_fixed_torch)
            initial_mis.append(mi_loss.item())


            # --- After Registration ---
            # Warp the moving mask
            warped_mask = stn(moving_mask, flow)
            
            # Binarize the warped mask
            warped_mask_bin = (warped_mask > 0.5).float()

            # Calculate Dice score after warping
            dice_metric(y_pred=warped_mask_bin, y=fixed_mask_bin)
            registered_dice_score = dice_metric.aggregate().item()
            registered_dice_scores.append(registered_dice_score)
            # --- SSIM Calculation ---
            warped_image = stn(test_data['M'].cuda(), flow)
            ssim_loss_reg = SSIM_loss(warped_image, test_data['F'].cuda())
            registered_ssims.append(ssim_loss_reg.item())
            # --- LNCC Calculation ---
            lncc_loss_reg = LNCC_loss(warped_image, test_data['F'].cuda())
            registered_lnccs.append(lncc_loss_reg.item())
            # --- MI Calculation ---
            mi_loss_reg = MI_loss(warped_image, test_data['F'].cuda())
            registered_mis.append(mi_loss_reg.item())
            
            print(f"Dice score for {dataXinfo} to {dataYinfo}: Initial = {initial_dice_score:.4f}, Registered = {registered_dice_score:.4f}")
            print(f"SSIM for {dataXinfo} to {dataYinfo}: Initial = {ssim_loss.item():.4f}, Registered = {ssim_loss_reg.item():.4f}")
            print(f"LNCC for {dataXinfo} to {dataYinfo}: Initial = {lncc_loss.item():.4f}, Registered = {lncc_loss_reg.item():.4f}")
            print(f"MI for {dataXinfo} to {dataYinfo}: Initial = {mi_loss.item():.4f}, Registered = {mi_loss_reg.item():.4f}")

        # --- Final Dice Score Comparison ---
        mean_initial_dice = np.mean(initial_dice_scores)
        std_initial_dice = np.std(initial_dice_scores)
        mean_registered_dice = np.mean(registered_dice_scores)
        std_registered_dice = np.std(registered_dice_scores)

        print("\n--- Final Metric Results ---")
        mean_initial_ssim = np.mean(initial_ssims)
        std_initial_ssim = np.std(initial_ssims)
        mean_registered_ssim = np.mean(registered_ssims)
        std_registered_ssim = np.std(registered_ssims)
        print(f"Initial SSIM (Before Registration): {mean_initial_ssim:.4f} (+/- {std_initial_ssim:.4f})")
        print(f"Registered SSIM (After Registration): {mean_registered_ssim:.4f} (+/- {std_registered_ssim:.4f})")
        mean_initial_lncc = np.mean(initial_lnccs)
        std_initial_lncc = np.std(initial_lnccs)
        mean_registered_lncc = np.mean(registered_lnccs)
        std_registered_lncc = np.std(registered_lnccs)
        print(f"Initial LNCC (Before Registration): {mean_initial_lncc:.4f} (+/- {std_initial_lncc:.4f})")
        print(f"Registered LNCC (After Registration): {mean_registered_lncc:.4f} (+/- {std_registered_lncc:.4f})")
        mean_initial_mi = np.mean(initial_mis)
        std_initial_mi = np.std(initial_mis)
        mean_registered_mi = np.mean(registered_mis)
        std_registered_mi = np.std(registered_mis)
        print(f"Initial MI (Before Registration): {mean_initial_mi:.4f} (+/- {std_initial_mi:.4f})")
        print(f"Registered MI (After Registration): {mean_registered_mi:.4f} (+/- {std_registered_mi:.4f})")
        
        print("\n--- Final Dice Score Results ---")
        print(f"Initial Dice Score (Before Registration): {mean_initial_dice:.4f} (+/- {std_initial_dice:.4f})")
        print(f"Registered Dice Score (After Registration): {mean_registered_dice:.4f} (+/- {std_registered_dice:.4f})")