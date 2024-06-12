import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset import Dataset, mel_spectrogram, amp_pha_specturm, get_dataset_filelist
from models_pghi import (
    Generator,
    MultiPeriodDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
    amplitude_loss,
    phase_loss,
    STFT_consistency_loss,
    MultiResolutionDiscriminator,
)
from utils import (
    AttrDict,
    build_env,
    plot_spectrogram,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

torch.backends.cudnn.benchmark = True


def train(h):
    torch.cuda.manual_seed(h.seed)
    device = torch.device("cuda:{:d}".format(0))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    print(generator)
    os.makedirs(h.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", h.checkpoint_path)

    if os.path.isdir(h.checkpoint_path):
        cp_g = scan_checkpoint(h.checkpoint_path, "g_")
        cp_do = scan_checkpoint(h.checkpoint_path, "do_")

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        mrd.load_state_dict(state_dict_do["mrd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch
    )

    training_filelist, validation_filelist = get_dataset_filelist(
        h.input_training_wav_list, h.input_validation_wav_list
    )

    trainset = Dataset(
        training_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        h.meloss,
        n_cache_reuse=0,
        shuffle=True,
        device=device,
        inv_mel=True,
        use_pghi=True,
    )

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=True,
        sampler=None,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    validset = Dataset(
        validation_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        h.meloss,
        False,
        False,
        n_cache_reuse=0,
        device=device,
        inv_mel=True,
        use_pghi=True,
    )
    validation_loader = DataLoader(
        validset,
        num_workers=1,
        shuffle=False,
        sampler=None,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
    )

    sw = SummaryWriter(os.path.join(h.checkpoint_path, "logs"))

    generator.train()
    mpd.train()
    mrd.train()

    for epoch in range(max(0, last_epoch), h.training_epochs):
        start = time.time()
        print("Epoch: {}".format(epoch + 1))

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            x, logamp, pha, rea, imag, y, meloss, inv_mel, pghid = map(
                lambda x: x.to(device, non_blocking=True), batch
            )
            y = y.unsqueeze(1)
            logamp_g, pha_g, rea_g, imag_g, y_g = generator(
                x, inv_mel=inv_mel, pghi=pghid
            )
            y_g_mel = mel_spectrogram(
                y_g.squeeze(1),
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.meloss,
            )

            optim_d.zero_grad()

            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g
            )

            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g
            )

            L_D = loss_disc_s * 0.1 + loss_disc_f

            L_D.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # Losses defined on log amplitude spectra
            L_A = amplitude_loss(logamp, logamp_g)

            L_IP, L_GD, L_PTD = phase_loss(pha, pha_g, h.n_fft, pha.size()[-1])
            # Losses defined on phase spectra
            L_P = L_IP + L_GD + L_PTD

            _, _, rea_g_final, imag_g_final = amp_pha_specturm(
                y_g.squeeze(1), h.n_fft, h.hop_size, h.win_size
            )
            L_C = STFT_consistency_loss(rea_g, rea_g_final, imag_g, imag_g_final)
            L_R = F.l1_loss(rea, rea_g)
            L_I = F.l1_loss(imag, imag_g)
            # Losses defined on reconstructed STFT spectra
            L_S = L_C + 2.25 * (L_R + L_I)

            y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(y, y_g)
            y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = mrd(y, y_g)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_g)
            L_GAN_G = loss_gen_s * 0.1 + loss_gen_f
            L_FM = loss_fm_s * 0.1 + loss_fm_f
            L_Mel = F.l1_loss(meloss, y_g_mel)
            # Losses defined on final waveforms
            L_W = L_GAN_G + L_FM + 45 * L_Mel

            L_G = 45 * L_A + 100 * L_P + 20 * L_S + L_W

            L_G.backward()
            optim_g.step()

            # STDOUT logging
            if steps % h.stdout_interval == 0:
                with torch.no_grad():
                    A_error = amplitude_loss(logamp, logamp_g).item()
                    IP_error, GD_error, PTD_error = phase_loss(
                        pha, pha_g, h.n_fft, pha.size()[-1]
                    )
                    IP_error = IP_error.item()
                    GD_error = GD_error.item()
                    PTD_error = PTD_error.item()
                    C_error = STFT_consistency_loss(
                        rea_g, rea_g_final, imag_g, imag_g_final
                    ).item()
                    R_error = F.l1_loss(rea, rea_g).item()
                    I_error = F.l1_loss(imag, imag_g).item()
                    Mel_error = F.l1_loss(x, y_g_mel).item()

                print(
                    "Steps : {:d}, Gen Loss Total : {:4.3f}, Amplitude Loss : {:4.3f}, Instantaneous Phase Loss : {:4.3f}, Group Delay Loss : {:4.3f}, Phase Time Difference Loss : {:4.3f}, STFT Consistency Loss : {:4.3f}, Real Part Loss : {:4.3f}, Imaginary Part Loss : {:4.3f}, Mel Spectrogram Loss : {:4.3f}, s/b : {:4.3f}".format(
                        steps,
                        L_G,
                        A_error,
                        IP_error,
                        GD_error,
                        PTD_error,
                        C_error,
                        R_error,
                        I_error,
                        Mel_error,
                        time.time() - start_b,
                    )
                )

            # checkpointing
            if steps % h.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, {"generator": generator.state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(
                    checkpoint_path,
                    {
                        "mpd": mpd.state_dict(),
                        "mrd": mrd.state_dict(),
                        "optim_g": optim_g.state_dict(),
                        "optim_d": optim_d.state_dict(),
                        "steps": steps,
                        "epoch": epoch,
                    },
                )

            # Tensorboard summary logging
            if steps % h.summary_interval == 0:
                sw.add_scalar("Training/Generator_Total_Loss", L_G, steps)
                sw.add_scalar("Training/Mel_Spectrogram_Loss", Mel_error, steps)

            # Validation
            if steps % h.validation_interval == 0:  # and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_A_err_tot = 0
                val_IP_err_tot = 0
                val_GD_err_tot = 0
                val_PTD_err_tot = 0
                val_C_err_tot = 0
                val_R_err_tot = 0
                val_I_err_tot = 0
                val_Mel_err_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        x, logamp, pha, rea, imag, y, meloss, inv_mel, pghid = map(
                            lambda x: x.to(device, non_blocking=True), batch
                        )
                        logamp_g, pha_g, rea_g, imag_g, y_g = generator(
                            x, inv_mel=inv_mel, pghi=pghid
                        )
                        x = x.cpu()
                        y_g_mel = mel_spectrogram(
                            y_g.squeeze(1),
                            h.n_fft,
                            h.num_mels,
                            h.sampling_rate,
                            h.hop_size,
                            h.win_size,
                            h.fmin,
                            h.meloss,
                        )

                        _, _, rea_g_final, imag_g_final = amp_pha_specturm(
                            y_g.squeeze(1), h.n_fft, h.hop_size, h.win_size
                        )
                        val_A_err_tot += amplitude_loss(logamp, logamp_g).item()
                        val_IP_err, val_GD_err, val_PTD_err = phase_loss(
                            pha, pha_g, h.n_fft, pha.size()[-1]
                        )
                        val_IP_err_tot += val_IP_err.item()
                        val_GD_err_tot += val_GD_err.item()
                        val_PTD_err_tot += val_PTD_err.item()
                        val_C_err_tot += STFT_consistency_loss(
                            rea_g, rea_g_final, imag_g, imag_g_final
                        ).item()
                        val_R_err_tot += F.l1_loss(rea, rea_g).item()
                        val_I_err_tot += F.l1_loss(imag, imag_g).item()
                        val_Mel_err_tot += F.l1_loss(meloss, y_g_mel).item()

                        if j <= 4:
                            if steps == 0:
                                sw.add_audio(
                                    "gt/y_{}".format(j), y[0], steps, h.sampling_rate
                                )
                                sw.add_figure(
                                    "gt/y_spec_{}".format(j),
                                    plot_spectrogram(x[0]),
                                    steps,
                                )

                            sw.add_audio(
                                "generated/y_g_{}".format(j),
                                y_g[0],
                                steps,
                                h.sampling_rate,
                            )
                            y_g_spec = mel_spectrogram(
                                y_g.squeeze(1),
                                h.n_fft,
                                h.num_mels,
                                h.sampling_rate,
                                h.hop_size,
                                h.win_size,
                                h.fmin,
                                h.fmax,
                            )
                            sw.add_figure(
                                "generated/y_g_spec_{}".format(j),
                                plot_spectrogram(y_g_spec.squeeze(0).cpu().numpy()),
                                steps,
                            )

                    val_A_err = val_A_err_tot / (j + 1)
                    val_IP_err = val_IP_err_tot / (j + 1)
                    val_GD_err = val_GD_err_tot / (j + 1)
                    val_PTD_err = val_PTD_err_tot / (j + 1)
                    val_C_err = val_C_err_tot / (j + 1)
                    val_R_err = val_R_err_tot / (j + 1)
                    val_I_err = val_I_err_tot / (j + 1)
                    val_Mel_err = val_Mel_err_tot / (j + 1)
                    sw.add_scalar("Validation/Amplitude_Loss", val_A_err, steps)
                    sw.add_scalar(
                        "Validation/Instantaneous_Phase_Loss", val_IP_err, steps
                    )
                    sw.add_scalar("Validation/Group_Delay_Loss", val_GD_err, steps)
                    sw.add_scalar(
                        "Validation/Phase_Time_Difference_Loss", val_PTD_err, steps
                    )
                    sw.add_scalar("Validation/STFT_Consistency_Loss", val_C_err, steps)
                    sw.add_scalar("Validation/Real_Part_Loss", val_R_err, steps)
                    sw.add_scalar("Validation/Imaginary_Part_Loss", val_I_err, steps)
                    sw.add_scalar("Validation/Mel_Spectrogram_loss", val_Mel_err, steps)

                generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        print(
            "Time taken for epoch {} is {} sec\n".format(
                epoch + 1, int(time.time() - start)
            )
        )


def main():
    print("Initializing Training Process..")

    config_file = "config_pghi.json"

    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(config_file, "config_pghi.json", h.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    else:
        pass

    train(h)


if __name__ == "__main__":
    main()
