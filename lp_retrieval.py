import os, time, datetime
import torch
import numpy as np
import lp_data as data
import lp_model as model
from lp_args import parse_arguments
import omni_torch.utils as util
import lp_preset as preset
from omni_torch.networks.optimizer.adastand import Adastand
import omni_torch.visualize.basic as vb

PIC = os.path.expanduser("~/Pictures/")
TMPJPG = os.path.expanduser("~/Pictures/tmp.jpg")

args = parse_arguments()
opt = util.get_args(preset.PRESET)
args = util.cover_edict_with_argparse(opt, args)
dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")


def fit(args, net, dataset, optimizer, is_train):
    if is_train:
        net.train()
    else:
        net.eval()
    Loss = []
    start_time = time.time()
    for batch_idx, (img, label) in enumerate(dataset):
        img = img.cuda()
        optimizer.zero_grad()
        pred = net(img)
        loss = torch.sum(pred)
        loss.backward()
        optimizer.step()
        #losses = torch.sum(torch.sum(pred.data.view(2, 3, 3), dim=0), dim=0)
        Loss.append(torch.mean(pred.data.view(torch.cuda.device_count(), -1), dim=0))
    Loss = torch.mean(torch.stack(Loss, dim=0), dim=0)
    print(" --- Total loss: %.4f, at epoch %04d, cost %.2f seconds ---" %
          (float(torch.sum(Loss)), args.curr_epoch, time.time() - start_time))
    print(Loss)
    return Loss.cpu()


def val(args, net, dataset, optimizer):
    with torch.no_grad():
        fit(args, net, dataset, optimizer, False)


def main():
    dataset = data.fetch_dataset(args, verbose=False)

    net = model.Encoder(args)
    net = torch.nn.DataParallel(net).cuda()
    torch.backends.cudnn.benchmark = True

    # Using the latest optimizer, better than Adam and SGD
    optimizer = Adastand(net.parameters(), lr=args.learning_rate,
                         weight_decay=args.weight_decay,)

    TrainLoss = []
    for epoch in range(args.epoch_num):
        epoch_loss = fit(args, net, dataset, optimizer, is_train=True)
        TrainLoss.append(epoch_loss)
        if (epoch + 1) % 100 == 0:
            util.save_model(args, args.curr_epoch, net.state_dict(), prefix=args.model_prefix,
                            keep_latest=25)
        if epoch >= 5:
            # Train losses
            plot_loss = torch.stack(TrainLoss, dim=0).view(-1, 3, 3).permute(2, 1, 0).numpy()
            #vb.plot_curves(plot_loss, ["L1_Loss", "L2_Loss", "KL_Div_Loss"],
                           #args.loss_log, dt + "_loss", window=5, title=args.model_prefix)
            vb.plot_multi_loss_distribution(plot_loss, [["Definite_S", "Challenge_S", "Definite_D"] for _ in range(3)],
                                            save_path=args.loss_log, name=dt + "_loss",
                                            window=5, fig_size=(15, 15), grid=True,
                                            titles=["L1_Loss", "L2_Loss", "KL_Div_Loss"],)
                                            #bound=[{"low": 0, "high": 1} for _ in range(3)])
        args.curr_epoch += 1


if __name__ == "__main__":
    main()
