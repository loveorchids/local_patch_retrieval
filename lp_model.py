import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import omni_torch.networks.blocks as omth_blocks
from itertools import combinations, product

class Encoder(nn.Module):
    def __init__(self, args, batch_norm=None, channel=(32, 32, 32, 16, 16, 8, 8),
                 kernel_size=(5, 3, 3, 3, 3, 3, 3), stride=(2, 1, 1, 1, 1, 1, 1),
                 global_padding=(2, 0, 0, 0, 0, 0, 0), local_padding=(2, 0, 0, 0, 0, 0, 0)):
        super().__init__()
        self.args = args
        self.group = args.open_pose_joint_num
        self.channels = channel
        self.kernel_sizes = kernel_size
        self.strides = stride
        self.batch_norm = batch_norm
        self.global_padding = global_padding
        self.local_padding = local_padding
        self.feature_comb = args.feature_comb

        #self.global_feature_comparison = args.global_feature_comparison
        #self.loss_weight = args.loss_weight
        assert type(args.loss_weight) is list and len(args.loss_weight) in [1, 9]
        if len(args.loss_weight) == 1:
            if args.loss_weight[0].lower() == "auto":
                loss_weight = None
                self.loss_weight = None
            elif args.loss_weight[0].lower() == "l1":
                #                          | Def_S  | Cha_S  | Def_D |
                loss_weight = torch.tensor([1, 0, 0, 1, 0, 0, 1, 0, 0]).float()
            elif args.loss_weight[0].lower() == "l2":
                loss_weight = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 0]).float()
            elif args.loss_weight[0].lower() == "kl":
                loss_weight = torch.tensor([0, 0, 1, 0, 0, 1, 0, 0, 1]).float()
            elif args.loss_weight[0].lower() in ["average", "default", "avg"]:
                loss_weight = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1]).float()
            else:
                raise NotImplementedError()
        else:
            loss_weight = torch.tensor(args.loss_weight).float()
        if loss_weight is not None:
            self.loss_weight = [loss_weight.cuda(i) for i in range(torch.cuda.device_count())]
        self.global_embedding = args.global_embedding
        if self.global_embedding:
            self.dropout_050 = nn.Dropout(0.50)
            self.fc_layer1 = nn.Linear(2176, 512)
            self.fc_layer2 = nn.Linear(512, 128)
        self.normalize_embedding = args.normalize_embedding
        self.loss = [lp_L1_Loss(), lp_L2_Loss(), lp_KL_divergence()]
        self.comb_feature = CombFeature(channel[-1], self.group,
                                        comb_method=self.feature_comb)
        self.attn_module = Attn_Generator(channel[-1], self.group)
        #self.maxout = nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                                   #dilation=1, ceil_mode=True)
        self.create_global_conv()
        self.create_local_conv()
        #self.generate_triplet_comparison_idx()


    def create_global_conv(self):
        global_conv = []
        for i, channel in enumerate(self.channels):
            if i == 0:
                input_channel = self.group * self.args.img_channel
            else:
                input_channel = self.channels[i-1]
            global_conv.append(
                omth_blocks.Conv_Block(
                    input_channel, filters=channel, kernel_sizes=self.kernel_sizes[i],
                    padding=self.global_padding[i],
                    stride=self.strides[i], batch_norm=self.batch_norm
                )
            )
        self.global_conv = nn.Sequential(*global_conv)

    def create_local_conv(self):
        local_conv = []
        for i, channel in enumerate(self.channels):
            if i == 0:
                input_channel = self.group * self.args.img_channel
            else:
                input_channel = self.group * self.channels[i-1]
            local_conv.append(
                omth_blocks.Conv_Block(
                    input_channel, filters=self.group * channel,
                    kernel_sizes=self.kernel_sizes[i], padding=self.local_padding[i],
                    stride=self.strides[i], groups=self.group, batch_norm=self.batch_norm
                )
            )
        self.local_conv = nn.Sequential(*local_conv)

    def generate_triplet_comparison_idx(self):
        args = self.args
        batch = self.args.batch_size_per_gpu
        common_person = args.batch_common_person * args.n_cam_views
        all_common_person = common_person * args.n_cam_views_same
        person_in_batch = all_common_person + args.batch_rare_person
        base = [list([comb
                      for comb in combinations(range(i, i+args.n_cam_views_same), 2)][0])
                for i in range(0, all_common_person, args.n_cam_views_same)]
        _definite_same = torch.tensor(base).transpose(1, 0)
        person_pairs = list(zip(*[[random.choice(pair)
                                   for pair in base[i:len(base):args.n_cam_views]]
                                  for i in range(args.n_cam_views)]))
        _challenging_same = torch.tensor([[list(comb)
                                  for comb in combinations(person_pair, 2)]
                                 for person_pair in person_pairs]).view(-1, 2).transpose(1, 0)
        interval = args.n_cam_views_same * args.n_cam_views
        all_range = list(range(0, all_common_person, interval))
        common_choice = [random.choice(range(i, i+interval)) for i in all_range]
        _definite_diff = torch.tensor(list(
            product(common_choice, range(all_common_person, person_in_batch)))
        ).transpose(1, 0)
        definite_same = []
        challenging_same = []
        definite_diff = []
        for i in range(0, batch):
            definite_same.append(_definite_same + person_in_batch * i)
            challenging_same.append(_challenging_same + person_in_batch * i)
            definite_diff.append(_definite_diff + person_in_batch * i)
        self.definite_same = torch.cat(definite_same, dim=1)
        self.challenging_same = torch.cat(challenging_same, dim=1)
        self.definite_diff = torch.cat(definite_diff, dim=1)
        #self.definite_same = [definite_same.cuda(i) for i in range(torch.cuda.device_count())]
        #self.challenging_same = [challenging_same.cuda(i) for i in range(torch.cuda.device_count())]
        #self.definite_diff = [definite_diff.cuda(i) for i in range(torch.cuda.device_count())]

    def compare_distance(self, comb_feature, attn_map):
        loss = []
        l_r_attn_map = 1
        # Calculate the distance of samples shall definitely be same
        if not self.global_embedding:
            l_r_attn_map = attn_map[self.definite_same[0]] * \
                           attn_map[self.definite_same[1]]
        l_tensor = comb_feature[self.definite_same[0]] * l_r_attn_map
        try:
            r_tensor = comb_feature[self.definite_same[1]] * l_r_attn_map
        except:
            xxx=0
        loss += [5 * loss(l_tensor, r_tensor) for loss in self.loss]

        # Calculate the distance of samples shall be same but maybe challenging
        if not self.global_embedding:
            l_r_attn_map = attn_map[self.challenging_same[0]] * \
                           attn_map[self.challenging_same[1]]
        l_tensor = comb_feature[self.challenging_same[0]] * l_r_attn_map
        r_tensor = comb_feature[self.challenging_same[1]] * l_r_attn_map
        loss += [loss(l_tensor, r_tensor) for loss in self.loss]

        # Calculate the distance of samples shall definitely be different
        if not self.global_embedding:
            l_r_attn_map = attn_map[self.definite_diff[0]] * \
                           attn_map[self.definite_diff[1]]
        l_tensor = comb_feature[self.definite_diff[0]] * l_r_attn_map
        r_tensor = comb_feature[self.definite_diff[1]] * l_r_attn_map
        loss += [-2 * loss(l_tensor, r_tensor) for loss in self.loss]

        loss = torch.stack(loss) * self.loss_weight[comb_feature.device.index] * 100
        return loss

    def forward(self, x, verbose=False, test=False):
        b, c, h, w = x.shape
        self.generate_triplet_comparison_idx()
        # normalize, so the range of x is [-128, 127]
        x = x - self.args.non_part_intensity
        # Feature Extraction
        global_feature = self.global_conv(x)
        if verbose:
            print("Global feature size: ", global_feature.shape)
        local_feature = self.local_conv(x)
        if verbose:
            print("Local feature size: ", local_feature.shape)

        # Combine the local feature and global feature
        comb_feature = self.comb_feature(local_feature, global_feature)
        comb_feature = comb_feature.view(b, self.group, -1)
        if verbose:
            print("Feature combined and resize to: ", comb_feature.shape)

        # Generate attention map
        attn_map = self.attn_module(x, local_feature)
        attn_map = attn_map.unsqueeze(-1).repeat(1, 1, comb_feature.shape[-1])
        if verbose:
            print("attention map generated: ", attn_map.shape)

        comb_feature = comb_feature * attn_map
        if self.args.normalize_embedding:
            comb_feature = F.softmax(comb_feature, dim=-1)
        if self.global_embedding:
            comb_feature = comb_feature.view(b, -1)
            comb_feature = self.fc_layer2(self.dropout_050(self.fc_layer1(comb_feature)))
            comb_feature = F.softmax(comb_feature, dim=-1)
        if test:
            return comb_feature, attn_map
        else:
            return self.compare_distance(comb_feature, attn_map)


class CombFeature(nn.Module):
    def __init__(self, input, group, comb_method="add"):
        super().__init__()
        self.group = group
        self.channel = input
        self.comb_method = comb_method.lower()
        assert self.comb_method in ["add", "conv"]
        if self.comb_method == "conv":
            self.comb_conv = omth_blocks.Conv_Block(
                input * group * 2,
                filters=[input * group * 2, input * group],
                kernel_sizes=[3, 1],
                stride=[1, 1],
                padding=[1, 0],
                groups=[group, group],
            )

    def forward(self, loc_feature, glo_feature):
        if self.comb_method == "conv":
            comb = []
            for i in range(self.group):
                comb.append(
                    torch.cat([
                        loc_feature[:, self.channel * i : self.channel * (i + 1), :, :],
                        glo_feature
                    ], dim=1)
                )
            comb = torch.cat(comb, dim=1)
            return self.comb_conv(comb)
        elif self.comb_method == "add":
            glo_feature = glo_feature.repeat(1, self.group, 1, 1)
            return loc_feature + glo_feature
        else:
            raise NotImplementedError()

class Attn_Generator(nn.Module):
    def __init__(self, input, group):
        super().__init__()
        self.group = group
        self.channel = input
        self.attn_estimation = omth_blocks.Conv_Block(
            input * group,
            filters=[group * input, group],
            kernel_sizes=[3, 1],
            stride=[1, 1],
            padding=[1, 0],
            groups=[group, group],
        )

    def visualize_imgs(self, x, idx):
        import cv2
        imgs = x[idx].view(17, 3, 32, 32).permute(0, 2, 3, 1) + 128
        imgs = imgs.cpu().numpy()
        for i, img in enumerate(imgs):
            cv2.imwrite("/home/wang/Pictures/tmp_%s.jpg"%i, img)

    def forward(self, x, loc_feature):
        b, c, h, w = x.shape
        x = x.view(b, self.group, -1)
        # this below get the image which is not an empty img
        zero_idx = torch.sum((x < 1) & (x > -1), dim=-1) < (x.shape[-1] * 0.9)
        attn = self.attn_estimation(loc_feature)
        attn = torch.sum(torch.sum(attn, dim=-1), dim=-1)
        attn = F.softmax(attn, dim=-1)
        attn = attn * zero_idx.float()
        return attn

class lp_L1_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="sum")

    def forward(self, x, y):
        b = x.shape[0]
        loss = self.loss(x, y)
        return loss / b

class lp_L2_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="sum")

    def forward(self, x, y):
        b = x.shape[0]
        loss = self.loss(x, y)
        return loss / b

class lp_KL_divergence(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction="batchmean")
        self.normalize = nn

    def forward(self, x, y):
        embed_dim = x.shape[-1]
        x = x.view(-1, embed_dim)
        y = y.view(-1, embed_dim)
        loss = self.loss(x, y)
        return loss


if __name__ == '__main__':
    from lp_args import parse_arguments
    import omni_torch.utils as util
    import lp_preset as preset
    args = parse_arguments()
    opt = util.get_args(preset.PRESET)
    args = util.cover_edict_with_argparse(opt, args)

    args.feature_combinition = "conv"

    net = Encoder(args)
    net = torch.nn.DataParallel(net).cuda()
    x = torch.rand(56 * 8, 51, 40, 40) * 256
    x = x.cuda()
    #net = Piecewise_Compare()
    #x = torch.rand(56, 272, 6, 6)
    y = net(x, verbose=True)
    print(y.shape)