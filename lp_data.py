import os, sys, glob, random
import torch, cv2
import numpy as np

from torch.utils.data import *
from imgaug import augmenters
#import torchvision.transforms as T



class LP_ReID_Dataset():
    def __init__(self, args, dataset, common_rare_threshold=5, verbose=False):
        self.root = os.path.join(args.path, dataset)
        self.args = args
        self.verbose = verbose
        self.common_rare_threshold = common_rare_threshold
        self.create_augmentation_seq()

    def __len__(self):
        return self.args.batch_per_epoch * torch.cuda.device_count()

    def prepare(self):
        all_person_id, unique_count = self.get_all_person_id()
        self.create_id_label_mapping(all_person_id)
        self.rare_person = {}
        rare_list = []
        for i in range(1, self.common_rare_threshold):
            rare_list += unique_count[i]
        for rare_person in rare_list:
            self.rare_person.update({rare_person: all_person_id[rare_person]})
            all_person_id.pop(rare_person)
        self.common_person = all_person_id
        self.common_person_list = self.common_person.keys()
        self.rare_person_list = self.rare_person.keys()

    def get_all_person_id(self):
        """
        2895 persons appeared 1 times
        535 persons appeared 2 times
        182 persons appeared 3 times
        21 persons appeared 4 times
        3 persons appeared 6 times
        4 persons appeared 7 times
        2 persons appeared 8 times
        7 persons appeared 9 times
        4 persons appeared 10 times
        9 persons appeared 11 times
        16 persons appeared 12 times
        16 persons appeared 13 times
        12 persons appeared 14 times
        13 persons appeared 15 times
        6 persons appeared 16 times
        5 persons appeared 17 times
        4 persons appeared 18 times
        1 persons appeared 19 times
        1 persons appeared 20 times
        2 persons appeared 21 times
        1 persons appeared 22 times
        2 persons appeared 24 times
        1 persons appeared 25 times
        1 persons appeared 28 times
        1 persons appeared 31 times
        :param root_path:
        :param unique_count:
        :return:
        """
        all_person_id = {}
        all_id = 0
        for i, cam in enumerate(sorted(glob.glob(self.root + "/*"))):
            if not os.path.isdir(cam):
                continue
            cam_id = cam[cam.rfind("/") + 1:]
            person_id = os.listdir(cam)
            for p_id in person_id:
                person_path = os.path.join(self.root, cam_id, p_id)
                if os.path.isdir(person_path):
                    if p_id.rfind("_") > 0:
                        id = p_id[:p_id.rfind("_")]
                    else:
                        id = p_id
                    if id in all_person_id:
                        all_person_id[id].append(person_path)
                    else:
                        all_person_id.update({id: [person_path]})
            # print("Cam_%d has %d person ids."%(i+1, len(person_id)))
            all_id += len(person_id)
            # all_person_id += person_id
        unique_count = {}
        for p_id in all_person_id.keys():
            if p_id.rfind("_") > 0:
                id = p_id[:p_id.rfind("_")]
            else:
                id = p_id
            if len(all_person_id[id]) in unique_count:
                unique_count[len(all_person_id[id])].append(id)
            else:
                unique_count.update({len(all_person_id[id]): [id]})
        if self.verbose:
            for k in sorted(unique_count.keys()):
                print("%d persons appeared %d times" % (len(unique_count[k]), k))
        return all_person_id, unique_count

    def create_id_label_mapping(self, all_person_id):
        self.mapping = {}
        keys = [key.zfill(5) for key in all_person_id.keys()]
        true_keys = [x for _, x in sorted(zip(keys, all_person_id.keys()))]
        for i, id in enumerate(true_keys):
            self.mapping.update({id: i + 1})

    def create_augmentation_seq(self):
        aug_list = []
        # Slight Affine Transformation
        #aug_list.append(augmenters.Affine(scale=(0.9, 1),
                                          #translate_px = (-2, 2),
                                          #rotate=(-2, 2),
                                          #shear=(-2, 2),
                                          #cval=self.args.bg_color))
        # Crop out a 32x32 patches
        aug_list.append(augmenters.CropToFixedSize(width=32, height=32,
                                                  position=(0.5, 0.5)))

        # Random quality compression
        aug_list.append(augmenters.JpegCompression(compression=[0, 85]))

        # Perform Flip
        # aug_list.append(augmenters.Fliplr(0.33, name="horizontal_flip"))
        # aug_list.append(augmenters.Flipud(0.33, name="vertical_flip"))
        self.aug_seq = augmenters.Sequential(aug_list, random_order=False)

    def load_img(self, load_img_list):
        aug_seq = self.aug_seq.to_deterministic()
        batch = []
        bad_img_sum = 0
        for img_folder in load_img_list:
            imgs = []
            bad_img_num = 0
            for i in range(self.args.open_pose_joint_num):
                path = os.path.join(img_folder, "%s.jpg"%str(i).zfill(2))
                if os.path.exists(path):
                    imgs.append(cv2.imread(path))
                else:
                    bad_img_num += 1
                    imgs.append(np.ones(
                        (self.args.ori_img_size, self.args.ori_img_size, self.args.img_channel)
                    ).astype("uint8") * self.args.non_part_intensity)
            if bad_img_num == self.args.open_pose_joint_num:
                raise FileNotFoundError()
            bad_img_sum += bad_img_num
            if self.args.augment_img:
                imgs = aug_seq.augment_images(imgs)
            imgs = np.concatenate(imgs, axis=2)
            batch.append(imgs)
        batch = torch.tensor(batch, dtype=torch.float)
        batch = batch.permute(0, 3, 1, 2)
        #self.visualize_batch(batch)
        if self.verbose:
            print("This batch has %d local patches which is empty"%(bad_img_sum))
        return batch

    def visualize_batch(self, batch):
        batch = batch.view(54, 17, 3, 32, 32).permute(0, 1, 3, 4, 2)
        persons = []
        for i in range(batch.shape[0]):
            horizon = []
            for img in batch[i]:
                horizon.append(img)
            horizon = torch.cat(horizon, dim=0)
            persons.append(horizon)
        img = torch.cat(persons, dim=1)
        cv2.imwrite(os.path.expanduser("~/Pictures/tmps.jpg"), img.numpy())

    def __getitem__(self, index):
        # Select several persons from both common and rare person dict
        common_person_list = random.sample(list(self.common_person_list), self.args.batch_common_person)
        rare_person_list = random.sample(list(self.rare_person_list), self.args.batch_common_person)
        # Select N different camera view from common_person_list
        load_img_list, label = [], []
        for common_person in common_person_list:
            label += [self.mapping[common_person]] * \
                     (self.args.n_cam_views * self.args.n_cam_views_same)
            person_view_list = random.sample(
                    self.common_person[common_person],
                    self.args.n_cam_views) # N different camera view
            for person_frame_folder in person_view_list:
                all_views = glob.glob(person_frame_folder + "/*")
                if len(all_views) == 0:
                    raise FileExistsError(all_views)
                elif len(all_views) == 1:
                    #print("unsuficient data at %s"%all_views[0])
                    load_img_list += all_views * 2
                else:
                    person_view = random.sample(
                        all_views, self.args.n_cam_views_same
                    )
                    load_img_list += person_view
        for rare_person in rare_person_list:
            person_view_list = random.choice(self.rare_person[rare_person])
            all_views = glob.glob(person_view_list + "/*")
            if len(all_views) == 0:
                raise FileExistsError(all_views)
            label.append(self.mapping[rare_person])
            person_view = random.choice(all_views)
            load_img_list.append(person_view)
        imgs = self.load_img(load_img_list)
        label = torch.tensor(label).int()
        #print(label)
        return imgs, label

def retrieval_collector(batch):
    imgs, labels = [], []
    for i, (img, label) in enumerate(batch):
        imgs.append(img)
        labels.append(label)
    imgs = torch.cat(imgs, dim=0)
    labels = torch.cat(labels, dim=0)
    return imgs, labels

def fetch_dataset(args, shuffle=True, verbose=False, for_test=False):
    dataset = LP_ReID_Dataset(args, dataset=args.train_sources, verbose=verbose)
    dataset.prepare()
    args.loading_threads = round(args.loading_threads * torch.cuda.device_count())
    batch_size = args.batch_size_per_gpu * torch.cuda.device_count()
    kwargs = {'num_workers': args.loading_threads, 'pin_memory': True}
    train_set = DataLoader(dataset, batch_size=batch_size,
                           shuffle=shuffle, drop_last=True,
                           collate_fn=retrieval_collector, **kwargs)
    if for_test:
        return train_set, dataset.mapping
    else:
        return train_set


def eliminate_empty_folder(remove_num_less_then=1, execute_deletion=False):
    root = os.path.expanduser("~/Pictures/dataset/reid/OP_local_patch")
    for i, camera in enumerate(sorted(glob.glob(root))):
        for j, person in enumerate(sorted(glob.glob(camera + "/*"))):
            for k, frame in enumerate(sorted(glob.glob(person + "/*"))):
                if len(os.listdir(frame)) < remove_num_less_then:
                    command = "rm -rf %s" % frame
                    if execute_deletion:
                        os.system(command)
                    else:
                        print(command)
    print("Empty folder check finished.")

if __name__ == '__main__':
    import time
    from lp_args import parse_arguments
    import omni_torch.utils as util
    import lp_preset as preset
    eliminate_empty_folder(remove_num_less_then=1, execute_deletion=False)
    args = parse_arguments()
    opt = util.get_args(preset.PRESET)
    args = util.cover_edict_with_argparse(opt, args)

    # Disable the imgaugmentation could save 60% of loading time.
    eliminate_empty_folder()
    dataset = fetch_dataset(args, verbose=False)
    start = time.time()
    for i, (img, label) in enumerate(dataset):
        print("Batch: %d, data shape: %s"%(i, str(img.shape)))
    print("One epoch finished, cost %.3f seconds"%(time.time() - start))