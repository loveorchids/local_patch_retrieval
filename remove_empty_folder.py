import os, glob

root = os.path.expanduser("~/Pictures/dataset/reid/OP_local_patch/Cam_*")

for cam in sorted(glob.glob(root)):
    for person in sorted(glob.glob(cam + "/*")):
        for view in sorted(glob.glob(person + "/Cam_*")):
            if len(os.listdir(view)) == 0:
                cmd = "rm -r %s"%view
                print(cmd)
                os.system(cmd)