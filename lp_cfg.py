

cfg_default = {
    "channel":        (32, 32, 32, 32, 48, 48, 48, 64, 64, 64, 64, 64, 64),
    "kernel_size":    (5,  3,  3,  1,  3,  3,  1,  3,  3,  1,  3,  3,  3),
    "stride":         (2,  1,  1,  1,  2,  1,  1,  1,  1,  1,  1,  1,  1),
    "global_padding": (2,  1,  1,  0,  1,  1,  0,  1,  0,  0,  0,  0,  0),
    "local_padding":  (2,  1,  1,  0,  1,  1,  0,  1,  0,  0,  0,  0,  0),
}

cfg = {
    "default": cfg_default
}
