# ==================================================================
# Copyright (c) 2023, uyolo.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the
# distribution.
# 3. All advertising materials mentioning features or use of this software
# must display the following acknowledgement:
# This product includes software developed by the uyolo Group. and
# its contributors.
# 4. Neither the name of the Group nor the names of its contributors may
# be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY UYOLO, GROUP AND CONTRIBUTORS
# ===================================================================

import os
import platform
import warnings

import torch.multiprocessing as mp


def set_multi_processing(
    mp_start_method: str = "fork", opencv_num_threads: int = 0, distributed: bool = True
) -> None:
    """Set multi-processing related environment.
    This function is refered from https://github.com/RangiLyu/nanodet/blob/main/nanodet/util/env_utils.py
    Args:
        mp_start_method (str): Set the method which should be used to start
            child processes. Defaults to 'fork'.
        opencv_num_threads (int): Number of threads for opencv.
            Defaults to 0.
        distributed (bool): True if distributed environment.
            Defaults to False.
    """  # noqa
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != "Windows":
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(
                f"Multi-processing start method `{mp_start_method}` is "
                f"different from the previous setting `{current_method}`."
                f"It will be force set to `{mp_start_method}`. You can "
                "change this behavior by changing `mp_start_method` in "
                "your config."
            )
        mp.set_start_method(mp_start_method, force=True)

    try:
        import cv2

        # disable opencv multithreading to avoid system being overloaded
        cv2.setNumThreads(opencv_num_threads)
    except ImportError:
        pass

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if "OMP_NUM_THREADS" not in os.environ and distributed:
        omp_num_threads = 1
        warnings.warn(
            "Setting OMP_NUM_THREADS environment variable for each process"
            f" to be {omp_num_threads} in default, to avoid your system "
            "being overloaded, please further tune the variable for "
            "optimal performance in your application as needed."
        )
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    # setup MKL threads
    if "MKL_NUM_THREADS" not in os.environ and distributed:
        mkl_num_threads = 1
        warnings.warn(
            "Setting MKL_NUM_THREADS environment variable for each process"
            f" to be {mkl_num_threads} in default, to avoid your system "
            "being overloaded, please further tune the variable for "
            "optimal performance in your application as needed."
        )
        os.environ["MKL_NUM_THREADS"] = str(mkl_num_threads)