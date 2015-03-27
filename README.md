cuda-smi
====

On Windows/Linux, there is a program called `nvidia-smi` for checking the memory and general statistics of GPUs.

There is no such thing on MacOS.

This program tries to bring that ability to MacOS. However at the current version, it can only check the free memory and some basic info.

Usage
====

Run
```sh
./compile.sh
```

The output file is `cuda-smi`. Feel free to copy it to somewhere in your `PATH` directories.
