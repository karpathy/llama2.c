@rem assumes you have setup the environment via a command like
@rem "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
nvcc -DUSE_CUDA --compiler-options "/nologo /fp:fast /Ox /openmp /I." run.cu win.c -o runcuda.exe
