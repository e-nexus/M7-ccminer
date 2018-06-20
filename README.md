ccminer
=======

Based on Christian Buchner's &amp; Christian H.'s CUDA project, no more active on github recently.


A part of the recent algos were originally written by [djm34](https://github.com/djm34) and [pallas] (https://github.com/pallas1)

This variant was tested and built on VStudio 2015 on Windows 7.

Note that the x86 releases are generally faster than x64 ones on Windows.

The recommended CUDA Toolkit version is [9.1]

About source code dependencies
------------------------------

This project requires some libraries to be built :

- OpenSSL (prebuilt for win)

- Curl (prebuilt for win)

- pthreads (prebuilt for win)

The tree now contains recent prebuilt openssl and curl .lib for both x86 and x64 platforms (windows).

To rebuild them, you need to clone this repository and its submodules :
    git clone https://github.com/peters/curl-for-windows.git compat/curl-for-windows

On Linux, you can use the helper ./build.sh (edit it if required)

There is also an old [Tutorial for windows](http://cudamining.co.uk/url/tutorials/id/3) on [CudaMining](http://cudamining.co.uk) website.

