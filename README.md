# Color Unmixing

This is an unofficial re-implementation of the soft image segmentation technique proposed in the following paper:
```
Yağiz Aksoy, Tunç Ozan Aydin, Aljoša Smolić, and Marc Pollefeys. 
Unmixing-Based Soft Color Segmentation for Image Manipulation. 
ACM Trans. Graph. 36, 2, Article 19 (March 2017), 19 pages. 
DOI: https://doi.org/10.1145/3002176
```
Please note that some steps of the original algorithm (e.g., acceleration of the color model determination (Sec. 5.1)) are not implemented. Also, we do not guarantee the correctness of the implementation.

## Dependencies

- [Eigen3](http://eigen.tuxfamily.org/)
- [Qt5](http://doc.qt.io/qt-5/)
- [NLopt](https://nlopt.readthedocs.io/)

## How to Compile and Run

We use [cmake](https://cmake.org/) for managing the sources codes. You can compile the software by typing, for example,
```
$ cd PATH_FOR_THE_SOURCE_DIRECTORY
$ mkdir build
$ cd build
$ cmake ../
$ make
```
You can also try to run the software using one of the cmake's features. After the compile, you can run the test by typing
```bash
$ ctest
```

## License

We distribute the source codes under the [MIT license](https://opensource.org/licenses/mit-license.php).

