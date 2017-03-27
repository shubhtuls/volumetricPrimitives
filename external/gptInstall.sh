## GMP
# wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
# tar -xvf gmp-6.1.2.tar.xz
# cd gmp-6.1.2/
# ./configure --prefix=$HOME/local/
# make; make install;
# cd ..

## MPFR
# wget http://www.mpfr.org/mpfr-current/mpfr-3.1.5.tar.gz
# tar -zxvf mpfr-3.1.5.tar.gz
## follow instructions at https://stackoverflow.com/questions/7561509/how-to-add-include-and-lib-paths-to-configure-make-cycle
# cd mpfr-3.1.5
# ./configure --prefix=$HOME/local/
# make; make install;
# cd ..

## CGAL
# wget https://github.com/CGAL/cgal/releases/download/releases%2FCGAL-4.9/CGAL-4.9.zip
# unzip CGAL-4.9.zip; mv CGAL-4.9 CGAL
# cd CGAL
# ccmake .
## manually set gmp and mpfr dir
## might need to install boost
# cmake .
# make
# cp ./lib/* $HOME/local/lib/
# cp -r ./include/* $HOME/local/include/
# cd ..

## GPTOOLBOX
#git clone --recursive https://github.com/libigl/libigl.git
#git clone git@github.com:alecjacobson/gptoolbox.git
# cp ./compile_gptoolbox_mex_modified.m ./gptoolbox/mex/
## RUN the compilation from within matlab
