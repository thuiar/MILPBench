## About

The original environment for Learn2Branch is a classic and well-established setup, which has been relied upon by subsequent works such as LearnLNS and GNN-MILP. However, existing configuration schemes often have issues. We have carefully compiled the relevant packages and resources to create this improved setup.

## Steps

1. ### Initialisation

   First, install conda and run the following commands:

   ```
   conda create -n Learn2Branch python=3.7
   conda activate Learn2Branch
   ```

   Install cmake version 3.22.1 and cython version 0.28.1.

2. ### Install SoPlex

   Upload the `SoPlex 4.0.1.tgz` file (free for academic uses) from the same folder to the Linux server where the environment needs to be set up, and run the following commands:

   ```
   tar -xzf soplex-4.0.1.tgz
   cd soplex-4.0.1/
   mkdir build
   cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR
   make -C ./build -j 4
   make -C ./build install
   cd ..
   ```

   Set the `SCIPOPTDIR` to the `build` folder generated during SoPlex compilation, for example:

   ```
   export SCIPOPTDIR=/home/sharing/disk1/soplex-4.0.1/build
   ```

3. ### Install SCIP

   Upload the `scip-6.0.1.tgz` file (free for academic uses) and the `vanillafullstrong.patch` file from the same folder to the Linux server where the environment needs to be set up, and run the following commands:

   ```
   tar -xzf scip-6.0.1.tgz
   cd scip-6.0.1/
   patch -p1 < ../vanillafullstrong.patch
   mkdir build
   cmake -S . -B build -DSOPLEX_DIR=$SCIPOPTDIR -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR
   make -C ./build -j 4
   make -C ./build install
   cd ..
   ```

4. ### Install PySCIPOpt

   Upload the customized version of `PySCIPOpt` (free for academic uses) from the same folder to the Linux server where the environment needs to be set up, and run the following commands:

   ```
   cd PySCIPOpt
   pip install .
   ```

5. ### Install Tensorflow

   ```
   conda install tensorflow-gpu=1.15.0
   ```

## Finish！
