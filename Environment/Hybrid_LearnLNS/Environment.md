## About

The original environment for Hybrid-Learn2Branch. We have carefully compiled the relevant packages and resources to create this improved setup.

## Steps

1. ### Initialisation

   First, install conda and run the following commands:

   ```bash
   conda create -n Learn2Branch python=3.7
   conda activate Learn2Branch
   ```

   Install cmake version 3.22.1 and cython version 0.29.13.

   Set the `SCIPOPTDIR` environment variable before proceeding:

   ```bash
   export SCIPOPTDIR='/home/your_name/opt/scip'
   ```

2. ### Install SoPlex

   Upload the `SoPlex 4.0.1.tgz` file (free for academic use) from the same folder to the Linux server where the environment needs to be set up.

   Run the following commands to install SoPlex:

   ```bash
   tar -xzf soplex-4.0.1.tgz
   cd soplex-4.0.1/
   mkdir build
   cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR
   make -C ./build -j 4
   make -C ./build install
   cd ..
   ```

   Confirm that `SCIPOPTDIR` is correctly pointing to the compiled SoPlex `build` folder:

   ```bash
   export SCIPOPTDIR=/home/your_name/opt/scip
   ```

3. ### Install Readline (Required for SCIP)

   Before installing SCIP, you need to install the `readline` library. Run the following commands:

   ```bash
   wget http://ftp.gnu.org/gnu/readline/readline-8.1.tar.gz
   tar -xvzf readline-8.1.tar.gz
   cd readline-8.1
   ./configure --prefix=$HOME/readline
   make
   make install
   export C_INCLUDE_PATH=$HOME/readline/include:$C_INCLUDE_PATH
   export LIBRARY_PATH=$HOME/readline/lib:$LIBRARY_PATH
   export LD_LIBRARY_PATH=$HOME/readline/lib:$LD_LIBRARY_PATH
   source ~/.bashrc
   ```

4. ### Install SCIP

   Upload the `scip-6.0.1.tgz` file (free for academic use) and the `vanillafullstrong.patch` file from the same folder to the Linux server.

   Run the following commands to install SCIP:

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

5. ### Install PySCIPOpt

   Upload the customized version of `PySCIPOpt` (free for academic use) from the same folder to the Linux server.

   Modify `src/pyscipopt/scip.pyx` in the `PySCIPOpt` source code by adding the following function in the `class Column`:

   ```python
   def getIndex(self):
       return SCIPcolGetIndex(self.scip_col)
   ```

   Run the following commands to install `PySCIPOpt`:

   ```bash
   cd PySCIPOpt-hybrid-l2b
   pip install .
   ```

6. ### Install Tensorflow

   Install TensorFlow GPU version 1.15.0 using `conda`:

   ```
   conda install tensorflow-gpu=1.15.0
   ```

## Finish!

The environment setup is now complete. Make sure all dependencies and paths are correctly configured before running the program.
