# Distributed Graph Neural Network Learning

## Install Metis

download and extract metis:

    http://glaros.dtc.umn.edu/gkhome/metis/metis/download
    gunzip metis-5.x.y.tar.gz
    tar -xvf metis-5.x.y.tar

if you don't have root access, compile metis with `prefix`

    make config prefix=<lib-folder>


## Build PyTorch Sparse

make Pytorch Sparse with metis support:

    cd pytorch_sparse
    export CPATH=/export/local/mfr5226/lib/metis/include/:/usr/local/cuda/include/:$CPATH 
    export LD_LIBRARY_PATH=/export/local/mfr5226/lib/metis/lib/:$LD_LIBRARY_PATH 
    WITH_METIS=1 LDFLAGS='-L/export/local/mfr5226/lib/metis/lib/' python setup.py build -j 10
    python setup.py install

If you have metis in regular location, just compile with:
    WITH_METIS=1 python setup.py install -j 10