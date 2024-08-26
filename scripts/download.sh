SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

# cd $SCRIPTPATH/..
cd /Users/lee/Git\ Projects/pointnet.pytorch
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
rm shapenetcore_partanno_segmentation_benchmark_v0.zip
cd -
