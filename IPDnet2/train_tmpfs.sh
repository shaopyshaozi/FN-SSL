#!/bin/bash -l
set -e

#$ -N ipdnet2_train
#$ -cwd
#$ -l gpu=1
#$ -l mem=32G
#$ -l tmpfs=50G
#$ -l h_rt=72:00:00
#$ -ac allow=UVL
#$ -m beas
#$ -M shaopengyuanspy@outlook.com
#$ -o /home/ucabps5/code/FN-SSL/IPDnet2/logs/train.log
#$ -e /home/ucabps5/code/FN-SSL/IPDnet2/logs/train.err

module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/12.2.2/gnu-10.2.0

mkdir -p $TMPDIR/extract
cd /home/ucabps5/code/FN-SSL/IPDnet2/RealMAN
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o train_gen.zip -d $TMPDIR/extract
unzip -o val_gen_1.zip -d $TMPDIR/extract
unzip -o test_gen_1.zip -d $TMPDIR/extract

cd /home/ucabps5/code/FN-SSL/IPDnet2/
source venv/bin/activate

echo "Running on $(hostname)"
which python
python --version
nvidia-smi

python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
EOF

python run_IPDnet2.py fit \
  --data.batch_size=[16,16]