python Simu.py --train --sources 3



python runIPDnetOn.py fit \
  --data.batch_size=[16,16] \
  --trainer.devices=1 \
  --trainer.strategy=auto