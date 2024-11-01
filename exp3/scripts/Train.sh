jupyter nbconvert --to script  Train.ipynb
mkdir logs
rm logs/train.log
nohup python Train.py > logs/train.log &

