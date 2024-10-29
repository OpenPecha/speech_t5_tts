jupyter nbconvert --to script  Train.ipynb
mkdir logs
nohup python Train.py > logs/train.log &

