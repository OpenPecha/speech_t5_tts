jupyter nbconvert --to script  Dataset.ipynb
mkdir logs
nohup python Dataset.py > logs/dataset.log &

