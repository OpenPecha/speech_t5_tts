jupyter nbconvert --to script  Dataset.ipynb
mkdir logs
rm logs/dataset.log
nohup python Dataset.py > logs/dataset.log &

