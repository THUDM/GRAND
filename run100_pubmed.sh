mkdir pubmed
for num in $(seq 0 99)
do
	python train_grand.py --lam 1.0 --tem 0.2 --order 5 --sample 4 --dataset pubmed --input_droprate 0.6 --hidden_droprate 0.8 --hidden 32 --seed $num --dropnode_rate 0.5 --patience 100 --lr 0.2  --use_bn > pubmed/"$num".txt
done

