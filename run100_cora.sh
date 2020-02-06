mkdir cora
for num in $(seq 0 99)
do
	python train_grand.py --lam 1.0 --tem 0.5 --order 8 --sample 4 --dataset cora --input_droprate 0.5 --hidden_droprate 0.5 --hidden 32 --lr 0.01 --patience 200 --seed $num --dropnode_rate 0.5  --cuda_device 4 > cora/"$num".txt
done
