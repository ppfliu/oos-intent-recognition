export CUDA_VISIBLE_DEVICES=0

train-single:
	python train_single.py --epochs 10 --train_batch_size 32 --patience 3 --lr 4.0E-5 --data_path dataset/${dataset}/ \
		--warmup_proportion 0.1 --dropout 0.3 --gradient_accumulation_steps 1 --pretrained bert-base-uncased

train:
	python train.py --epochs 10 --train_batch_size 32 --patience 3 --lr 4.0E-5 --data_path dataset/${dataset}/ \
		--warmup_proportion 0.1 --dropout 0.5 --gradient_accumulation_steps 1 --pretrained bert-base-uncased \
		--domain_first ${domain_first} --subspace true --hierarchy true

predict-single:
	python predict.py --single --pretrained bert-base-uncased --threshold 0 \
		--model_path best_model_single.pt --data_path dataset/${dataset}/ --test_batch_size ${test_batch_size}

predict:
	python predict.py --pretrained bert-base-uncased --threshold 0 --data_path dataset/${dataset}/ \
		--domain_first ${domain_first} --model_path best_model.pt --test_batch_size ${test_batch_size} --subspace true --hierarchy true
