TASK='Bangs'
EXP_NAME='GeneGAN_${TASK}'
DATA_PATH='./datasets/'
PORT=8087

train:
	python3 train.py --dataroot ${DATA_PATH} --name ${EXP_NAME} \
		--model genegan --gpu_ids 0,1,2,3 --batch_size 64 --task ${TASK} --dataset_mode unaligned

test:
	python3 test.py --dataroot ${DATA_PATH} --name ${EXP_NAME} \
		--model genegan --task ${TASK} --dataset_mode unaligned --eval

show_train:
	-cd 'checkpoints/${EXP_NAME}/tb_log' && tensorboard --logdir=.

show_test:
	-cd 'results/${EXP_NAME}/test_latest' && tensorboard --logdir=.

clean:
	-rm -r checkpoints results
