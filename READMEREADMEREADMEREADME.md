conda activate image_classification


python test_train_data.py
python test.py

python visualize_errors.py


CUDA_VISIBLE_DEVICES=2 nohup python main.py 1>log.log 2>&1 &


