# First four datasets, in csv format, originally used for forecasting
python run_pretrain.py --data_format csv --data_name ETTm1 --root_path ./datasets/ETT/ --data_path ETTm1.csv --data_split 34560,11520,11520 --checkpoints ./pretrain-library/ \
--data_dim 7 --patch_size 12 --min_patch_num 20 --max_patch_num 200 --mask_ratio 0.5 \
--batch_size 256 --train_steps 500000 --valid_freq 5000 --tolerance 10 --gpu 0 --label ETTm1-Base \
--efficient_loader True

python run_pretrain.py --data_format csv --data_name weather --root_path ./datasets/weather/ --data_path weather.csv --data_split 0.7,0.1,0.2 --checkpoints ./pretrain-library/ \
--data_dim 21 --patch_size 12 --min_patch_num 20 --max_patch_num 200 --mask_ratio 0.5 \
--batch_size 256 --train_steps 500000 --valid_freq 5000 --tolerance 10 --gpu 0 --label Weather-Base \
--efficient_loader True

python run_pretrain.py --data_format csv --data_name Electricity --root_path ./datasets/ECL/ --data_path ECL.csv --data_split 0.7,0.1,0.2 --checkpoints ./pretrain-library/ \
--data_dim 321 --patch_size 12 --min_patch_num 20 --max_patch_num 200 --mask_ratio 0.5 \
--batch_size 256 --train_steps 500000 --valid_freq 5000 --tolerance 10 --gpu 0 --label ECL-Base \
--efficient_loader True

python run_pretrain.py --data_format csv --data_name Traffic --root_path ./datasets/traffic/ --data_path traffic.csv --data_split 0.7,0.1,0.2 --checkpoints ./pretrain-library/ \
--data_dim 862 --patch_size 12 --min_patch_num 20 --max_patch_num 200 --mask_ratio 0.5 \
--batch_size 256 --train_steps 500000 --valid_freq 5000 --tolerance 10 --gpu 0 --label Traffic-Base \
--efficient_loader True

#Last four datasets, in npy format, originally used for anomaly detection
python run_pretrain.py --data_format npy --data_name SMD --root_path ./datasets/SMD/ --valid_prop 0.2 --checkpoints ./pretrain-library/ \
--data_dim 38 --patch_size 10 --min_patch_num 5 --max_patch_num 100 --mask_ratio 0.5 \
--train_steps 500000 --valid_freq 10000 --valid_batches 1000 --tolerance 10 --gpu 0 --label SMD-Base \
--efficient_loader True

python run_pretrain.py --data_format npy --data_name PSM --root_path ./datasets/PSM/ --valid_prop 0.2 --checkpoints ./pretrain-library/ \
--data_dim 25 --patch_size 10 --min_patch_num 5 --max_patch_num 100 --mask_ratio 0.5 \
--train_steps 500000 --valid_freq 10000 --valid_batches 1000 --tolerance 10 --gpu 0 --label PSM-Base \
--efficient_loader True

python run_pretrain.py --data_format npy --data_name SWaT --root_path ./datasets/SWaT/ --valid_prop 0.2 --checkpoints ./pretrain-library/ \
--data_dim 51 --patch_size 10 --min_patch_num 5 --max_patch_num 100 --mask_ratio 0.5 \
--train_steps 500000 --valid_freq 10000 --valid_batches 1000 --tolerance 10 --gpu 0 --label SWaT-Base \
--efficient_loader True

python run_pretrain.py --data_format npy --data_name NIPS_Water --root_path ./datasets/NIPS_Water/ --valid_prop 0.2 --checkpoints ./pretrain-library/ \
--data_dim 9 --patch_size 10 --min_patch_num 5 --max_patch_num 100 --mask_ratio 0.5 \
--train_steps 500000 --valid_freq 10000 --valid_batches 1000 --tolerance 10 --gpu 0 --label NIPS_Water-Base \
--efficient_loader True