
# maia2

nohup python unseen.py --low_resource 800 --gpu_id 0 --batch_size_train 800 --num_workers 0 --base_model maia2 >../log/unseen_maia2_800.log 2>&1 &

nohup python unseen.py --low_resource 2000 --gpu_id 0 --batch_size_train 2000 --num_workers 2 --base_model maia2 >../log/unseen_maia2_2000.log 2>&1 &

nohup python unseen.py --low_resource 8000 --gpu_id 0 --batch_size_train 4000 --num_workers 4 --base_model maia2 >../log/unseen_maia2_8000.log 2>&1 &

nohup python unseen.py --low_resource 20000 --gpu_id 0 --batch_size_train 4000 --num_workers 4 --base_model maia2 >../log/unseen_maia2_20000.log 2>&1 &

nohup python unseen.py --low_resource 0 --gpu_id 0 --batch_size_train 4096 --num_workers 4 --base_model maia2 >../log/unseen_maia2_full.log 2>&1 &



# prototype 100

nohup python unseen.py --low_resource 800 --gpu_id 0 --batch_size_train 800 --num_workers 0 --base_model prototype --checkpoint_step 200000 >../log/unseen_100_800.log 2>&1 &

nohup python unseen.py --low_resource 2000 --gpu_id 0 --batch_size_train 2000 --num_workers 2 --base_model prototype --checkpoint_step 200000 >../log/unseen_100_2000.log 2>&1 &

nohup python unseen.py --low_resource 8000 --gpu_id 0 --batch_size_train 4000 --num_workers 4 --base_model prototype --checkpoint_step 200000 >../log/unseen_100_8000.log 2>&1 &

nohup python unseen.py --low_resource 20000 --gpu_id 0 --batch_size_train 4000 --num_workers 4 --base_model prototype --checkpoint_step 100000 >../log/unseen_100_20000.log 2>&1 &

nohup python unseen.py --low_resource 0 --gpu_id 0 --batch_size_train 4096 --num_workers 4 --base_model prototype --checkpoint_step 100000 >../log/unseen_100_full.log 2>&1 &