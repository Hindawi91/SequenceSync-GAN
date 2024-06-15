for i in {10000..200000..10000}
  do 
     python3 main.py --mode val --dataset Boiling --image_size 256 --c_dim 1 \
                 --image_dir ./data \
                 --result_dir boiling/results_$i \
                 --batch_size 8 --num_workers 4 --lambda_id 0.1 --test_iters $i \
                 --random_seed 42 \
                 --direction B2A
 done