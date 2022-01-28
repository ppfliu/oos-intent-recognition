export CUDA_VISIBLE_DEVICES=0

mkdir -p batch-logs

# Single Models
for dataset in oos_full oos_plus oos_small oos_imbalanced
do
  rm -rf checkpoints
  make train-single dataset=${dataset}
  sleep 5
  cp checkpoints/best_model*.pt best_model_single.pt
  for size in 1
  do
    make predict-single dataset=${dataset} test_batch_size=${size} > batch-logs/single-${dataset}-${size}.log
  done
done


# Joint Models
for dataset in oos_full oos_plus oos_small oos_imbalanced
do
  for flag in true false
  do
      rm -rf checkpoints
      make train dataset=${dataset} domain_first=${flag}
      sleep 5
      cp checkpoints/best_model*.pt best_model.pt
      for size in 1
      do
        make predict dataset=${dataset} domain_first=${flag} test_batch_size=${size} > batch-logs/joint-${dataset}-${flag}-${size}.log
      done
  done
done

