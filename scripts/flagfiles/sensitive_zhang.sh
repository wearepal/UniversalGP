--data=sensitive_zhang_simple
# --inf=VariationalYbar
--inf=VariationalWithS

# --tf_mode=eager
# --train_steps=50
# --batch_size=5

--tf_mode=graph

--plot=
--num_samples=1000
--num_samples_pred=2000
# --cov=Linear
--target_rate1=0.573
--target_rate2=0.64
--biased_acceptance1=0.518
--biased_acceptance2=0.642
# --save_dir=/its/home/tk324/tensorflow/
--model_name=fair1
