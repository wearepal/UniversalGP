--data=sensitive_example
--inf=VariationalYbar

# --tf_mode=eager
# --train_steps=50
# --batch_size=5

--tf_mode=graph

--plot=simple_1d
--num_samples=1000
--num_samples_pred=2000
# --cov=Linear
--target_rate1=0.7
--target_rate2=0.3
--p_ybary0_s0=0.93
--p_ybary1_s0=.6
--p_ybary0_s1=.67
--p_ybary1_s1=.6
--biased_acceptance1=0.288
--biased_acceptance2=0.760
# --save_dir=/its/home/tk324/tensorflow/
# --model_name=fair1
