--data=flipped_labels
--inf=LogisticRegression

--tf_mode=eager
--train_steps=100
--batch_size=500

# --tf_mode=graph

# --s_as_input=False
--s_as_input=True

--plot=simple_1d
--num_samples=1000
--num_samples_pred=2000
# --cov=Linear
# --target_rate1=0.7
# --target_rate2=0.3
# --p_ybary0_s0=0.93
# --p_ybary1_s0=.6
# --p_ybary0_s1=.67
# --p_ybary1_s1=.6
# --biased_acceptance1=0.288
# --biased_acceptance2=0.760
--num_samples=1000
--num_samples_pred=2000
--num_all=3000
--num_train=1500
--num_inducing=500
# --cov=Linear
--target_rate1=0.5
--target_rate2=0.5
# --biased_acceptance1=0.35
# --biased_acceptance2=0.50
# --reject_flip_probability=0.3
# --accept_flip_probability=0.3
--reject_flip_probability=0.0
--accept_flip_probability=0.0
--probs_from_flipped=True
--flip_sensitive_attribute=False
--test_on_ybar=False

--lr=0.005
--optimizer=GradientDescentOptimizer
# --save_dir=/its/home/tk324/tensorflow/
# --model_name=fair1
