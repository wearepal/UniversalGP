--data=flipped_labels
--inf=VariationalYbar

# --tf_mode=eager
# --train_steps=50
# --batch_size=5

--tf_mode=graph
--train_steps=400

--plot=simple_1d
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
--reject_flip_probability=0.3
--accept_flip_probability=0.3
--s_as_input=True
--probs_from_flipped=True
--flip_sensitive_attribute=False
--test_on_ybar=False

# --save_dir=/its/home/tk324/tensorflow/
# --model_name=fair2
# --save_preds=True
--logging_steps=1
--summary_steps=100
--chkpnt_steps=5000
