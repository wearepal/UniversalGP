--data=simple_example
# --data=simple_multi_out
--inf=Variational
# --inf=Exact
--cov=SquaredExponential

# Eager mode
# --tf_mode=eager
# --train_steps=50
# --batch_size=5

# Graph mode
--tf_mode=graph

# No plotting
# --plot=
--plot=simple_1d

--num_samples=100
--num_samples_pred=2000

# Don't save predictions
--preds_path=''

# Uncomment this to save the trained model
# --save_dir=/its/home/tk324/tensorflow/
# --model_name=sm1
