--data=simple_example
--inf=Variational
--cov=SquaredExponential

# Eager mode
# --tf_mode=eager
# --train_steps=50
# --batch_size=5

# Graph mode
--tf_mode=graph

# No plotting
--plot=

--num_samples=100
--num_samples_pred=2000

# Don't save predictions
--save_preds=False

# Uncomment this to save the trained model
# --save_dir=/its/home/tk324/tensorflow/
# --model_name=sm1
