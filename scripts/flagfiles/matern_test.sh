--data=simple_example
# --data=simple_multi_out
--inf=Variational
# --inf=Exact
--cov=Matern
--order=5

# Eager mode
# --tf_mode=eager
# --train_steps=50
# --batch_size=5

# Graph mode
--tf_mode=graph
# --tf_mode=eager

# No plotting
# --plot=
--plot=simple_1d

--num_samples=100
--num_samples_pred=2000

# Don't save predictions
--preds_path=

--lr=0.005
--optimizer=RMSPropOptimizer
--train_steps=500

# Uncomment this to save the trained model
# --save_dir=/its/home/tk324/tensorflow/
# --model_name=sm1
