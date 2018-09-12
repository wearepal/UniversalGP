--data=sensitive_from_numpy
--dataset_path=/Users/zc223/PycharmProjects/small-comparison/data_files/propublica-recidivism_race_0.npz
--inf=VariationalWithS
# --inf=Exact
--cov=SquaredExponential

# Eager mode
# --tf_mode=eager
# --train_steps=50
# --batch_size=5

# Graph mode
--tf_mode=graph

--plot=print_results
# --plot=simple_1d

--num_samples=100
--num_samples_pred=2000

# Don't save predictions
--preds_path=predictions.npz

--lr=0.001
--optimizer=AdamOptimizer
#--lr_drop_steps=50
#--lr_drop_factor=0.1
--train_steps=500
--num_components=1
--num_samples=1000
--diag_post=False
--optimize_inducing=True
--use_loo=False
--loo_steps=0
--length_scale=1.0
--sf=1.0
--iso=False
--num_samples_pred=2000
