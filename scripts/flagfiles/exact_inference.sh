--inf=Exact
--tf_mode=eager
--data=simple_example
# --data=simple_multi_out
--cov=SquaredExponential

--train_steps=50
--batch_size=50

# --plot=
--plot=simple_1d

--num_samples=100
--num_samples_pred=2000

# Don't save predictions
--preds_path=

--lr=0.05
--optimizer=RMSPropOptimizer
--lr_drop_steps=10
--lr_drop_factor=0.1
