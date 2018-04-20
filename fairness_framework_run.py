"""Class for algo-fairness framework"""
import pandas as pd
import numpy as np

from datasets.definition import Dataset, to_tf_dataset_fn
import universalgp as ugp


class UniversalGPAlgorithm():
    """
    This class calls the UniversalGP code
    """
    name = "UniversalGP"

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
        """
        Runs the algorithm and returns the predicted classifications on the test set.  The given
        train and test data still contains the sensitive_attrs.  This run of the algorithm
        should focus on the single given sensitive attribute.

        Be sure that the returned predicted classifications are of the same type as the class
        attribute in the given test_df.  If this is not the case, some metric analyses may fail to
        appropriately compare the returned predictions to their desired values.

        Args:
            train_df: Pandas datafram with the training data
            test_df: Pandas datafram with the test data
            class_attr:
            positive_class_val:
            sensitive_attrs:
            single_sensitive:
            privileged_vals:
            params: a dictionary mapping from algorithm-specific parameter names to the desired values.
                If the implementation of run uses different values, these should be modified in the params
                dictionary as a way of returning the used values to the caller.
        """
        # remove sensitive attributes from the training set
        train_sensitive = train_df[single_sensitive]
        train_target = train_df[class_attr]
        train_df_nosensitive = train_df.drop(columns=sensitive_attrs)
        train_df_nosensitive = train_df_nosensitive.drop(columns=class_attr)

        test_sensitive = test_df[single_sensitive]
        test_sensitive = pd.DataFrame()
        test_sensitive = pd.get_dummies(test_sensitive)
        test_target = test_df[class_attr]
        test_df_nosensitive = test_df.drop(columns=sensitive_attrs)
        test_df_nosensitive = test_df_nosensitive.drop(columns=class_attr)

        data_train = (train_df_nosensitive.values, train_sensitive.values.reshape(-1, 1),
                      train_target.values.reshape(-1, 1))
        data_test = (test_df_nosensitive.values, test_sensitive.values.reshape(-1, 1),
                     test_target.values.reshape(-1, 1))

        x_size = data_train[0].shape[1]
        s_size = data_train[1].shape[1]
        y_size = data_train[2].shape[1]

        dataset = Dataset(
            train_fn=to_tf_dataset_fn(data_train[0], data_train[2], data_train[1]),
            test_fn=to_tf_dataset_fn(data_test[0], data_test[2], data_test[1]),
            input_dim=x_size,
            xtrain=data_train[0],
            ytrain=data_train[2],
            strain=data_train[1],
            xtest=data_test[0],
            ytest=data_test[2],
            stest=data_test[1],
        )
        gp = ugp.train_eager.train_gp(dataset, dict(
            inf="Variational",
            cov="SquaredExponential",
            plot="",
            lr=0.005,
        ))
        return gp.predict(data_test[0])

    def get_param_info(self):
        """
        Returns a dictionary mapping algorithm parameter names to a list of parameter values to
        be explored.  This function should only be implemented if the algorithm has specific
        parameters that should be tuned, e.g., for trading off between fairness and accuracy.
        """
        return {}

    def get_supported_data_types(self):
        """
        Returns a set of datatypes which this algorithm can process.
        """
        raise NotImplementedError("get_supported_data_types() in Algorithm is not implemented")

    def get_name(self):
        """
        Returns the name for the algorithm.  This must be a unique name, so it is suggested that
        this name is simply <firstauthor>.  If there are mutliple algorithms by the same author(s), a
        suggested modification is <firstauthor-algname>.  This name will appear in the resulting
        CSVs and graphs created when performing benchmarks and analysis.
        """
        return self.name

    def get_default_params(self):
        """
        Returns a dictionary mapping from parameter names to default values that should be used with
        the algorithm.  If not implemented by a specific algorithm, this returns the empty
        dictionary.
        """
        return {}
