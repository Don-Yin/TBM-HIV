import pandas
from pathlib import Path
import numpy as np
from rich import print
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# LinearRegression, RandomForestRegressor, or KNeighborsRegressor.
from sklearn.neighbors import KNeighborsRegressor


path_data = Path("data", "2_clinical_only_selected", "clinical.csv")
save_path = Path("data", "3_clinical_imputations")
data = pandas.read_csv(path_data)

non_numeric_columns = []
for column in data.columns:
    if not pandas.api.types.is_numeric_dtype(data[column]):
        non_numeric_columns.append(column)

non_numeric_columns = [i for i in non_numeric_columns if i not in ["subject_id", "project"]]

# recode -> imputation datatype?
recode_dicts = {
    "gender_base": {"Female": 0, "Male": 1},
    "weightloss_clinical": {"Yes": 1, "No": 0},
    "paraplegia_clinical": {"Yes": 1, "No": 0},
    "tetraplegia_clinical": {"Yes": 1, "No": 0},
    "urinaryretention_clinical": {"Yes": 1, "No": 0},
    "papilloedema_clinical": {"Yes": 1, "No": 0},
    "newneue_followupsigns": {"Yes": 1, "No": 0},
    "requirehelp_opfu": {"Yes": 1, "No": 0},
    "disproblem_opfu": {"Yes": 1, "No": 0},
}

# the main bit of recoding and set the dtype to int
for column in recode_dicts.keys():
    data[column] = data[column].replace(recode_dicts[column])

# remove label columns
subject_id, project = data["subject_id"], data["project"]
data = data.drop(["subject_id", "project"], axis=1)

# calculate the number of imputations we need by Rubin's rule
"""
Instead of filling in a single value for each missing value, Rubin's (1987) multiple imputation procedure replaces each missing value with a set of plausible values that represent the uncertainty about the right value to impute.
"""
missing_data_ratio = data.isnull().sum().sum() / np.product(data.shape)
missing_data_percentage = round(missing_data_ratio * 100)
print(f"Missing data percentage: {missing_data_percentage}%")
num_imputations = max(5, missing_data_percentage)


"""
During the imputation, the imputation needs to know which columns are float and which are int as we don't want to assign float values with several decimal points to int columns. So we need to define the float and int columns here.
"""
# float columns is defined as when dropping the NA values, the unique values have at least 1 have a decimal point
# later just round all the int column values
float_columns = []
for column in data.columns:
    if data[column].dropna().apply(lambda x: int(x) != x).any():
        float_columns.append(column)

int_columns = [i for i in data.columns if i not in float_columns]

if __name__ == "__main__":
    """
    Note that this imputation method here is currently assuming all the values to be float and round them to int at the end. It is unclear to what extent this method is appropriate in this case.

    Note that initially the n_neighbors is set to 5 but it does not converge at all. The current value of 16 is experimented with and it seems to work. But it is unclear how to choose this value systematically.
    """
    for i in range(num_imputations):
        imputer = IterativeImputer(
            estimator=KNeighborsRegressor(n_neighbors=16),  # Adjust n_neighbors as needed
            missing_values=np.nan,
            sample_posterior=False,  # Not used with KNeighborsRegressor
            max_iter=1000,
            tol=1e-3,
            n_nearest_features=None,
            initial_strategy="mean",
            imputation_order="ascending",  # from features with fewest missing values to most.
            skip_complete=False,
            min_value=-np.inf,
            max_value=np.inf,
            verbose=1,
            random_state=i,
            add_indicator=False,
        )

        imputed_data = imputer.fit_transform(data)
        imputed_data_df = pandas.DataFrame(imputed_data, columns=data.columns)

        # round the int columns
        for column in int_columns:
            imputed_data_df[column] = imputed_data_df[column].round(0).astype(int)

        for column in recode_dicts.keys():
            # recode back
            imputed_data_df[column] = imputed_data_df[column].replace({v: k for k, v in recode_dicts[column].items()})
            imputed_data_df[column] = imputed_data_df[column].astype("category")

        # add back the label columns and put them in the front
        imputed_data_df = pandas.concat([subject_id, project, imputed_data_df], axis=1)
        imputed_data_df.to_pickle(save_path / f"imputed_data_{i}.pkl")
