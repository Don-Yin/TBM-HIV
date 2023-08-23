"""
Splitting train and test participants
"""

from src.select_variables import variables
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import random
import json
import os

# -------- reading the data --------
imputation_data_folder = Path("data", "3_clinical_imputations")
imputation_data_files = [i for i in os.listdir(imputation_data_folder) if i.endswith(".pkl")]
data = pd.read_pickle(Path(imputation_data_folder, imputation_data_files[0]))

variables_to_be_balanced = [i for i in variables if i.nature == "outcome" or i.is_demographic]
variables_to_be_balanced = [i.data_name_in_merged for i in variables_to_be_balanced]
variables_to_be_balanced.append("project")


class SplitAndBalanceData:
    def __init__(self):
        self.split_ratio = [4, 1, 1]  # train / valid / test; [4, 1, 1]
        self.batch_sizes = [1, 2, 3]  # number of samples to exchange between two random sets each time
        self.max_iteration = 30000  # max number of iterations to attempt before giving up
        self.save_image_to = Path("results", "images", "split_data")
        self.save_image_to.mkdir(parents=True, exist_ok=True)

        # -------- calculate samples sizes --------
        self.samples_sizes = [(len(data) * i / sum(self.split_ratio)) for i in self.split_ratio]
        self.samples_sizes = [int(i) for i in self.samples_sizes]
        self.samples_sizes[-1] = len(data) - sum(self.samples_sizes[:-1])  # make sure the sum is correct
        error_message = f"Sum of the currnet split: {sum(self.samples_sizes)}; total data size: {len(data)}"
        assert sum(self.samples_sizes) == len(data), error_message

        # -------- setup progress bar --------
        self.progress_bar = tqdm(total=self.max_iteration)

        # self.start_from_scratch()
        self.start_from_saved()

    def start_from_scratch(self):
        # -------- main loop --------
        self.indices = self.initial_step()  # this is a list containing n lists of indices
        for _ in range(self.max_iteration):
            self.mutate()
            if self.best_ks_mean <= 0.05:
                self.on_finish()
                break

        self.on_finish()

    def start_from_saved(self):
        self.load_saved()  # loading the indices for further mutation
        self.plot_distribution()

    def on_finish(self):
        self.progress_bar.close()
        # retrive the subject ids from the indices
        self.subject_ids = [data["subject_id"][self.indices[i]].tolist() for i in range(len(self.indices))]

        # -------- save the subject ids and a explanatory message --------
        self.plot_distribution()
        splited_dict = {"train": self.subject_ids[0], "valid": self.subject_ids[1], "test": self.subject_ids[2]}
        save_message = f"Current max KS stats across variables for train / valid / test: {self.get_ks_stats()}"
        json.dump(splited_dict, open(Path("cache", "split.json"), "w"))
        with open(Path("cache", "README.MD"), "w") as f:
            f.write(save_message)

    def initial_step(self):
        total_indices = list(range(len(data)))
        initial_indices = []
        for size in self.samples_sizes:
            indices = random.sample(total_indices, size)
            total_indices = [index for index in total_indices if index not in indices]
            initial_indices.append(indices)
        return initial_indices

    def get_ks_stats(self):
        """
        getting pair-wise ks statistics for the current split
        this returns three max ks statistics for each pair of the three sets
        """
        pairs = list(itertools.combinations(range(len(self.indices)), 2))
        ks_stats_across_pairs = []
        for pair in pairs:
            ks_stats_across_variables = [
                ks_2samp(
                    data.iloc[self.indices[pair[0]]][variable],
                    data.iloc[self.indices[pair[1]]][variable],
                ).statistic
                for variable in variables_to_be_balanced
            ]
            ks_stats_across_pairs.append(max(ks_stats_across_variables))
        return ks_stats_across_pairs

    def mutate(self):
        """
        this function randomly exchange samples between two sets in the initial indices
        """
        # backup self.indices
        backup_indices = deepcopy(self.indices)
        self.best_ks_mean = np.mean(self.get_ks_stats())
        self.best_ks_max = np.max(self.get_ks_stats())
        self.current_best_ks = self.best_ks_mean + self.best_ks_max

        # take two random index between 0 and len(self.ratio)
        random_pair = random.sample(range(len(self.indices)), 2)
        batch_size = random.choice(self.batch_sizes)

        # from the two random sets, take batch_size number of samples from each set and exchange them
        random_indices_1 = random.sample(self.indices[random_pair[0]], batch_size)
        random_indices_2 = random.sample(self.indices[random_pair[1]], batch_size)

        # remove the random indices from the original sets
        self.indices[random_pair[0]] = [i for i in self.indices[random_pair[0]] if i not in random_indices_1]
        self.indices[random_pair[1]] = [i for i in self.indices[random_pair[1]] if i not in random_indices_2]

        # add the random indices to the other sets
        self.indices[random_pair[0]] = self.indices[random_pair[0]] + random_indices_2
        self.indices[random_pair[1]] = self.indices[random_pair[1]] + random_indices_1

        # update the progress bar
        new_ks_mean = np.mean(self.get_ks_stats())
        new_ks_max = np.max(self.get_ks_stats())
        new_ks_stats = new_ks_mean + new_ks_max
        self.progress_bar.update(1)
        self.progress_bar.set_description(f"Best KS mean: {self.best_ks_mean:.3f} | Best KS max: {self.best_ks_max:.3f}")

        # if the new ks stats is better than the previous one, keep the new indices
        if new_ks_stats < self.current_best_ks:
            self.best_ks_mean = new_ks_mean
            self.best_ks_max = new_ks_max
            self.current_best_ks = new_ks_stats
        else:
            self.indices = backup_indices

    def plot_distribution(self):
        """
        plot the distribution of the current split
        """
        fig, axs = plt.subplots(len(variables_to_be_balanced), 4, figsize=(22, len(variables_to_be_balanced) * 5))

        # birthyear_base: Birth Year
        # gender_base: Gender
        # weight_base: Weight
        # height_base: Height
        # tbmgrade_followupsigns: TBM Grade
        # project: Project

        title_recode_dict = {
            "birthyear_base": "Birth Year",
            "gender_base": "Gender",
            "weight_base": "Weight",
            "height_base": "Height",
            "tbmgrade_followupsigns": "TBM Grade",
            "project": "Project",
        }

        for i, var in enumerate(variables_to_be_balanced):  # order matters
            order = data[var].value_counts().index
            data[var] = pd.Categorical(data[var], categories=order, ordered=True)
            for index in range(len(self.indices)):
                data.loc[data.index[self.indices[index]], var] = pd.Categorical(
                    data.loc[data.index[self.indices[index]], var], categories=order, ordered=True
                )

            recoded_title = title_recode_dict[var]

            sns.histplot(data[var], kde=True, color="green", ax=axs[i, 0], stat="percent")
            axs[i, 0].set_title(f"Original Data - {recoded_title}", fontsize=15)

            sns.histplot(data.iloc[self.indices[0]][var], kde=True, color="blue", ax=axs[i, 1], stat="percent")
            axs[i, 1].set_title(f"Train Set - {recoded_title}", fontsize=15)

            sns.histplot(data.iloc[self.indices[1]][var], kde=True, color="orange", ax=axs[i, 2], stat="percent")
            axs[i, 2].set_title(f"Valid Set - {recoded_title}", fontsize=15)

            sns.histplot(data.iloc[self.indices[2]][var], kde=True, color="red", ax=axs[i, 3], stat="percent")
            axs[i, 3].set_title(f"Test Set - {recoded_title}", fontsize=15)

        plt.tight_layout()
        plt.savefig(self.save_image_to / "split_data.png", dpi=420)

    # ---------------- loading the samples for further mutation ----------------

    def load_saved(self, path=Path("cache") / "split.json"):
        with open(path, "r") as f:
            self.subject_ids = json.load(f)

        # return the subject IDs to the indices
        self.indices = [
            [data[data["subject_id"] == i].index[0] for i in self.subject_ids["train"]],
            [data[data["subject_id"] == i].index[0] for i in self.subject_ids["valid"]],
            [data[data["subject_id"] == i].index[0] for i in self.subject_ids["test"]],
        ]


if __name__ == "__main__":
    split_data = SplitAndBalanceData()
