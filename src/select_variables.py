"""
this script prepares variables for analysis by some criteria for the rest of the scripts
"""

from pathlib import Path
from src.clinical.modules.data_classes import Variable, Markdown
from rich import print
from src.utils.general import banner
import pandas

markdown = Markdown(Path("assets", "inspect.md"))
path_merged_data = Path("data", "1_clinical_merged", "clinical.csv")
all_variables = markdown.content.values()
all_variables = [item for sublist in all_variables for item in sublist]
all_variables = [Variable(**variable) for variable in all_variables]

banner("Filtering Variables")
selected_variables = all_variables
print(f"Total initial variables: {len(selected_variables)}.")

# filter out variables that are not included in the data or not not has a data file
selected_variables = [i for i in selected_variables if i.whether_included]
print(f"After cutting irrelavant variables: {len(selected_variables)}.")

selected_variables = [i for i in selected_variables if i.check_has_file()]
print(f"After cutting variables without data file: {len(selected_variables)}.")

selected_variables = [i for i in selected_variables if i.check_exist_in_data()]
print(f"After cutting variables not in data: {len(selected_variables)}.")

selected_variables = [i for i in selected_variables if i.check_percentage_available_merged(path_merged_data) > 50]
print(f"After cutting variables with less than 50% available data: {len(selected_variables)}.")
banner("End Filtering")

for variable in selected_variables:
    if variable.data_name == "TBMGrade":
        variable.nature = "outcome"
    else:
        variable.nature = "predictor"

    if variable.data_name in ["BirthYear", "Weight", "Height", "Gender"]:  # plus HIV severity
        variable.is_demographic = True
    else:
        variable.is_demographic = False


class Variables:
    def __init__(self, variables: list[Variable]):
        self.variables = variables

    def __iter__(self):
        return iter(self.variables)

    def retrieve(self, variable_name: str):
        """
        Retriving the variable instance by its name
        """
        # some special cases
        # The resulting feature names are meant to be interpreted in the context of one-hot encoding: requirehelp_opfu_Yes refers to the feature created for the Yes category of the requirehelp_opfu variable.
        if variable_name == "requirehelp_opfu_Yes":
            variable_name = "requirehelp_opfu"

        possible_variables = [
            i for i in self.variables if i.data_name.lower() == variable_name or i.data_name_in_merged.lower() == variable_name
        ]
        assert len(possible_variables) == 1, f"More than one or 0 variable found for {variable_name}: {possible_variables}"
        return possible_variables[0]

    def plot_table(self, selected_variables):
        data = []
        self.variables = [self.retrieve(i) for i in selected_variables]
        for i in self.variables:
            data.append([i.data_name, i.question_title])

        df = pandas.DataFrame(data, columns=["Data Name", "Question Title"])
        latex_code = df.to_latex(index=False, escape=False, caption="The selected variables clinical variables by random forests", label="tab:selected_variables")
        save_to = Path("results", "images", "selected_variables.tex")
        save_to.write_text(latex_code.replace("_", "\\_"))


variables = Variables(selected_variables)

if __name__ == "__main__":
    print(variables)
    pass
