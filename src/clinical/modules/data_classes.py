from rich import print
from pathlib import Path
import pandas
from src.clinical.planning.variable_selection_constants import (
    SHEET_NAMES,
    DICT_FORM_DETAILS,
    PATH_SHEETS_ORIGINAL,
    COLUMN_NAME_ERRORS,
)
import re


class Form:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.description = kwargs.get("description")
        self.index = kwargs.get("index")
        self.level = kwargs.get("level")
        self.title = kwargs.get("title")
        self.variables = kwargs.get("variables")

        self._parse_variables()

    def _parse_variables(self):
        """Run all functions that gets ready the secondary variables"""
        # to check whether there is a datafile available run self.path_data_file.exists()
        self.path_data_file = Path("data", "clinical", f"oucru:{self.name}.csv")
        self.dataframe = pandas.read_excel(
            PATH_SHEETS_ORIGINAL, sheet_name=self.index.replace(" ", "")
        )
        # rename dataframe columns using locator_dict - name the column as the key if the current name is in the list of values
        for key, value in COLUMN_NAME_ERRORS.items():
            for column_name in self.dataframe.columns:
                if column_name in value:
                    self.dataframe.rename(columns={column_name: key}, inplace=True)

        # make a new column called "full_name" which is the data_name + subname, if subname is NaN then just use the data_name
        self.dataframe["full_name"] = self.dataframe.apply(
            lambda row: row["data_name"]
            if pandas.isna(row["subname"])
            else row["data_name"] + "_" + row["subname"],
            axis=1,
        )


FORMS = [Form(**form_info) for form_info in DICT_FORM_DETAILS]


class Variable:
    def __init__(self, **kwargs):
        self.form = [form for form in FORMS if form.index == kwargs.get("form")][0]
        self.question_title = kwargs.get("question_title")
        self.data_name = kwargs.get("data_name")
        self.whether_included = kwargs.get("whether_included")
        self.reason = kwargs.get("reason")
        self.percentage_of_data_available = self.check_percentage_available()

        self.data_name_in_merged = (
            self.data_name
            + "_"
            + self.form.path_data_file.stem.split(".")[0].split(":")[1]
        ).lower()

    def check_exist_in_sheet(self):
        """Check whether the variable exists in the variable sheet"""
        return self.data_name in self.form.dataframe["full_name"].values

    def check_exist_in_data(self):
        """Check whether the variable exists in the data"""
        if self.check_has_file():
            frame = pandas.read_csv(self.form.path_data_file)
            return self.data_name.lower() in frame.columns.str.lower().values
        else:
            return False

    def check_exist_in_merged(self, path_merged_data: Path):
        """Check whether the variable exists in the merged data"""
        frame = pandas.read_csv(path_merged_data)
        return self.data_name_in_merged.lower() in frame.columns.str.lower().values

    def check_has_file(self):
        """Check whether the variable has a file"""
        return self.form.path_data_file.exists()

    def check_percentage_available(self):
        if self.check_has_file():
            frame = pandas.read_csv(self.form.path_data_file)
            # change the column names to lower case
            frame.columns = frame.columns.str.lower()
            # check whether the data_name is in the columns
            if self.data_name.lower() in frame.columns:
                # if it is, then check the percentage of data available
                total_rows = len(frame)
                non_null_rows = len(frame[self.data_name.lower()].dropna())
                percentage_available = (non_null_rows / total_rows) * 100
                return percentage_available
            else:
                return 0
        else:
            return 0

    def check_percentage_available_merged(self, path_merged_data: Path):
        if path_merged_data.exists():
            frame = pandas.read_csv(path_merged_data)
            # change the column names to lower case
            frame.columns = frame.columns.str.lower()
            # check whether the data_name_in_merged is in the columns
            if self.data_name_in_merged.lower() in frame.columns:
                # if it is, then check the percentage of data available
                total_rows = len(frame)
                non_null_rows = len(frame[self.data_name_in_merged.lower()].dropna())
                percentage_available = (non_null_rows / total_rows) * 100
                return percentage_available
            else:
                return 0
        else:
            return 0

    def __repr__(self) -> str:
        return f"\n\nForm: {self.form.name},\n\nQuestion: {self.question_title},\n\nData name: {self.data_name},\n\nWhether included: {self.whether_included},\n\nReason: {self.reason},\n\nPercentage of data available: {self.percentage_of_data_available}\n\n"


class Markdown:
    def __init__(self, path: Path):
        self.path = path
        self.raw_content = path.read_text()
        self.content = self.extract_forms()
        for key, value in self.content.items():
            self.content[key] = self.parse_form_content(key, value)

    def extract_forms(self):
        forms = {}
        form_pattern = r"## (Form \d+)"
        form_matches = re.finditer(form_pattern, self.raw_content)
        form_positions = [match.start() for match in form_matches]

        for i, form_start in enumerate(form_positions):
            form_name = self.raw_content[
                form_start + 3 : self.raw_content.find("\n", form_start)
            ].strip()
            if i + 1 < len(form_positions):
                form_end = form_positions[i + 1]
                form_content = self.raw_content[form_start:form_end].strip()
            else:
                form_content = self.raw_content[form_start:].strip()
            forms[form_name] = form_content
        return forms

    def parse_form_content(self, which_form, form_content):
        item_dicts = []
        item_sections = form_content.split("------")

        for section in item_sections:
            item_dict = {}
            lines = section.strip().split("\n")
            for line in lines:
                if "### Item:" in line:
                    item_dict["question_title"] = line.replace("### Item:", "").strip()
                elif "Variable name:" in line:
                    item_dict["data_name"] = line.replace("Variable name:", "").strip()
                elif "Whether included:" in line:
                    item_dict["whether_included"] = (
                        "True" in line.replace("Whether included:", "").strip()
                    )
                elif "Reasons:" in line:
                    item_dict["reason"] = line.replace("Reasons:", "").strip()
                # elif "Percentage of data available:" in line:
                #     item_dict["percentage_of_data_available"] = line.replace(
                #         "Percentage of data available:", ""
                #     ).strip()
            if item_dict:
                item_dict["form"] = which_form
                item_dicts.append(item_dict)
        return item_dicts


class ClinicalData:
    """
    Each object is an csv file in the clinical data folder with a set of functions to
    check the data quality and merge with other dataframes.
    """

    def __init__(self, path: Path):
        self.files_path = path
        self.filename = path.name
        self.dataframe = pandas.read_csv(path)

        self.keep_earliest_date_each_subject()
        self.rename_columns()

    def rename_columns(self):
        # rename dataframe columns other than subject ID to avoid name conflicts
        # when merging dataframes
        self.dataframe.rename(
            columns={i: self.get_new_colname(i) for i in self.dataframe.columns},
            inplace=True,
        )

    def get_new_colname(self, colname: str):
        if colname == "subject_id":
            return colname
        return colname + "_" + self.filename.split(".")[0].split(":")[1]

    def check_has_essential_columns(self):
        return all([i in self.dataframe.columns for i in ["subject_id", "age", "date"]])

    def check_num_unique_subjects(self):
        num = self.dataframe.groupby(["subject_id"]).size().shape[0]
        total_rows = self.dataframe.shape[0]
        print(f"{self.filename}: {num} / {total_rows}")
        return num

    def keep_earliest_date_each_subject(self):
        "dates are strings in the format of yyyy-mm-dd"
        self.dataframe["date"] = pandas.to_datetime(self.dataframe["date"])
        idx = self.dataframe.groupby(["subject_id"])["date"].idxmin()
        self.dataframe = self.dataframe.loc[idx]


if __name__ == "__main__":
    markdown = Markdown(Path("assets", "inspect.md"))
    all_variables = markdown.content.values()
    all_variables = [item for sublist in all_variables for item in sublist]
    all_variables = [Variable(**variable) for variable in all_variables]

    for i in all_variables:
        if i.whether_included and i.check_has_file():
            if not i.check_exist_in_data():
                print(i.question_title)
