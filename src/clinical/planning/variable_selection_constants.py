import pandas
from pathlib import Path
from natsort import natsorted

PATH_SHEETS = Path("data", "variable_sheets")
PATH_SHEETS_ORIGINAL = PATH_SHEETS / "original.xlsx"

# --------------------read in the sheets--------------------------------------
FRAME_FORM_DETAILS = pandas.read_excel(PATH_SHEETS_ORIGINAL, sheet_name="FormDetails")
FRAME_FORM_DETAILS.columns = FRAME_FORM_DETAILS.iloc[0]
FRAME_FORM_DETAILS.rename(columns={FRAME_FORM_DETAILS.columns[0]: "index"}, inplace=True)
FRAME_FORM_DETAILS = FRAME_FORM_DETAILS[(FRAME_FORM_DETAILS["index"].str.startswith("Form")) | (FRAME_FORM_DETAILS["index"].isna())]
FRAME_FORM_DETAILS["Desciprtion (optional)"].fillna("", inplace=True)

rename_dict = {
    "Name (nospaces or _)": "name",
    "Desciprtion (optional)": "description",
    "index": "index",
    "Extension or level (subject/image)": "level",
    "Form Title": "title",
}

for key, value in rename_dict.items():
    FRAME_FORM_DETAILS.rename(columns={key: value}, inplace=True)

# -------------------create a list of form names-----------------------------
SHEET_NAMES = FRAME_FORM_DETAILS["index"].unique()
SHEET_NAMES = natsorted(SHEET_NAMES)


# -------------------create a dict of form details-----------------------------
DICT_FORM_DETAILS = FRAME_FORM_DETAILS.to_dict(orient="records")

# -------------------create a dict of column names and errors-----------------------------
COLUMN_NAME_ERRORS = {
    "question_title": ["Question title:"],
    "data_name": ["Data Name (no spaces):", "Data Name (no spaces or underscores _ ):", "Data Name (no spaces or underscores _ ):"],
    "subname": ["Subname (optional -no spaces or underscores _ ):", "Subname (optional):"],
    "type": [
        "type: tickbox, textbox, radio, dropdown, yn, into, ynu (if yes/no then open subquestions, yes/no/unable for helath reasons/unable other )",
        "type: tickbox, textbox, radio, dropdown, iyto, into, ynu (if yes/no then open subquestions, yes/no/unable for helath reasons/unable other )",
        "type: tickbox, textbox, radio, dropdown",
    ],
    "data_type": [
        "Data typeype (default String)- bool, float, int. Automatically int for radio, int for dropdown, bool for tickbox",
        "Data typeype (default String)- bool float, int:",
    ],
    "length_restriction": ["Length restriction (optional):"],
    "options": ["options: if dropdown/radio. Comma sepreated", "options: if dropdown. Comma sepreated"],
}
