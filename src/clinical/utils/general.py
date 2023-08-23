from ast import literal_eval


def get_dict_from_str(target_str):
    assert "{" in target_str and "}" in target_str, "The target string must contain curly braces"

    start_index = target_str.find("{")
    end_index = target_str.rfind("}")
    target_str = target_str[start_index : end_index + 1]
    target_str = str(target_str).replace("nan", "None")
    result_dict = literal_eval(target_str)

    return result_dict
