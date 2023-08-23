def c_print(text: str, color: str):
    if color == "red":
        print(f"\033[1;31;40m {text} \033[0;37;40m")
    elif color == "green":
        print(f"\033[1;32;40m {text} \033[0;37;40m")
    elif color == "yellow":
        print(f"\033[1;33;40m {text} \033[0;37;40m")
    elif color == "blue":
        print(f"\033[1;34;40m {text} \033[0;37;40m")
    elif color == "magenta":
        print(f"\033[1;35;40m {text} \033[0;37;40m")
    elif color == "cyan":
        print(f"\033[1;36;40m {text} \033[0;37;40m")
    elif color == "white":
        print(f"\033[1;37;40m {text} \033[0;37;40m")
    else:
        print(text)


def c_paste(text: str, color: str):
    if color == "red":
        return f"\033[1;31;40m {text} \033[0;37;40m"
    elif color == "green":
        return f"\033[1;32;40m {text} \033[0;37;40m"
    elif color == "yellow":
        return f"\033[1;33;40m {text} \033[0;37;40m"
    elif color == "blue":
        return f"\033[1;34;40m {text} \033[0;37;40m"
    elif color == "magenta":
        return f"\033[1;35;40m {text} \033[0;37;40m"
    elif color == "cyan":
        return f"\033[1;36;40m {text} \033[0;37;40m"
    elif color == "white":
        return f"\033[1;37;40m {text} \033[0;37;40m"
    else:
        return text
