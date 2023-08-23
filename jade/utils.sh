print_blue() {
    echo -e "\033[34m$1\033[0m"
}

print_centered() {
    term_width=$(tput cols)
    padding=$(printf '%0.s-' $(seq 1 $(((term_width - ${#1}) / 2))))
    echo "${padding}${1}${padding}"
}
