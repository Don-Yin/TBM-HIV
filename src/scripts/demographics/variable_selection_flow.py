from pathlib import Path
import schemdraw
import schemdraw.elements as elm
from schemdraw import flow

save_to = Path("results", "images", "demographics")
save_to.mkdir(parents=True, exist_ok=True)


steps = [
    "Initial variables",
    "Manual curation:",
    "Dropping variables\nwithout data file:",
    "Dropping variables\nnot in data:",
    "Dropping systematically\nmissing variables:",
    "Dropping variables\nwith < 50% data available",
]

variables_left = [387, 209, 88, 84, 70, 70]  # variables left at each step

with schemdraw.Drawing(facecolor="white") as d:
    for step, variable in zip(steps, variables_left):
        d += (decision := flow.Decision(w=5.5, h=4).label(step))
        d += flow.Line().right().at(decision.E).length(1)  # Reduced length
        d += (box := flow.Box(w=5.5, h=2).label(f"{variable} variables left"))
        if step != steps[-1]:
            d += flow.Line().left().at(box.W).length(1)  # Reduced length
            d += flow.Arrow().down().at(decision.S).length(0.5)  # You may also want to adjust the vertical distance

d.save(save_to / "variable_selection_flowchart.png")
