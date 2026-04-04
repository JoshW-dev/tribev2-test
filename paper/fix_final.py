"""Fix the last remaining dollar sign issue on line 263."""
from pathlib import Path

TEX = Path(__file__).resolve().parent / "latex" / "main.tex"
tex = TEX.read_text(encoding="utf-8")

# Direct string replace for the known broken pattern
tex = tex.replace(
    r"$w_1 = 0.35$, \$w_2 = 0.25$, \$w_3 = 0.25$, \$w_4 = 0.15$",
    r"$w_1 = 0.35$, $w_2 = 0.25$, $w_3 = 0.25$, $w_4 = 0.15$"
)

TEX.write_text(tex, encoding="utf-8")
print("Fixed line 263")
