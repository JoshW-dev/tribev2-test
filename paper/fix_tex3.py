"""Final pass: fix remaining $ issues by direct string replacement."""
from pathlib import Path
import re

TEX = Path(__file__).resolve().parent / "latex" / "main.tex"
tex = TEX.read_text(encoding="utf-8")

# Fix specific remaining issues found in the warnings:

# Line 39/270: "$500 billion" -> "\$500 billion"
# Line 91: "$1,000" -> "\$1,000"
# Line 263: "\$w_2 = 0.25$" -> "$w_2 = 0.25$" (math was broken)
# Line 500: "$10--50" "$150,000" "$1.00"

# Fix all currency amounts: $ followed by digit (with possible comma/period)
tex = re.sub(r'(?<!\\)\$(\d[\d,.]*)', r'\\$\1', tex)

# Fix broken math: \$VAR = VALUE$ -> $VAR = VALUE$
# These are inline math where the opening $ got escaped
tex = re.sub(r'\\\$(\\?[a-zA-Z_{}]+\s*=\s*[^$]+?)\$', r'$\1$', tex)
tex = re.sub(r'\\\$(\\?[a-zA-Z_{}]+\(t\))\$', r'$\1$', tex)

# Fix \$w$ pattern
tex = re.sub(r'\\\$([a-zA-Z])\$', r'$\1$', tex)

TEX.write_text(tex, encoding="utf-8")

# Verify
lines = tex.split('\n')
problems = 0
for i, line in enumerate(lines, 1):
    matches = re.findall(r'(?<!\\)\$', line)
    if len(matches) % 2 != 0:
        problems += 1
        print(f"  Problem line {i}: {line[:150]}")
print(f"Remaining problems: {problems}")
if problems == 0:
    print("All dollar signs balanced!")
