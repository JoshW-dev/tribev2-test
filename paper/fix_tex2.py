"""Brute-force fix remaining dollar sign issues in main.tex."""
from pathlib import Path
import re

TEX = Path(__file__).resolve().parent / "latex" / "main.tex"
tex = TEX.read_text(encoding="utf-8")

# Fix: any $ followed by a digit that isn't already escaped
# This catches $500, $1,000, $15,000, $0.10 etc.
tex = re.sub(r'(?<!\\)\$(\d)', r'\\$\1', tex)

# Fix: $T_h$ where the first $ got escaped but shouldn't have
# Pattern: \$T_h$ -> $T_h$  (restore math that was broken)
# Actually, find \$LETTER and check if it's math
# Let's fix specific known patterns from the markdown
tex = tex.replace('\\$T_h$', '$T_h$')
tex = tex.replace('\\$T_{cta}$', '$T_{cta}$')
tex = tex.replace('\\$w$', '$w$')
tex = tex.replace('\\$\\hat{N}\\$', '$\\hat{N}$')

# Fix broken math where \$ appears inside what should be math mode
# Pattern: $...\$...$ (one escaped $ inside math)
# e.g. $\hat{N}\$ should be $\hat{N}$
tex = re.sub(r'\$([^$]+)\\\$', lambda m: '$' + m.group(1) + '$', tex)

TEX.write_text(tex, encoding="utf-8")
print("Applied brute-force fixes")

# Verify
lines = tex.split('\n')
problems = 0
for i, line in enumerate(lines, 1):
    matches = re.findall(r'(?<!\\)\$', line)
    if len(matches) % 2 != 0:
        problems += 1
        print(f"  Still broken line {i}: {line[:150]}")
print(f"Remaining problems: {problems}")
