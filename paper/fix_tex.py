"""Post-process main.tex to fix unescaped dollar signs and em dashes."""
import re
from pathlib import Path

TEX = Path(__file__).resolve().parent / "latex" / "main.tex"
tex = TEX.read_text(encoding="utf-8")

# Strategy: protect real math blocks, escape everything else, restore math

# Step 1: Pull out real math environments
math_phs = []
def save(m):
    math_phs.append(m.group(0))
    return f"SAFEMATH{len(math_phs)-1}END"

# Block math: $$...$$ and \[...\]
tex = re.sub(r'\$\$(.+?)\$\$', save, tex, flags=re.DOTALL)
tex = re.sub(r'\\\[(.+?)\\\]', save, tex, flags=re.DOTALL)
# Inline math: $...\command...$  (real math has backslash commands inside)
tex = re.sub(r'\$([^$]*\\[a-zA-Z]+[^$]*)\$', save, tex)

# Step 2: Escape ALL remaining unescaped $
tex = re.sub(r'(?<!\\)\$', r'\\$', tex)

# Step 3: Fix double-escaping (\\\$ -> \$)
tex = tex.replace('\\\\$', '\\$')

# Step 4: Restore real math
for i, m in enumerate(math_phs):
    tex = tex.replace(f"SAFEMATH{i}END", m)

# Step 5: Remove em dashes (replace with commas or colons)
tex = tex.replace('\u2014', ', ')  # unicode em dash
tex = tex.replace('\u2013', '--')  # en dash -> LaTeX en dash

# Step 6: Fix N_v(t) style text that should be in math mode
# These are variable names in prose that need $...$
tex = re.sub(r'(?<!\$)N_([a-z]+)\(t\)(?!\$)', r'$N_{\1}(t)$', tex)

TEX.write_text(tex, encoding="utf-8")
print(f"Fixed {TEX}")

# Verify: count remaining unescaped $
lines = tex.split('\n')
problems = 0
for i, line in enumerate(lines, 1):
    # Count $ not preceded by \
    matches = re.findall(r'(?<!\\)\$', line)
    if len(matches) % 2 != 0:
        problems += 1
        if problems <= 5:
            print(f"  Warning line {i}: {len(matches)} unescaped $: {line[:100]}")
print(f"Lines with odd $ count: {problems}")
