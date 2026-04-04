"""Convert the paper markdown to arXiv-ready LaTeX source."""

import re
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
MD_FILE = PAPER_DIR / "neural_content_intelligence.md"
LATEX_DIR = PAPER_DIR / "latex"
TEX_FILE = LATEX_DIR / "main.tex"
BIB_FILE = LATEX_DIR / "references.bib"


def md_to_latex(md_text: str) -> tuple[str, list[dict]]:
    """Convert markdown to LaTeX body text. Returns (latex_body, references)."""

    # ── Pre-process: extract references section ──
    refs_match = re.search(r'^## (?:\d+\.\s*)?References\s*\n(.+)', md_text, re.DOTALL | re.MULTILINE)
    refs_text = refs_match.group(1) if refs_match else ""
    if refs_match:
        md_text = md_text[:refs_match.start()]

    # ── Remove YAML-style metadata ──
    md_text = re.sub(r'^---\s*$', '', md_text, flags=re.MULTILINE)

    # ── Remove title/author (handled in preamble) ──
    md_text = re.sub(r'^# .+\n', '', md_text, count=1)
    md_text = re.sub(r'^\*\*Josh W\.\*\*\s*\n', '', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^Independent Researcher\s*\n', '', md_text, flags=re.MULTILINE)

    # ── Remove abstract header (handled in preamble) ──
    md_text = re.sub(r'^## Abstract\s*\n', '', md_text, flags=re.MULTILINE)

    # ── Extract abstract ──
    abstract_match = re.search(r'^(The digital content.+?)(?=\n\n\*\*Keywords)', md_text, re.DOTALL | re.MULTILINE)
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    if abstract_match:
        # Remove abstract + keywords from body
        kw_end = md_text.find('\n---', abstract_match.start())
        if kw_end > 0:
            md_text = md_text[kw_end:]
        else:
            md_text = md_text[abstract_match.end():]

    # Remove keywords line
    md_text = re.sub(r'^\*\*Keywords:\*\*.*$', '', md_text, flags=re.MULTILINE)

    body = md_text

    # ── Convert images to figures ──
    fig_counter = [0]
    def convert_image(match):
        fig_counter[0] += 1
        alt = match.group(1)
        path = match.group(2)
        # Convert path to just filename
        fname = Path(path).name
        label = f"fig:{fname.replace('.png','').replace('.jpg','').replace('-','_')}"
        caption = alt.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
        return (
            f'\\begin{{figure}}[htbp]\n'
            f'\\centering\n'
            f'\\includegraphics[width=\\textwidth]{{figures/{fname}}}\n'
            f'\\caption{{{caption}}}\n'
            f'\\label{{{label}}}\n'
            f'\\end{{figure}}'
        )
    body = re.sub(r'!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif))\)', convert_image, body)

    # ── Remove figure caption italics that follow the image (already in \caption) ──
    body = re.sub(r'\n\*Figure \d+[a-z]?:.+?\*\n', '\n', body)
    body = re.sub(r'\n\*Supplementary figure.+?\*\n', '\n', body)

    # ── Convert headers ──
    body = re.sub(r'^#### (.+)$', r'\\paragraph{\1}', body, flags=re.MULTILINE)
    body = re.sub(r'^### (\d+\.\d+ .+)$', r'\\subsection{\1}', body, flags=re.MULTILINE)
    body = re.sub(r'^### (.+)$', r'\\subsection*{\1}', body, flags=re.MULTILINE)
    body = re.sub(r'^## (\d+\. .+)$', r'\\section{\1}', body, flags=re.MULTILINE)
    body = re.sub(r'^## (.+)$', r'\\section*{\1}', body, flags=re.MULTILINE)

    # ── Escape special LaTeX characters in body text ──
    # Do this BEFORE other conversions, but protect $math$ and existing LaTeX
    def escape_latex_chars(text):
        """Escape special LaTeX characters, preserving real math mode."""
        # Step 1: Protect real math blocks
        math_phs = []
        def save_math(m):
            math_phs.append(m.group(0))
            return f"LATEXMATH{len(math_phs)-1}ENDMATH"
        # Block math $$...$$
        text = re.sub(r'\$\$(.+?)\$\$', save_math, text, flags=re.DOTALL)
        # Inline math: $...$ containing LaTeX commands (\frac, \cdot, etc)
        text = re.sub(r'\$([^$]*\\[a-zA-Z]+[^$]*)\$', save_math, text)

        # Step 2: Escape ALL remaining $ (they're currency, not math)
        text = text.replace('$', r'\$')

        # Step 3: Escape other special chars
        text = text.replace('%', r'\%')
        text = text.replace('&', r'\&')
        text = text.replace('#', r'\#')

        # Step 4: Restore real math
        for i, m in enumerate(math_phs):
            text = text.replace(f"LATEXMATH{i}ENDMATH", m)
        return text

    body = escape_latex_chars(body)

    # ── Apply copywriting guidelines: remove em dashes ──
    body = body.replace('---', ',')  # triple dash (md em dash)
    body = body.replace('\u2014', ',')  # unicode em dash
    body = body.replace('\u2013', '--')  # en dash stays as LaTeX --
    body = body.replace(' -- ', ', ')  # spaced double dash

    # ── Convert bold and italic ──
    body = re.sub(r'\*\*\*(.+?)\*\*\*', r'\\textbf{\\textit{\1}}', body)
    body = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', body)
    # Italic: only single * not inside a list or bold
    body = re.sub(r'(?<![*\\])(?<!\n)\*([^*\n]+?)\*(?!\*)', r'\\textit{\1}', body)

    # ── Convert inline code ──
    body = re.sub(r'`([^`]+)`', r'\\texttt{\1}', body)

    # ── Convert blockquotes ──
    def convert_blockquote(match):
        text = match.group(1).strip()
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
        return f'\\begin{{quote}}\n{text}\n\\end{{quote}}'
    body = re.sub(r'((?:^>.*\n?)+)', convert_blockquote, body, flags=re.MULTILINE)

    # ── Convert markdown tables ──
    def convert_table(match):
        lines = match.group(0).strip().split('\n')
        # Filter out separator lines
        data_lines = [l for l in lines if not re.match(r'^\|[\s\-:|]+\|$', l)]
        if not data_lines:
            return match.group(0)

        # Parse cells
        rows = []
        for line in data_lines:
            cells = [c.strip() for c in line.strip('|').split('|')]
            rows.append(cells)

        if not rows:
            return match.group(0)

        ncols = len(rows[0])
        # Use p{} columns for wrapping: first col narrow, last col wide
        if ncols == 2:
            col_spec = '|p{0.3\\textwidth}|p{0.6\\textwidth}|'
        elif ncols == 3:
            col_spec = '|p{0.2\\textwidth}|p{0.2\\textwidth}|p{0.5\\textwidth}|'
        elif ncols >= 4:
            # First cols narrow, last col gets remaining space
            narrow = ncols - 1
            col_spec = '|' + '|'.join([f'p{{{0.7/narrow:.2f}\\textwidth}}'] * narrow) + f'|p{{0.25\\textwidth}}|'
        else:
            col_spec = '|l|'

        tex = f'\\begin{{table}}[htbp]\n\\centering\n\\small\n\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}\n\\hline\n' if False else \
              f'\\begin{{table}}[htbp]\n\\centering\n\\small\n\\begin{{tabular}}{{{col_spec}}}\n\\hline\n'
        for i, row in enumerate(rows):
            # Escape special chars in cells
            escaped = []
            for cell in row:
                cell = cell.replace('&', r'\&').replace('%', r'\%')
                cell = cell.replace('_', r'\_').replace('#', r'\#')
                cell = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', cell)
                cell = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\textit{\1}', cell)
                escaped.append(cell)
            tex += ' & '.join(escaped) + ' \\\\\n'
            if i == 0:
                tex += '\\hline\n'
        tex += '\\hline\n\\end{tabular}\n\\end{table}'
        return tex

    body = re.sub(r'(?:^\|.+\|\n)+', convert_table, body, flags=re.MULTILINE)

    # ── Convert bullet lists ──
    def convert_list(match):
        items = match.group(0).strip().split('\n')
        tex = '\\begin{itemize}\n'
        for item in items:
            item = re.sub(r'^\s*[-*]\s+', '', item)
            if item.strip():
                tex += f'  \\item {item}\n'
        tex += '\\end{itemize}'
        return tex
    body = re.sub(r'((?:^\s*[-*] .+\n?)+)', convert_list, body, flags=re.MULTILINE)

    # ── Convert numbered lists ──
    def convert_enum(match):
        items = match.group(0).strip().split('\n')
        tex = '\\begin{enumerate}\n'
        for item in items:
            item = re.sub(r'^\s*\d+\.\s+', '', item)
            if item.strip():
                tex += f'  \\item {item}\n'
        tex += '\\end{enumerate}'
        return tex
    body = re.sub(r'((?:^\s*\d+\. .+\n?)+)', convert_enum, body, flags=re.MULTILINE)

    # ── Convert links ──
    body = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\\href{\2}{\1}', body)

    # ── Convert horizontal rules ──
    body = re.sub(r'^---+\s*$', r'\\bigskip\\hrule\\bigskip', body, flags=re.MULTILINE)

    # ── Fix LaTeX math (keep $...$ and $$...$$ as-is) ──
    # Already valid LaTeX

    # ── Escape remaining special chars (careful not to break existing LaTeX) ──
    # Only escape & in plain text, not in tables or commands
    # This is tricky, so we skip aggressive escaping

    # ── Clean up multiple blank lines ──
    body = re.sub(r'\n{4,}', '\n\n\n', body)

    return abstract, body, refs_text


def build_bibtex(refs_text: str) -> str:
    """Build a .bib file from the references section."""
    bib = ""
    # Parse numbered references like [1] Author, Title, ...
    entries = re.findall(r'\[(\d+)\]\s*(.+?)(?=\n\[|\n\n|\Z)', refs_text, re.DOTALL)

    for num, text in entries:
        text = text.strip().replace('\n', ' ')
        # Try to extract URL
        url_match = re.search(r'https?://\S+', text)
        url = url_match.group(0).rstrip('.,)') if url_match else ""

        # Clean text for title
        title = text[:120].strip().rstrip('.')

        bib += f"""@misc{{ref{num},
  title = {{{title}}},
  note = {{[{num}]}},
  howpublished = {{\\url{{{url}}}}},
  year = {{2024}}
}}

"""
    return bib


def build():
    md_text = MD_FILE.read_text(encoding="utf-8")
    abstract, body, refs_text = md_to_latex(md_text)

    # Build bibliography
    bib_content = build_bibtex(refs_text)
    BIB_FILE.write_text(bib_content, encoding="utf-8")
    print(f"Bibliography: {BIB_FILE}")

    # Build the full .tex file
    tex = r"""\documentclass[11pt,letterpaper]{article}

% ── Packages ──
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}           % Times font
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{float}
\usepackage{caption}
\usepackage{natbib}
\usepackage{setspace}

% ── Settings ──
\singlespacing
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}
\captionsetup{font=small,labelfont=bf}

% ── Title ──
\title{Neural Content Intelligence: Using Brain Encoding Models\\to Predict Social Media Engagement Before Publication}
\author{Josh W.\\Independent Researcher}
\date{}

\begin{document}

\maketitle

% ── Abstract ──
\begin{abstract}
""" + abstract + r"""
\end{abstract}

\noindent\textbf{Keywords:} brain encoding models, neuromarketing, content optimization, fMRI prediction, social media engagement, TRIBE v2, Yeo parcellation, attention networks

\bigskip

% ── Body ──
""" + body + r"""

% ── References ──
\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
"""

    TEX_FILE.write_text(tex, encoding="utf-8")
    print(f"LaTeX source: {TEX_FILE}")
    print(f"Figures dir: {LATEX_DIR / 'figures'} ({len(list((LATEX_DIR / 'figures').glob('*')))} files)")
    print(f"\nTo compile locally: cd latex && pdflatex main && bibtex main && pdflatex main && pdflatex main")
    print(f"Or upload the latex/ folder as a .zip to arXiv.")


if __name__ == "__main__":
    build()
