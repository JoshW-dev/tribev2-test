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

    # ── Escape special LaTeX characters, preserving math mode ──
    def escape_latex_chars(text):
        """Escape LaTeX specials. Preserves $$..$$ display math and $..$
        inline math (detected as balanced pairs on the same line)."""
        # Stash $$...$$ display math
        math_phs = []
        def stash(m):
            math_phs.append(m.group(0))
            return f"\x00MATH{len(math_phs)-1}\x00"
        text = re.sub(r'\$\$.+?\$\$', stash, text, flags=re.DOTALL)
        # Stash inline math: $X$ where X begins with letter or backslash
        # (excludes currency like $15,000 which begins with a digit).
        text = re.sub(r'\$([a-zA-Z\\][^$\n]*?)\$', lambda m: stash(m), text)
        # Stash \includegraphics{...} file paths (underscores are valid
        # in filenames; escaping would break the path).
        text = re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}',
                      stash, text)
        # Stash figure labels so \_ doesn't leak into them
        text = re.sub(r'\\label\{[^}]*\}', stash, text)
        # At this point, any remaining $ is currency — escape it.
        text = text.replace('$', r'\$')
        # Other specials
        text = text.replace('%', r'\%')
        text = text.replace('&', r'\&')
        text = text.replace('#', r'\#')
        text = text.replace('_', r'\_')
        # Restore math blocks
        for i, m in enumerate(math_phs):
            text = text.replace(f"\x00MATH{i}\x00", m)
        return text

    body = escape_latex_chars(body)
    abstract = escape_latex_chars(abstract)

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
        # Filter out separator lines. At this point em-dashes have been
        # replaced by commas, so separator rows look like |,|,|,| too.
        def is_sep(l):
            if re.match(r'^\|[\s\-:|]+\|$', l):
                return True
            inner = l.strip().strip('|')
            cells = [c.strip() for c in inner.split('|')]
            return bool(cells) and all(c in ('', ',', '-', '--') for c in cells)
        data_lines = [l for l in lines if not is_sep(l)]
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
            # Specials already escaped by escape_latex_chars — don't
            # double-escape. Just handle inline markdown here.
            processed = []
            for cell in row:
                cell = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', cell)
                cell = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\textit{\1}', cell)
                processed.append(cell)
            tex += ' & '.join(processed) + ' \\\\\n'
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


def _escape_ref_text(text: str) -> str:
    """Escape LaTeX specials in reference text, leaving URLs alone."""
    # Pull URLs out so we can wrap them in \url{}
    urls = []
    def stash_url(m):
        urls.append(m.group(0).rstrip('.,);'))
        return f"URLPH{len(urls)-1}END"
    text = re.sub(r'https?://\S+', stash_url, text)

    text = text.replace('\\', r'\textbackslash{}')
    text = text.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
    text = text.replace('_', r'\_').replace('{', r'\{').replace('}', r'\}')
    text = text.replace('$', r'\$').replace('~', r'\textasciitilde{}')
    text = text.replace('^', r'\textasciicircum{}')

    # Convert *italic* markers
    text = re.sub(r'\*([^*]+?)\*', r'\\textit{\1}', text)

    # Restore URLs
    for i, u in enumerate(urls):
        text = text.replace(f"URLPH{i}END", f"\\url{{{u}}}")
    return text


def build_references_section(refs_text: str) -> str:
    """Build a plain LaTeX \\section*{References} with a formatted list."""
    entries = re.findall(
        r'\[(R?\d+)\]\s*(.+?)(?=\n\[R?\d+\]|\n\*\*Additional|\n---|\Z)',
        refs_text, re.DOTALL,
    )
    if not entries:
        return ""

    # Split into main refs and "additional foundational" refs
    main_refs = [(n, t) for n, t in entries if not n.startswith('R')]
    extra_refs = [(n, t) for n, t in entries if n.startswith('R')]

    def render(items):
        s = "\\begingroup\n\\small\n"
        s += ("\\begin{list}{}{\\setlength{\\leftmargin}{2em}"
              "\\setlength{\\itemindent}{-2em}\\setlength{\\itemsep}{2pt}}\n")
        for num, text in items:
            text = re.sub(r'\s+', ' ', text.strip())
            s += f"\\item[{{[{num}]}}] {_escape_ref_text(text)}\n"
        s += "\\end{list}\n\\endgroup\n"
        return s

    out = "\\section*{References}\n" + render(main_refs)
    if extra_refs:
        out += "\n\\subsection*{Additional foundational references}\n" + render(extra_refs)
    return out


def build():
    md_text = MD_FILE.read_text(encoding="utf-8")
    abstract, body, refs_text = md_to_latex(md_text)

    # Build references as a plain LaTeX section (paper uses [N] not \cite{})
    refs_section = build_references_section(refs_text)
    if BIB_FILE.exists():
        BIB_FILE.unlink()

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
\usepackage[hyphens]{url}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{float}
\usepackage{caption}
\usepackage{setspace}

% ── Settings ──
\singlespacing
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue,
    breaklinks=true
}
\Urlmuskip=0mu plus 1mu\relax
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
""" + body + "\n\n" + refs_section + r"""

\end{document}
"""

    TEX_FILE.write_text(tex, encoding="utf-8")
    print(f"LaTeX source: {TEX_FILE}")
    print(f"Figures dir: {LATEX_DIR / 'figures'} ({len(list((LATEX_DIR / 'figures').glob('*')))} files)")
    print(f"\nTo compile locally: cd latex && pdflatex main && bibtex main && pdflatex main && pdflatex main")
    print(f"Or upload the latex/ folder as a .zip to arXiv.")


if __name__ == "__main__":
    build()
