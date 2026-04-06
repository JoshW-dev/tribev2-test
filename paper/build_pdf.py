"""Convert the paper markdown to an arXiv-compliant PDF using Playwright (headless Chromium)."""

import base64
import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

PAPER_DIR = Path(__file__).resolve().parent
MD_FILE = PAPER_DIR / "neural_content_intelligence.md"
OUT_PDF = PAPER_DIR / "neural_content_intelligence.pdf"
OUT_HTML = PAPER_DIR / "neural_content_intelligence.html"


def build():
    md_text = MD_FILE.read_text(encoding="utf-8")

    # Replace markdown images with base64 inline HTML
    def replace_img(match):
        alt = match.group(1)
        rel_path = match.group(2)
        img_path = PAPER_DIR / rel_path
        if img_path.exists():
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            ext = img_path.suffix.lower().lstrip(".")
            mime = {"png": "image/png", "jpg": "image/jpeg",
                    "jpeg": "image/jpeg"}.get(ext, "image/png")
            return (
                f'<div class="figure">'
                f'<img src="data:{mime};base64,{b64}" alt="{alt}">'
                f'<p class="caption"><em>{alt}</em></p>'
                f'</div>'
            )
        return match.group(0)

    md_text = re.sub(
        r'!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif))\)',
        replace_img, md_text
    )

    # Convert LaTeX math to MathML (server-side)
    import latex2mathml.converter

    math_blocks = []
    def save_math(match):
        math_blocks.append(match.group(0))
        return f"MATHPLACEHOLDER{len(math_blocks)-1}END"

    def save_inline_math(match):
        """Only treat as math if content contains LaTeX-y characters,
        not a currency amount like $500 or $1,000."""
        inner = match.group(1)
        # LaTeX indicators: backslash, underscore, caret, braces, or short single-letter vars
        is_latex = bool(re.search(r'[\\_^{}]', inner)) or (
            len(inner) <= 4 and not re.fullmatch(r'[\d.,\s]+', inner)
        )
        # Currency-like content (digits, commas, dots, dashes only) is NOT math
        is_currency = bool(re.fullmatch(r'[\d.,\-\s]+', inner))
        if is_latex and not is_currency:
            return save_math(match)
        return match.group(0)  # leave untouched

    md_text = re.sub(r'\$\$(.+?)\$\$', save_math, md_text, flags=re.DOTALL)
    md_text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', save_inline_math, md_text)

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "footnotes"],
    )

    # Restore math as rendered MathML
    for i, math_str in enumerate(math_blocks):
        is_block = math_str.startswith("$$")
        latex = math_str.strip("$").strip()
        try:
            mathml = latex2mathml.converter.convert(latex)
            if is_block:
                mathml = mathml.replace('display="inline"', 'display="block"')
                repl = f'<div style="text-align:center;margin:0.8em 0;">{mathml}</div>'
            else:
                repl = mathml
        except Exception:
            repl = f'<code>{latex}</code>'
        html_body = html_body.replace(f"MATHPLACEHOLDER{i}END", repl)

    # arXiv-compliant: single-spaced, 12pt serif, 1in margins
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: letter;
    margin: 1in;
}}
body {{
    font-family: "Times New Roman", Georgia, serif;
    font-size: 12pt;
    line-height: 1.4;
    color: #000;
}}
h1 {{
    font-size: 18pt;
    text-align: center;
    margin-bottom: 0.2em;
}}
h1 + p {{ text-align: center; }}
h1 + p + p {{ text-align: center; font-style: italic; }}
h2 {{
    font-size: 14pt;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}}
h3 {{
    font-size: 12pt;
    font-style: italic;
    margin-top: 1.2em;
}}
p {{ margin: 0.6em 0; text-align: justify; }}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 10pt;
}}
th, td {{
    border: 1px solid #555;
    padding: 5px 8px;
    text-align: left;
}}
th {{ background: #eee; font-weight: bold; }}
blockquote {{
    border-left: 3px solid #666;
    margin: 1em 0;
    padding: 0.4em 1em;
    font-size: 11pt;
    color: #333;
}}
code {{
    background: #f0f0f0;
    padding: 1px 4px;
    font-size: 10pt;
}}
pre {{
    background: #f5f5f5;
    padding: 0.8em;
    font-size: 10pt;
    overflow-x: auto;
}}
.figure {{
    text-align: center;
    margin: 1.5em auto;
    page-break-inside: avoid;
}}
.figure img {{
    max-width: 100%;
    max-height: 7in;
}}
.figure .caption {{
    font-size: 10pt;
    color: #333;
    margin-top: 0.4em;
    max-width: 90%;
    margin-left: auto;
    margin-right: auto;
}}
hr {{ border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    # Save HTML for debugging
    OUT_HTML.write_text(full_html, encoding="utf-8")
    print(f"HTML saved: {OUT_HTML}")

    # Generate PDF with Playwright headless Chromium
    print("Generating PDF with Chromium...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(full_html, wait_until="networkidle")
        page.pdf(
            path=str(OUT_PDF),
            format="Letter",
            margin={"top": "1in", "bottom": "1in", "left": "1in", "right": "1in"},
            print_background=True,
        )
        browser.close()

    size_mb = OUT_PDF.stat().st_size / 1024 / 1024
    print(f"PDF saved: {OUT_PDF} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    build()
