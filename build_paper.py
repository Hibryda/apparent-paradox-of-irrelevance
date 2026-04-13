"""Convert docs/paper.md to output/document.tex for arxiv-classic template.

Strategy: read markdown, convert to LaTeX with careful math handling.
All math expressions in the markdown already use $...$ or $$...$$ delimiters.
Text outside math needs: underscore escaping, special char escaping, bold/italic.
"""
import re
import hashlib
from pathlib import Path

PAPER = Path("docs/paper.md")
OUTPUT = Path("output/document.tex")

# Abstract from md2pdf.toml (already LaTeX-safe, just needs math wrapping)
ABSTRACT = r"""Under normalized pairwise similarity, k-core outperforms degree for link prediction (mean AUC 0.659 vs 0.491) across 65 networks from 20 domains and 4 metrics---an apparent paradox since degree carries more raw predictive signal. We resolve this in three steps. First, we prove a ceiling effect: normalized similarity metrics compress variance at rate $O(1/\mu^2)$ (Z3-verified: 5 lemmas, 3 theorems; empirically confirmed in 29/30 networks). Second, we show this ceiling is real but not the primary mechanism---variance does not predict AUC ($\rho = 0.099$, $p$ = ns). The correct predictor is signed Fisher's $d'$ (pooled $\rho = 0.994$, $n = 130$ feature-network pairs). Third, we derive an exact mixture AUC decomposition: $\text{AUC} = p_e(1 - p_{ne}) + 0.5\,p_e\,p_{ne} + (1 - p_e)(1 - p_{ne})\,\text{AUC}_{\text{continuous}}$. For k-core, ties account for 59\% of AUC; the continuous gradient contributes 41\%. The resolution is structural: degree is disassortative (tie enrichment ratio $0.54\times$), while k-core is assortative (TER = $3.58\times$). Assortativity gates the effect (determines direction); $d'$ determines magnitude. K-core similarity wins in all 65 networks (100\%) across all metrics. We also show metric independence---three continuous normalized similarity formulas yield identical AUC on integer-valued features (cross-metric $\rho = 1.000$)---and provide a positional signal diagnostic (Edge-Driven Graph Equivalence, EDGE) that detects excess centrality similarity in 10/20 biological networks versus 3/45 non-biological (Fisher exact $p = 0.0002$, one-sided). All formal results are verified by the Z3 SMT solver."""


def protect_and_convert(text: str) -> str:
    """Convert a line of markdown to LaTeX, protecting math spans."""
    # Step 1: Protect existing $...$ math spans
    math_spans = []
    def save_math(m):
        math_spans.append(m.group(0))
        return f'\x00MATH{len(math_spans)-1}\x00'
    text = re.sub(r'\$[^$]+?\$', save_math, text)

    # Step 2: Bold **text** → \textbf{text}
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    # Italic *text* → \textit{text}
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\textit{\1}', text)
    # Inline code `text` → \texttt{text}
    text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', text)

    # Step 3: Escape special chars in non-math text
    text = text.replace('&', '\\&')
    text = text.replace('%', '\\%')
    text = text.replace('_', '\\_')
    text = text.replace('^', '\\^{}')
    # Don't escape # (already handled by section conversion)

    # Step 4: Common inline math patterns (wrap in $...$)
    # rho = N.NNN, sigma/mu, etc (standalone Greek letters with = or / context)
    text = re.sub(r'\brho\b', r'$\\rho$', text)
    text = re.sub(r'\bsigma\b', r'$\\sigma$', text)
    # mu as standalone word (not inside other words)
    text = re.sub(r'(?<![a-zA-Z])\bmu\b(?![a-zA-Z])', r'$\\mu$', text)

    # Step 5: Restore math spans
    for i, span in enumerate(math_spans):
        text = text.replace(f'\x00MATH{i}\x00', span)

    return text


def md_to_latex(md: str) -> str:
    """Convert paper markdown body to LaTeX."""
    lines = md.split('\n')
    out = []
    in_equation = False
    in_list = False
    skip_front = True

    for line in lines:
        # Skip frontmatter (title, authors, abstract)
        if skip_front:
            if re.match(r'^## 1', line.strip()):
                skip_front = False
            else:
                continue

        stripped = line.strip()

        # Horizontal rules
        if stripped == '---':
            out.append('')
            continue

        # Display equations: $$...$$
        if stripped.startswith('$$'):
            inner = stripped[2:]
            close = inner.find('$$')
            if close >= 0:
                # Single-line equation
                eq = inner[:close].strip()
                after = inner[close+2:].strip()
                label = re.match(r'\((\d+)\)', after)
                if label:
                    out.append('\\begin{equation}')
                    out.append(f'  {eq}')
                    out.append(f'  \\label{{eq:{label.group(1)}}}')
                    out.append('\\end{equation}')
                else:
                    out.append(f'\\[{eq}\\]')
            elif not in_equation:
                in_equation = True
                out.append('\\begin{equation}')
            else:
                in_equation = False
                out.append('\\end{equation}')
            continue

        if in_equation:
            out.append(f'  {line.strip()}')
            continue

        # Close list if needed
        if in_list and (stripped == '' or not re.match(r'^\d+\.', stripped)):
            out.append('\\end{enumerate}')
            in_list = False

        # References section
        if stripped == '## References':
            out.append('\\section*{References}')
            out.append('\\small')
            continue

        # Sections
        m = re.match(r'^## (\d+)\.?\s+(.+)$', stripped)
        if m:
            title = protect_and_convert(m.group(2))
            out.append(f'\\section{{{title}}}')
            continue

        # Subsections
        m = re.match(r'^### (\d+\.\d+)\s+(.+)$', stripped)
        if m:
            title = protect_and_convert(m.group(2))
            out.append(f'\\subsection{{{title}}}')
            continue

        # Numbered list
        m = re.match(r'^(\d+)\.\s+(.+)$', stripped)
        if m and not stripped.startswith('**'):
            if not in_list:
                out.append('\\begin{enumerate}')
                in_list = True
            out.append(f'  \\item {protect_and_convert(m.group(2))}')
            continue

        # Empty line
        if stripped == '':
            out.append('')
            continue

        # Figure placeholders → actual figures if PNG exists
        if stripped.startswith('[Figure'):
            fig_text = stripped[1:-1] if stripped.endswith(']') else stripped
            # Extract figure number
            fig_num_match = re.match(r'Figure (\d+)', fig_text)
            if fig_num_match:
                fig_num = fig_num_match.group(1)
                fig_files = {
                    '1': 'figures/fig1_bl2_scatter.png',
                    '2': 'figures/fig2_mixture_auc.png',
                    '3': 'figures/fig3_dprime_vs_auc.png',
                    '4': 'figures/fig4_metric_independence.png',
                    '5': 'figures/fig5_cumulative_validation.png',
                }
                fig_path = fig_files.get(fig_num)
                fig_widths = {'1': '0.7', '2': '1.0', '3': '0.7', '4': '1.0', '5': '0.7'}
                if fig_path and (Path('output') / fig_path).exists():
                    # Extract caption (everything after "Figure N: ")
                    caption = re.sub(r'^Figure \d+:\s*', '', fig_text)
                    caption = protect_and_convert(caption)
                    # Path relative to project root (lualatex runs from there)
                    full_fig_path = f'output/{fig_path}'
                    width = fig_widths.get(fig_num, '1.0')
                    out.append('\\begin{figure}[htbp]')
                    out.append('\\centering')
                    out.append(f'\\includegraphics[width={width}\\textwidth]{{{full_fig_path}}}')
                    out.append(f'\\caption{{{caption}}}')
                    out.append(f'\\label{{fig:{fig_num}}}')
                    out.append('\\end{figure}')
                    continue
            # Fallback: italic placeholder
            out.append(f'\\medskip\\noindent\\textit{{{protect_and_convert(fig_text)}}}\\medskip')
            continue

        # Reference entries [N] ...
        m = re.match(r'^\[(\d+)\]\s+(.+)$', stripped)
        if m:
            out.append(f'\\noindent [{m.group(1)}] {protect_and_convert(m.group(2))}')
            out.append('')
            continue

        # Regular text
        out.append(protect_and_convert(stripped))

    if in_list:
        out.append('\\end{enumerate}')

    return '\n'.join(out)


TITLE = "The Apparent Paradox of Irrelevance: Fisher Discriminability Explains Feature Performance in Normalized Similarity Link Prediction"
VERSION = "v1.0"


def compute_document_hash() -> str:
    """SHA512 hash of paper body — used in footer watermark. Uppercase hex."""
    body = PAPER.read_text()
    return hashlib.sha512(body.encode()).hexdigest().upper()


def write_variables_tex(doc_hash: str):
    """Write resolved variables.tex with computed hash."""
    variables = Path("output/setup/initial-setup/variables.tex")
    variables.write_text(f"""% variables.tex — auto-generated by build_paper.py

\\newcommand{{\\doctitle}}{{{TITLE}}}
\\newcommand{{\\docsubtitle}}{{}}
\\newcommand{{\\docsubject}}{{Network Science, Link Prediction}}
\\newcommand{{\\docauthor}}{{Hibryda}}
\\newcommand{{\\doccreator}}{{LuaLaTeX/arxiv-classic}}
\\newcommand{{\\docversion}}{{{VERSION}}}
\\newcommand{{\\docversionlabel}}{{{VERSION}}}
\\newcommand{{\\documenthash}}{{{doc_hash}}}
""")


def main():
    md = PAPER.read_text()
    body = md_to_latex(md)

    # Compute and write document hash
    doc_hash = compute_document_hash()
    write_variables_tex(doc_hash)
    print(f"Document hash: {doc_hash[:32]}...")

    doc = f"""%% document.tex — The Apparent Paradox of Irrelevance
%% Built from docs/paper.md via build_paper.py
%% Compile: lualatex -output-directory=output output/document.tex (x2)

\\documentclass[10pt,letterpaper]{{class/arxiv-classic}}

%% ── Setup ─────────────────────────────────────────────────
\\input{{setup/initial-setup/packages.tex}}
\\input{{setup/initial-setup/fonts.tex}}
\\input{{setup/initial-setup/colors.tex}}
\\input{{setup/initial-setup/variables.tex}}
\\input{{setup/initial-setup/footer.tex}}

%% ── Paths ─────────────────────────────────────────────────
\\graphicspath{{{{content/images/}}{{./}}}}

%% ── Style overrides ───────────────────────────────────────
\\input{{styles/packages.sty}}
\\input{{styles/listings.sty}}

%% ── PDF metadata ──────────────────────────────────────────
\\hypersetup{{
    pdftitle    = {{The Apparent Paradox of Irrelevance}},
    pdfauthor   = {{Hibryda}},
    pdfsubject  = {{Network Science, Link Prediction}},
    pdfcreator  = {{LuaLaTeX/arxiv-classic}},
}}

%% ── Title ─────────────────────────────────────────────────
\\title{{The Apparent Paradox of Irrelevance:\\\\Fisher Discriminability Explains Feature Performance\\\\in Normalized Similarity Link Prediction}}
\\author{{Hibryda\\\\\\texttt{{hibryda@protonmail.com}}\\\\Independent Researcher}}
\\date{{}}

\\begin{{document}}

\\maketitle
\\thispagestyle{{plain}}

%% ── Abstract ──────────────────────────────────────────────
\\begin{{abstract}}
{ABSTRACT}
\\end{{abstract}}

%% ── Hash footer ───────────────────────────────────────────
\\footerbar%

%% ── Content ───────────────────────────────────────────────
{body}

\\end{{document}}
"""
    OUTPUT.write_text(doc)
    print(f"Written {OUTPUT} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
