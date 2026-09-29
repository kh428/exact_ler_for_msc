"""Front matter of the detector-web guide: title page and detector lists."""

NAMES = {3: '$d=3$', 5: '$d=5$'}
SOURCES = {3: r'\texttt{data/inputs/clifft\_d3\_p001.stim}', 5: r'\texttt{data/inputs/soft\_cultivation\_d5\_p0005.stim}'}


def guide(keys, results, detector_table):
    summary = []
    for key in keys:
        r = results[key]
        dets = r['detectors']
        dep = [d['index'] for d in dets if d['defects']]
        a, b = f"d{key}_det{dets[0]['index']:03d}", f"d{key}_det{dets[-1]['index']:03d}"
        summary.append(rf"{NAMES[key]} & {len(r['parsed']['records'])} & {len(dets)} & {len(dets) - len(dep)} & "
                       rf"{len(dep)} & \pageref*{{{a}}}--\pageref*{{{b}}}\\")
    def listed(key):
        idx = [str(d['index']) for d in results[key]['detectors'] if d['defects']]
        return ', '.join(idx[:-1]) + ' and ' + idx[-1] + rf' at $d={key}$'
    dep_text = 'detectors ' + ', and '.join(listed(key) for key in keys)

    head = r"""\documentclass[11pt]{article}
\usepackage[a3paper,landscape,margin=12mm]{geometry}
\usepackage{amsmath,amssymb,bm,tikz,graphicx,adjustbox,booktabs,caption,longtable,hyperref}
\usetikzlibrary{calc,backgrounds}
\definecolor{highlight}{RGB}{0,0,0}
\input{drawings/preamble.tex}
\input{drawings/legend.tex}
\pdfinfoomitdate=1
\pdftrailerid{}
\pdfsuppressptexinfo=15
\hypersetup{pdftitle={Pauli webs of every declared detector},
 pdfauthor={},pdfsubject={},pdfkeywords={},pdfcreator={},pdfproducer={},
 colorlinks=true,allcolors=highlight,bookmarksopen=false}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\begin{document}
\begin{semiwebaddition}
\pdfbookmark[0]{Reading the diagrams}{guide}
{\LARGE Pauli webs of every declared detector}\par
\medskip
{\large Whole-circuit ZX diagrams of the $d=3$ and $d=5$ cultivation circuits}
\bigskip

\begin{minipage}[t]{.47\linewidth}
\textbf{Contents}

Each page after the detector lists shows one declared detector. The whole
noiseless circuit is one ZX diagram, and the page overlays a closed web whose
records are exactly that detector's records.

\medskip
\begin{tabular}{lrrrrr}
\toprule
 & Records & Declared & Pauli webs & Semiwebs with $T$ defects & Pages\\
\midrule
""" + '\n'.join(summary) + r"""
\bottomrule
\end{tabular}

\medskip
\textbf{Why 16 of the 20}

At $d=3$, 16 of the 20 declared detectors are Pauli webs, and at $d=5$, 93 of the 107.
The others, """ + dep_text + r""", are Pauli semiwebs with defects on $T$ spiders.

\medskip
\textbf{How each web was chosen}

Each detector is a closed Pauli web of the $S$ proxy, the circuit with each
$T$ replaced by $S$ and each $T^\dagger$ by $S^\dagger$. The proxy has the
same nodes and edges, so its webs can be drawn on the $T$ circuit. There they
keep every spider's phase except at $T$ spiders, where an $X$ that turns into
$Y$ is a defect of $\frac{\pi}{2}$ at $T$ and $-\frac{\pi}{2}$ at $T^\dagger$.
Two webs with the same records differ by one of $2^6=64$ record-free webs,
which certify the six qubits reset mid-injection in $|+\rangle$. The page shows
the web with the fewest defects, then the fewest labelled edges. A detector is
phase-independent exactly when that minimum is zero.

\medskip
\textbf{Sources}

Definitions and drawing styles: Kissinger and van de Wetering,
\emph{ZX-Flow}, \href{https://arxiv.org/abs/2603.09580}{arXiv:2603.09580}.
Circuits: the paper's inputs, SOFT's $d=3$ and $d=5$ cultivation circuits,
""" + SOURCES[3] + r""" and """ + SOURCES[5] + r""" in the companion code. They follow the construction of Gidney, Shutty and Jones,
\href{https://arxiv.org/abs/2409.17595}{arXiv:2409.17595}.
\end{minipage}\hfill
\begin{minipage}[t]{.49\linewidth}
\textbf{Reading a diagram}

Time runs to the right and each qubit is a row. Data qubits of the final code
have black labels, other qubits grey. Shaded bands mark the stages of the circuit.
At $d=3$ every page shows the whole circuit. At $d=5$ each page shows the columns
its web uses, and wires crossing the edge of the window are cut.

\textbf{Legend}\quad
\semiwebinput[.84\linewidth]{drawings/examples/reading_legend.tikz}

\mbox{\semiwebCaptionGate{\frac{\pi}{4}} ${}=T$ ($-\pi/4$: $T^\dagger$)};
\mbox{\semiwebCaptionGate{\frac{\pi}{2}} ${}=S$};
\mbox{\semiwebCaptionCnot ${}={\rm CNOT}$};
preparations \mbox{\semiwebCaptionState{X} ${}=|0\rangle$}
and \mbox{\semiwebCaptionState{Z} ${}=|+\rangle$}.
Measurements are effects drawn at outcome 0:
\mbox{\semiwebCaptionEffect{Z} ${}=\langle+|$ (MX)},
\mbox{\semiwebCaptionEffect{X} ${}=\langle0|$ (M)}.
Outcome $r_k$ adds a phase $r_k\pi$ to that spider, and $r_k$ is printed next to
it; the detector's own records are in bold blue. A qubit measured and then used
again without reset restarts from a second spider with the same record. A Pauli
product measurement joins one spider per qubit to a central spider carrying
$r_k\pi$; a $Y$ product has $S^\dagger$ and $S$ around it. A record-controlled
Pauli ($d=5$ corrections) is a two-legged spider of phase $r_k\pi$. A qubit reset
while in use ends in \semiwebCaptionEffect{Z}, the effect $\langle+|$: the circuit
has put it in $|+\rangle$ for every angle at the earlier $T$ gate.

\textbf{The final MPP $Y_L$}

Both circuits end with a noiseless layer of $T$ gates ($T$ and $T^\dagger$ at $d=5$)
followed by MPP $Y_L$, and because $T^\dagger YT=TXT^\dagger=(X+Y)/\sqrt2$ this
measures the same transversal operator as the last double check, a noiseless
projection onto the magic state. It is deterministic in the noiseless circuit,
by a statevector run at $d=3$ and by $a_0=1$ and $e_0=0$ in the exact series at
both distances, and it enters no declared detector.

\textbf{The identity on each page}

Write $D_{\boldsymbol r}$ for the diagram with records $\boldsymbol r$ and open
outputs, and $R$ for the records whose spiders the web meets in the opposite
colour. At every spider the labels satisfy
$w_v|v(\theta_v)\rangle=\lambda_v|v(\theta_v+\delta_v)\rangle$, and the product
over the diagram is
\[
 D_{\boldsymbol r}=\lambda(\boldsymbol r)\,D_{\boldsymbol r,\boldsymbol\delta},
 \qquad \lambda(\boldsymbol r)=\lambda_0\,(-1)^{\sum_{i\in R} r_i}.
\]
$D_{\boldsymbol r,\boldsymbol\delta}$ has the marked phases changed. With no
defects, $D_{\boldsymbol r}=\lambda(\boldsymbol r)D_{\boldsymbol r}$, so
$D_{\boldsymbol r}=0$ unless $\sum_{i\in R}r_i=b$, where $\lambda_0=(-1)^b$:
the records obey the check in every noiseless run. Every web drawn here has
$\lambda_0=1$. With defects, $D_{\boldsymbol r,\boldsymbol\delta}$ is a different
circuit, and the check needs the $T$ angles.

Numbered violet stars $\ast_n$ mark defects, listed under each diagram with the
gate, qubit and source line. The phases printed in spiders are the original
phases. Each TikZ file records every defect in a comment.
\end{minipage}
\vfill
\small The drawings suppress common ZX normalisation factors, which cancel between the two sides of each identity.
\newpage
"""
    lists = []
    for key in keys:
        r = results[key]
        rows = detector_table(key, r['parsed'], r['detectors'])
        lists.append(rf"\pdfbookmark[0]{{Detector list, d={key}}}{{list{key}}}")
        lists.append(rf"{{\Large {NAMES[key]}: all {len(rows)} declared detectors}}\par\medskip")
        size = r'\large' if len(rows) <= 30 else r'\small'
        lists += [rf'{{{size}\setlength{{\LTleft}}{{0pt}}\setlength{{\LTright}}{{\fill}}',
                  r'\begin{longtable}{rp{.3\linewidth}p{.43\linewidth}lr}', r'\toprule',
                  r'Det. & Check & Records & Drawn as & Page\\', r'\midrule', r'\endhead',
                  *rows, r'\bottomrule', r'\end{longtable}}']
        lists.append(r'\newpage')
    tail = r"""\input{detector_webs/pages.tex}
\end{semiwebaddition}
\end{document}
"""
    return head + '\n'.join(lists) + '\n' + tail
