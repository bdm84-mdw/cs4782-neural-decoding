"""Render the 2-page project report to group_topic_2page_report.pdf using fpdf2.

The LaTeX source in report.tex is the canonical version. This script reproduces
the same content with fpdf2 so the repo includes a submittable PDF without
requiring a local LaTeX installation. Run:

    python3 report/build_report_pdf.py
"""

from pathlib import Path

from fpdf import FPDF

HERE = Path(__file__).parent
OUT = HERE / "group_topic_2page_report.pdf"
FIG = HERE / "r2_comparison.png"


class Report(FPDF):
    def header(self):
        pass

    def footer(self):
        pass


def reset_x(pdf):
    pdf.set_x(pdf.l_margin)


def h1(pdf, text):
    reset_x(pdf)
    pdf.set_font("Helvetica", "B", 12.5)
    pdf.ln(1.5)
    pdf.cell(0, 5.5, text, ln=1)
    pdf.set_font("Helvetica", "", 10.5)


def body(pdf, text):
    reset_x(pdf)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.multi_cell(0, 4.5, text)
    pdf.ln(0.8)


def labeled(pdf, label, text):
    """Bold lead-in then regular body, as a single paragraph."""
    reset_x(pdf)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.write(4.5, label + " ")
    pdf.set_font("Helvetica", "", 10.5)
    pdf.write(4.5, text)
    pdf.ln(4.5)
    pdf.ln(0.8)


def bullet(pdf, label, text):
    reset_x(pdf)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.write(4.5, "  - " + label + " ")
    pdf.set_font("Helvetica", "", 10.5)
    pdf.write(4.5, text)
    pdf.ln(4.5)


def main():
    pdf = Report(format="Letter", unit="mm")
    pdf.set_margins(left=18, top=14, right=18)
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 13.5)
    pdf.cell(0, 6.5, "Reproducing POYO: A Unified, Scalable Framework", ln=1, align="C")
    pdf.cell(0, 6.5, "for Neural Population Decoding", ln=1, align="C")
    pdf.set_font("Helvetica", "", 10.5)
    pdf.cell(0, 4.8, "Benjamin Mun, Kaden Priebe  -  Cornell University, CS 4782 Final Project", ln=1, align="C")
    pdf.ln(1)

    h1(pdf, "1. Introduction")
    body(pdf,
        "Brain-computer interfaces decode behavior (e.g. reach kinematics) from neural population activity. "
        "A persistent failure mode is that decoders trained on one recording session generalize poorly to a "
        "different session - even on the same animal a few days later - because the recorded neuron set "
        "drifts from the array and per-neuron tuning shifts."
    )
    body(pdf,
        "Azabou et al. (2023) introduced POYO (\"Pre-training On manY neurOns\"), a transformer-based "
        "decoder that reframes the input as an unordered set of single-spike events tokenized by a learned "
        "per-neuron embedding rather than as a binned spike-count tensor. This unit-vocabulary view lets "
        "POYO transfer across sessions and animals with different neuron sets, addressing the "
        "cross-session generalization gap. We re-implement POYO and its closest non-transformer baselines "
        "on the Perich-Miller (2018) center-out reaching dataset and add an architectural ablation that "
        "isolates the contribution of POYO's tokenization scheme."
    )

    h1(pdf, "2. Chosen Result")
    body(pdf,
        "We reproduce the same-animal, new-day transfer R^2 comparison (the headline of the paper's "
        "transfer experiments): train decoders on Day 1 of monkey \"T\", adapt on a Day 2 train split, and "
        "evaluate on a held-out Day 2 split. This operationalizes the paper's central claim that POYO's "
        "tokenization survives the unit-set drift between sessions where binned baselines do not."
    )

    h1(pdf, "3. Methodology")
    labeled(pdf, "Data.",
        "Two sessions from Perich-Miller 2018 (t_20130819_center_out_reaching and "
        "t_20130821_center_out_reaching), accessed through torch_brain. Targets are 2-D hand kinematics; "
        "spike data covers ~196 sorted units per session."
    )
    reset_x(pdf)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(0, 4.5, "Models.", ln=1)
    pdf.set_font("Helvetica", "", 10.5)
    bullet(pdf, "Wiener / Ridge filter -",
        "lagged binned spike counts with Ridge regression. Adapted by refitting on Day 1 plus Day 2 windows."
    )
    bullet(pdf, "MLP (binned) -",
        "flattens a [T, U] binned tensor and regresses kinematics; U is the global unit vocabulary across both days."
    )
    bullet(pdf, "GRU (binned) -",
        "clocked recurrent processing of bins."
    )
    bullet(pdf, "POYO (event tokens) -",
        "official torch_brain POYO with per-spike tokens, learned unit/session embeddings, and a Perceiver-style cross-attention block."
    )
    bullet(pdf, "POYO-binned (independent study) -",
        "same Perceiver backbone, fed binned spike counts as time tokens. Identical depth, latent count, and head count; only the tokenizer differs."
    )
    pdf.ln(1)
    labeled(pdf, "Training protocol.",
        "Three stages: (1) train on Day 1, (2) adapt on Day 2 train split, (3) evaluate on held-out Day 2. "
        "Bin size 20 ms, sequence length 1 s. Day-1 training: 100 epochs; adaptation: 40 epochs. "
        "Single NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM)."
    )
    labeled(pdf, "Modifications from the paper.",
        "We restricted to a single-monkey, two-session subset rather than the multi-animal mass-pretraining "
        "corpus used in the original work, so absolute numbers are not directly comparable. The independent "
        "study (POYO-binned) is not in the paper - it isolates the tokenizer's contribution by holding the "
        "rest of the architecture fixed."
    )

    h1(pdf, "4. Results & Analysis")

    # Two-column results: table on left, figure on right.
    y_top = pdf.get_y()
    table_x = pdf.l_margin
    table_w = 80
    pdf.set_xy(table_x, y_top)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(44, 5, "Model", border="B")
    pdf.cell(18, 5, "Pre R^2", border="B", align="R")
    pdf.cell(20, 5, "Adapt R^2", border="B", align="R")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 10.5)
    rows = [
        ("POYO (event tokens)", "-0.05", "0.699", True),
        ("MLP (binned)",        "-0.01", "0.697", False),
        ("GRU (binned)",        "-0.07", "0.612", False),
        ("POYO-binned",         "-0.05", "0.217", False),
        ("Wiener/Ridge",        "-0.03", "0.215", False),
    ]
    for name, pre, adp, bold in rows:
        pdf.set_x(table_x)
        if bold:
            pdf.set_font("Helvetica", "B", 10.5)
        pdf.cell(44, 5, name)
        pdf.cell(18, 5, pre, align="R")
        pdf.cell(20, 5, adp, align="R")
        pdf.ln(5)
        if bold:
            pdf.set_font("Helvetica", "", 10.5)

    table_bottom = pdf.get_y()
    img_x = pdf.l_margin + table_w + 4
    page_w = 215.9  # Letter width in mm
    img_w = page_w - pdf.l_margin - pdf.r_margin - table_w - 4
    pdf.image(str(FIG), x=img_x, y=y_top, w=img_w)
    img_bottom = y_top + img_w * 0.5  # the figure has roughly 2:1 aspect

    pdf.set_xy(pdf.l_margin, max(table_bottom, img_bottom) + 1)

    labeled(pdf, "Headline reproduction.",
        "All models collapse pre-adaptation (R^2 ~ 0), confirming the cross-session generalization gap. "
        "After adaptation, POYO achieves R^2 = 0.699, the strongest method we tested, consistent with the "
        "paper's claim that its tokenization survives unit-set drift better than fixed-feature baselines. "
        "At this single-monkey scale the binned MLP nearly matches POYO (0.697); the paper's larger gap "
        "appears when pretraining spans many animals - a regime we did not reproduce."
    )
    labeled(pdf, "Independent study - what does the tokenizer buy you?",
        "Replacing the event tokenizer with binned input on the same Perceiver backbone collapses "
        "performance to 0.217, within noise of a Wiener filter (0.215) and a 0.48 R^2 drop from event-token "
        "POYO. Two complementary readings: (a) architectural isolation - the Perceiver backbone alone does "
        "not explain POYO's quality; the tokenizer carries the result. (b) temporal resolution - binning at "
        "20 ms discards sub-bin spike timing that the event tokenizer preserves through continuous "
        "timestamps, so attention has nothing finer than 50 Hz to attend to."
    )
    labeled(pdf, "Discrepancies.",
        "A pre-poster run produced POYO-binned = 0.318 and POYO = 0.708; a re-run for this report produced "
        "0.217 and 0.699 with the same seed and config. The POYO-binned variability is the loudest. We "
        "suspect the binned variant is sensitive to initialization of the input projection: it has fewer "
        "anchor cues than event tokens (no per-spike timestamps, no per-unit identity injected at every "
        "event). A multi-seed sweep was beyond our compute budget."
    )

    h1(pdf, "5. Reflections")
    labeled(pdf, "What we learned.",
        "The headline result is not really \"POYO has better attention\" - at this scale a properly-tuned "
        "MLP nearly ties it. The headline result is that POYO's tokenizer is what makes a transformer "
        "competitive on neural data; pour binned inputs into the same backbone and the transformer is no "
        "better than ridge. This matches the paper's emphasis on tokenization but is easier to see once "
        "architecture is held fixed."
    )
    labeled(pdf, "What we'd do differently.",
        "Run multi-seed (the POYO-binned variability flagged a real instability), sweep bin size for the "
        "binned baselines (to confirm the temporal-resolution framing), and pretrain on more sessions to "
        "test whether POYO's gap over MLP widens with scale - the paper's actual scaling claim."
    )
    labeled(pdf, "Future directions.",
        "Scalability: extend pretraining across the full Perich-Miller corpus and trace the POYO-MLP gap "
        "as a function of scale. Zero-shot: evaluate transfer to a new monkey with no adaptation data, a "
        "regime in which binned baselines cannot even produce aligned features. Multimodal: attach "
        "kinematic, EMG, and behavioral heads to test whether POYO's session embedding routes across "
        "modalities. Real-time latency: POYO's per-spike tokenization grows with firing rate; profile and "
        "explore a streaming variant for closed-loop BCI."
    )

    h1(pdf, "References")
    pdf.set_font("Helvetica", "", 9.5)
    reset_x(pdf)
    pdf.multi_cell(0, 4.0,
        "[1] Azabou, M., Arora, V., Ganesh, V., Mao, X., Nachimuthu, S., Mendelson, M. J., Richards, B., "
        "Perich, M. G., Lajoie, G., & Dyer, E. L. (2023). A Unified, Scalable Framework for Neural "
        "Population Decoding. Advances in Neural Information Processing Systems 36 (NeurIPS 2023)."
    )
    reset_x(pdf)
    pdf.multi_cell(0, 4.0,
        "[2] Perich, M. G., Gallego, J. A., & Miller, L. E. (2018). A neural population mechanism for "
        "rapid learning. Neuron 100(4), 964-976."
    )
    reset_x(pdf)
    pdf.multi_cell(0, 4.0,
        "[3] torch_brain - NeuroFoundation. https://github.com/neuro-galaxy/torch_brain"
    )

    pdf.output(str(OUT))
    print(f"Wrote {OUT} ({pdf.page_no()} pages)")


if __name__ == "__main__":
    main()
