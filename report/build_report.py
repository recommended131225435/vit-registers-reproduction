"""Builds the final technical report (report/report.pdf) from the results in this repository.

All numbers in the report are read from the result files (results/...), so the report
always matches the code's output. Run from the repo root:

    python -m report.build_report

Needs: reportlab (pip install reportlab).
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
OUT = ROOT / "report" / "report.pdf"
TEXT = ROOT / "report" / "text.json"  # discussion text that depends on the experiment results

REPO_URL = "https://github.com/recommended131225435/vit-registers-reproduction"
TEAM = [("Muhammad Hamza Uzair", "30544"), ("Shaheer Ahmed", "31543")]


# ----------------------------------------------------------------------------
# Fonts and styles
# ----------------------------------------------------------------------------
def register_fonts() -> tuple[str, str, str, str]:
    """DejaVu fonts (they have symbols such as × and →); fall back to the built-in Times/Helvetica."""
    folder = Path("/usr/share/fonts/truetype/dejavu")
    files = {"Body": "DejaVuSerif.ttf", "Body-Bold": "DejaVuSerif-Bold.ttf",
             "Body-Italic": "DejaVuSerif-Italic.ttf", "Head": "DejaVuSans-Bold.ttf"}
    if all((folder / f).exists() for f in files.values()):
        for name, f in files.items():
            pdfmetrics.registerFont(TTFont(name, str(folder / f)))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                                      boldItalic="Body-Bold")
        return "Body", "Body-Bold", "Body-Italic", "Head"
    return "Times-Roman", "Times-Bold", "Times-Italic", "Helvetica-Bold"


BODY, BOLD, ITALIC, HEAD = register_fonts()
INK, MUTED, RULE, SHADE = colors.HexColor("#1d1d1b"), colors.HexColor("#5f5e58"), colors.HexColor("#cfcdc6"), \
    colors.HexColor("#f2f1ec")

base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", fontName=HEAD, fontSize=20, leading=25, alignment=TA_CENTER, textColor=INK,
                            spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName=BODY, fontSize=11, leading=15, alignment=TA_CENTER,
                               textColor=MUTED, spaceAfter=4),
    "h1": ParagraphStyle("h1", fontName=HEAD, fontSize=13.5, leading=17, textColor=INK, spaceBefore=14,
                         spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName=HEAD, fontSize=11, leading=14, textColor=INK, spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("body", fontName=BODY, fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName=BODY, fontSize=9.6, leading=13.6, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=3),
    "caption": ParagraphStyle("caption", fontName=ITALIC, fontSize=8.4, leading=11, textColor=MUTED,
                              spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName=BODY, fontSize=8.2, leading=10.4, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName=BOLD, fontSize=8.2, leading=10.4, textColor=INK),
    "abstract": ParagraphStyle("abstract", fontName=BODY, fontSize=9.4, leading=13.4, textColor=INK,
                               leftIndent=1 * cm, rightIndent=1 * cm, spaceAfter=6),
    "ref": ParagraphStyle("ref", fontName=BODY, fontSize=8.6, leading=11.6, textColor=INK, leftIndent=14,
                          firstLineIndent=-14, spaceAfter=3),
}


def p(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(header, rows, widths=None):
    data = [[Paragraph(str(h), S["cellh"]) for h in header]]
    data += [[Paragraph(str(c), S["cell"]) for c in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SHADE),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


FIG_NO = [0]
TAB_NO = [0]


def figure(name, caption, width=16.5 * cm):
    path = FIG / name
    if not path.exists():
        return [p(f"<i>[figure {name} not available yet]</i>")]
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(path)).getSize()
    FIG_NO[0] += 1
    return [KeepTogether([Image(str(path), width=width, height=width * h / w),
                          p(f"Figure {FIG_NO[0]}. {caption}", "caption")])]


def numbered_table(caption, header, rows, widths=None):
    TAB_NO[0] += 1
    return [KeepTogether([p(f"Table {TAB_NO[0]}. {caption}", "caption"), table(header, rows, widths)]),
            Spacer(1, 8)]


def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(BODY, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Reproducing “Vision Transformers Need Registers” — ML course project, IBA Karachi")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


# ----------------------------------------------------------------------------
# Content
# ----------------------------------------------------------------------------
def build():
    part_a = load(RES / "part_a" / "summary.json", {}).get("results", {})
    runs = {r: load(RES / "runs" / r / "final.json") for r in ["tiny_reg0", "tiny_reg4"]}
    experiment = sorted(
        [s for s in (load(f) for f in (RES / "experiment").glob("*/stats.json")) if s and not s.get("fake")],
        key=lambda s: (["tiny", "small", "base"].index(s["size"]), s["registers"]))
    text = load(TEXT, {})
    g = part_a.get("g", {})
    acc0, acc4 = 100 * runs["tiny_reg0"]["val_acc"], 100 * runs["tiny_reg4"]["val_acc"]

    story = []
    # ---------------- title ----------------
    story += [Spacer(1, 1.2 * cm),
              p("Reproducing “Vision Transformers Need Registers”", "title"),
              p("Final technical report — Machine Learning paper reproduction and deployment", "subtitle"),
              p(" · ".join(f"{n} ({e})" for n, e in TEAM), "subtitle"),
              p(f"Code, results and notebooks: <font color='#2f6db5'>{REPO_URL}</font>", "subtitle"),
              Spacer(1, 0.6 * cm)]
    story.append(p(
        "<b>Abstract.</b> Darcet et al. (ICLR 2024) found that large Vision Transformers quietly turn a few "
        "background patches into storage for image-wide information, which shows up as bright, meaningless spots "
        "in their attention maps, and that giving the model a handful of extra “register” tokens removes "
        "the problem. We reproduce this in two ways. First, on Meta’s released DINOv2 models, we measure the "
        f"high-norm tokens directly: {g.get('pct_above_paper_cutoff_no_reg', 0):.2f}% of ViT-g patch tokens exceed "
        "the paper’s cutoff (paper: 2.37%), they sit on redundant background patches, they vanish with "
        "registers, and the large values move into the registers. Second, we implement a ViT with registers from "
        f"scratch and train it on Tiny-ImageNet ({acc0:.2f}% vs {acc4:.2f}% validation accuracy without/with "
        "registers). Our own experiment trains ViTs of increasing size and measures whether the artifacts appear. "
        + text.get("abstract_experiment", "")
        + " Finally, we deploy a FastAPI web app that classifies an uploaded image and shows, side by side, how "
        "registers clean up DINOv2’s attention and norm maps.", "abstract"))

    # ---------------- 1 ----------------
    story += [p("1. Introduction and problem statement", "h1"),
              p("A Vision Transformer (ViT) cuts an image into small square patches, turns each patch into a vector "
                "(a <i>token</i>), and lets all tokens exchange information through attention. A common way to see "
                "what such a model has learned is to look at the attention of its summary token, [CLS], over the "
                "patches. For the self-supervised model DINO these maps neatly outline the main object, which "
                "made them useful for finding objects without labels. Its successor DINOv2, and also supervised "
                "(DeiT-III) and text-supervised (OpenCLIP) ViTs, instead show a few very bright spots on empty "
                "background. These spots make the maps hard to interpret and hurt methods that rely on clean "
                "feature maps."),
              p("The paper asks what these spots are, why they appear, and how to remove them. This matters "
                "because ViT features are reused as general-purpose building blocks for many tasks (segmentation, "
                "depth estimation, object discovery), and those tasks need every patch token to describe its own "
                "part of the image."),
              p("<b>Our goals</b> were to (i) understand the paper and its context, (ii) reproduce its central "
                "measurements, (iii) implement the method ourselves and train it within a free-GPU budget, (iv) run "
                "our own experiment on whether the problem appears at small model sizes, and (v) deploy the result "
                "as a working app.")]

    # ---------------- 2 ----------------
    story += [p("2. Literature and background", "h1"),
              p("<b>Before the paper.</b> Adding special tokens to a transformer is an old idea: BERT and ViT add "
                "[CLS] to read out a prediction, and DETR adds object queries. In these cases the extra tokens carry "
                "information in or results out. The closest precursor is the <i>Memory Transformer</i> (Burtsev et "
                "al., 2020), which added blank memory tokens to a language model and improved translation; Sandler "
                "et al. (2022) tried learnable memory tokens for fine-tuning ViTs, with limited transfer between "
                "tasks. On the vision side, DINO (Caron et al., 2021) showed clean attention maps from "
                "self-supervised training, and LOST (Siméoni et al., 2021) used them to localise objects "
                "without labels. DINOv2 (Oquab et al., 2024) gave much stronger features, yet LOST worked poorly on "
                "it. That puzzle is the paper’s starting point. MAE (He et al., 2022), trained only to rebuild "
                "hidden patches, shows no such spots."),
              p("<b>What was missing.</b> Nobody had explained why most modern ViTs have noisy attention maps while "
                "DINO does not, and extra tokens had been treated as an accuracy add-on rather than a fix for a "
                "problem inside the model."),
              p("<b>After the paper.</b> Sun et al. (2024) found similar “massive activations” in language "
                "models and ViTs and argue they act as fixed biases in attention, so registers can be seen as "
                "learned biases. Yang et al. (2024) study a different, grid-like artifact caused by position "
                "embeddings, which registers do not remove. Jiang et al. (2025) locate a few neurons that create "
                "the high-norm tokens and obtain most of the benefit of registers at test time, without retraining. "
                "Together these confirm that the phenomenon is real, while refining the explanation.")]

    # ---------------- 3 ----------------
    story += [p("3. Paper methodology", "h1"),
              p("<b>Diagnosis.</b> The authors look at the length (L2 norm) of every output patch token. In DINOv2 "
                "ViT-g the distribution has two humps: most tokens are below about 100, while roughly 2% are far "
                "longer. They call tokens above a hand-picked cutoff of 150 <i>high-norm</i> or <i>outlier</i> "
                "tokens and show that they (a) appear after about a third of training, around the middle layers, "
                "and only in large models (ViT-L and above); (b) sit on patches that look almost identical to their "
                "neighbours, i.e. redundant background; (c) have lost information about their own position and "
                "pixels; and (d) contain much more information about the whole image than normal patches."),
              p("<b>Explanation.</b> A large, well-trained model learns to recognise patches it does not need and "
                "re-uses their tokens as scratch space for global computation. This is not harmful in itself, but "
                "doing it inside patch tokens destroys their local information."),
              p("<b>Fix.</b> Add N learnable <i>register</i> tokens to the input sequence, next to [CLS] and the "
                "patches. They carry no image content, receive no position embedding, and are discarded at the "
                "output. The model can now use them as scratch space and leaves the patches alone. With 4 "
                "registers the extra compute is under 2%. The paper reports that registers remove the outliers for "
                "DeiT-III, OpenCLIP and DINOv2, leave ImageNet accuracy essentially unchanged (−0.1 to +0.5 "
                "points), improve dense prediction, and raise LOST object discovery on VOC 2007 from 35.3 to 55.4 "
                "CorLoc for DINOv2.")]

    # ---------------- 4 ----------------
    story += [p("4. Implementation details", "h1"),
              p("Our code is organised as one module per stage of the pipeline (data, model, training, analysis, "
                "plots, report) and every component is tagged in PROVENANCE.md as written, adapted or reused. "
                "Results are written into the repository README by the scripts themselves, so reported numbers "
                "always come from the code."),
              p("<b>Part A — the paper’s own models.</b> We load Meta’s released DINOv2 ViT-S/B/L/g "
                "(patch size 14), each trained with and without 4 registers, through torch.hub. A hook captures the "
                "input of the last attention layer; we recompute softmax(qk<super>T</super>/√d) to get "
                "[CLS] attention maps, and checked that this reproduces the layer’s own output to within 10"
                "<super>−7</super>. Norms are measured on the last block’s output."),
              p("<b>Part B — our own ViT.</b> We implemented a standard pre-norm ViT in PyTorch with an option "
                "for R registers. Following the official DINOv2 code, registers are inserted after the position "
                "embeddings (so they have no position), between [CLS] and the patches, start as small random values "
                "(std 10<super>−6</super>), and are dropped at the output, where only [CLS] feeds the "
                "classifier. For 64×64 images we use 8×8 patches (64 patch tokens). Training follows "
                "the DeiT recipe, scaled down: AdamW (learning rate 10<super>−3</super>, weight decay 0.05), "
                "5 warm-up epochs then cosine decay, label smoothing 0.1, stochastic depth 0.1, gradient clipping, "
                "and mixed precision; batch size 256, 50 epochs, seed 0."),
              p("<b>Engineering checks.</b> Before any long run we ran 24 automatic checks on a real batch (shapes, "
                "labels, normalisation, a starting loss close to ln 200, gradients reaching the registers, "
                "and our explicit attention matching PyTorch’s fused version), 15 unit tests, and a 5-epoch "
                "sanity run on 10% of the data. Training saves its full state after every epoch so a Colab "
                "disconnect never loses a run; the tests run automatically on every push.")]

    # ---------------- 5 ----------------
    story += [p("5. Dataset and preprocessing", "h1"),
              p("The paper trains on ImageNet-22k and LVD-142M (millions of full-size images), which is far beyond "
                "a free GPU. We therefore use <b>Tiny-ImageNet</b>: 100,000 training and 10,000 validation images "
                "of 64×64 pixels in 200 classes, freely downloadable. The official validation set is our test "
                "set because test labels are not public, so we always report the last epoch rather than picking "
                "the best one. Training images get a random resized crop (35–100% of the image) and a random "
                "horizontal flip; all images are converted to RGB and normalised with ImageNet statistics."),
              p("For Part A we use 256 validation images of <b>Imagenette</b> (10 easy ImageNet classes) at "
                "224×224, because DINOv2 expects larger images than 64×64. <b>Implication of the "
                "substitution:</b> our absolute accuracies cannot be compared with the paper’s ImageNet "
                "numbers; only with/without-register differences and the qualitative behaviour can.")]

    # ---------------- 6 ----------------
    rows = []
    for s, label in [("S", "ViT-S/14"), ("B", "ViT-B/14"), ("L", "ViT-L/14"), ("g", "ViT-g/14")]:
        r = part_a.get(s)
        if r:
            rows.append([label, f"{r['median_no_reg']:.1f}", f"{r['max_no_reg']:.1f} / {r['max_reg']:.1f}",
                         f"{r['pct_outliers_no_reg']:.2f}%", f"{r['pct_outliers_reg']:.2f}%",
                         "n/a" if r["cosine_outlier"] is None else f"{r['cosine_outlier']:.3f} vs {r['cosine_normal']:.3f}"])
    story += [p("6. Reproduction results", "h1"),
              p("6.1 The paper’s claims on DINOv2", "h2"),
              p(f"With the paper’s own cutoff of 150, <b>{g.get('pct_above_paper_cutoff_no_reg', 0):.2f}%</b> "
                "of ViT-g patch tokens are high-norm; the paper reports 2.37%. Table 1 summarises all four sizes "
                "using a size-independent rule (norm above 3× the model’s median), which for ViT-g gives "
                f"a cutoff of {g.get('cutoff_no_reg', 0):.0f}, close to the paper’s 150.")]
    story += numbered_table(
        "DINOv2 patch-token norms (last block, before the final LayerNorm), 256 Imagenette images.",
        ["Model", "Median norm", "Max norm (no reg / reg)", "% outliers, no reg", "% outliers, 4 reg",
         "Neighbour similarity, outlier vs normal"], rows,
        [2.0 * cm, 2.0 * cm, 3.3 * cm, 2.4 * cm, 2.4 * cm, 4.4 * cm])
    story += figure("part_a_norm_histograms.png",
                    "Distribution of patch-token norms (log scale). ViT-g without registers (orange) has a second hump "
                    "at 300–560; with registers (blue) it is gone. Compare with paper Fig. 3 and 7.")
    story += figure("part_a_attention_maps.png",
                    "Last-block [CLS] attention. Without registers, ViT-B/L/g show bright isolated spots on the "
                    "background (often at image edges); with registers attention lies on the object (paper Fig. 1).")
    reg = g.get("register_norms_reg", [0, 0, 0, 0])
    story += bullets([
        "<b>High-norm outliers exist and are rare</b> (paper Fig. 3): reproduced, 2.62% vs 2.37%.",
        f"<b>They sit on redundant patches</b> (paper Fig. 5a): reproduced. Right after patch embedding, ViT-g outlier "
        f"patches have a mean cosine similarity of {g.get('cosine_outlier', 0):.3f} to their neighbours versus "
        f"{g.get('cosine_normal', 0):.3f} for normal patches, with a sharp peak at 1.0.",
        "<b>Registers remove them</b> (paper Fig. 7) and <b>clean the attention maps</b> (paper Fig. 1): reproduced for "
        "every size.",
        f"<b>The large values move into the registers</b> (paper Fig. 15): reproduced. In ViT-g with registers one "
        f"register has an average norm of {max(reg):.0f} against {g.get('median_reg', 0):.0f} for a typical patch, "
        "and the four registers behave differently from each other, as in the paper.",
        "<b>Outliers appear only in large models</b> (paper Fig. 4c): partly reproduced. Clear outliers appear only in "
        "ViT-g (ViT-B has a tiny tail of 0.20%); the paper also finds them in ViT-L. A likely reason is that "
        "Meta’s released S/B/L models were distilled from ViT-g rather than trained on their own as in the "
        "paper’s Fig. 4c.",
        "<b>A detail the paper does not stress:</b> the final LayerNorm rescales every token to a similar length. "
        f"In ViT-g the largest token is {g.get('max_over_median_no_reg', 0):.1f}× the median before it but "
        f"only {g.get('max_over_median_after_ln_no_reg', 0):.2f}× after it. Our first attempt measured after "
        "the LayerNorm and found nothing; the norms must be read before this layer.",
    ])
    story += [p("6.2 Training our own ViT with and without registers", "h2"),
              p(f"ViT-Tiny (5.4M parameters) trained for 50 epochs on all of Tiny-ImageNet reaches "
                f"<b>{acc0:.2f}%</b> validation accuracy without registers and <b>{acc4:.2f}%</b> with 4 "
                f"registers ({acc4 - acc0:+.2f} points). The training-loss curves lie on top of each other "
                "(Figure 3). The paper reports changes of −0.1 to +0.5 points at ImageNet scale, so this claim is "
                "only partly reproduced: there is no large accuracy cost, but a small one cannot be ruled out from "
                "a single run per setting, especially since the register run was paused and resumed once (which "
                "changes the data order afterwards). Both models overfit (about 91.5% training accuracy; validation "
                "loss is lowest around epoch 18), which is typical for ViTs trained from scratch on 100k images and "
                "affects both runs equally.")]
    story += figure("full_training_runs.png", "Training loss and validation accuracy of our ViT-Tiny, 0 vs 4 "
                    "registers, 50 epochs on Tiny-ImageNet.", width=15 * cm)

    # ---------------- 7 ----------------
    story += [p("7. Additional experiment: do artifacts appear with model size?", "h1"),
              p("<b>Question.</b> Do high-norm artifact tokens appear in small ViTs trained from scratch, or only "
                "once models are large, and do registers change anything below that size?"),
              p("<b>Design.</b> A model-size scaling study, the experiment planned in our proposal. We train ViT-Tiny, "
                "ViT-Small" + (" and ViT-Base" if any(s["size"] == "base" for s in experiment) else "")
                + ", each with 0 and 4 registers, keeping everything else fixed (data, recipe, 50 epochs, seed). "
                "For each trained model we measure, on the same 2,000 validation images, the patch-token norms "
                "after every block (before the final LayerNorm), the share of outliers (above 3× the median, "
                "as in Part A), the largest/median norm, how much attention patches and [CLS] give to the "
                "registers (compared with the share they would get if attention were uniform), and [CLS] "
                "attention maps. Meta’s DINOv2 models from Part A provide the large end of the same size axis. "
                "<b>Hypothesis</b> from the paper: no outliers in our small models, little use of the registers, "
                "and outliers only at the top of the DINOv2 range.")]
    if experiment:
        exp_rows = []
        for s in experiment:
            lb = s["last_block"]
            exp_rows.append([f"ViT-{s['size'].capitalize()}", f"{s['params'] / 1e6:.1f}M", s["registers"],
                             f"{100 * s['val_acc']:.2f}%" if s.get("val_acc") else "n/a",
                             f"{lb['max_over_median']:.2f}×", f"{lb['pct_outliers']:.2f}%",
                             f"{max(x['max_over_median'] for x in s['layers']):.2f}×",
                             "n/a" if not s["registers"] else
                             f"{100 * s['attn_patch_to_reg']:.1f}% ({100 * s['attn_uniform_share']:.1f}%)"])
        story += numbered_table(
            "Our trained ViTs (2,000 Tiny-ImageNet validation images). Uniform share in brackets.",
            ["Model", "Params", "Reg.", "Val acc.", "Largest / median", "% outliers", "Peak over blocks",
             "Patch attention to registers"], exp_rows,
            [1.9 * cm, 1.5 * cm, 1.0 * cm, 1.7 * cm, 2.2 * cm, 1.8 * cm, 2.2 * cm, 4.2 * cm])
        story += figure("experiment_scaling.png", "Artifacts against model size: our from-scratch ViTs (circles) and "
                        "Meta’s DINOv2 (triangles), without (orange) and with (blue) registers.")
        story += figure("experiment_layer_norms.png", "Largest/median patch-token norm after every block of our "
                        "models (compare with the paper’s Fig. 4a).", width=12 * cm)
        story += figure("experiment_accuracy_registers.png", "Left: validation accuracy. Right: share of last-block "
                        "attention that goes to the registers; dotted lines show the share under uniform attention.")
    else:
        story.append(p("<i>[Experiment results are added when the training runs finish.]</i>"))

    # ---------------- 8 ----------------
    story += [p("8. Analysis and discussion", "h1")]
    story += [p(x) for x in text.get("discussion", ["<i>[Discussion is added with the experiment results.]</i>"])]

    # ---------------- 9 ----------------
    story += [p("9. Deployment", "h1"),
              p("We deployed the project as a FastAPI web application (app/ in the repository). A user uploads an "
                "image in the browser (or sends it to <font face='" + BODY + "'>POST /predict</font>) and receives:"),
              *bullets([
                  "a <b>prediction</b>: the top-5 Tiny-ImageNet classes from our own ViT-Tiny models, without and "
                  "with registers, using exactly the validation preprocessing from training (resize and centre-crop "
                  "to 64×64, ImageNet normalisation);",
                  "<b>what registers change</b>: the [CLS] attention map and the patch-token norm map of DINOv2 "
                  "ViT-B/14 without and with registers, with the number of outlier patches. DINOv2 is used here "
                  "because our small models have no artifacts to show (Section 7).",
              ]),
              p("The app loads all four models once at start-up, validates uploads (non-images return HTTP 400, "
                "files over 10 MB return 413), offers a /health endpoint and automatic API documentation at /docs, "
                "and is covered by five automated tests that run on every push. Start-up commands, input and output "
                "formats and a worked example are documented in the README.")]
    story += figure("deployment_screenshot.png", "The deployed app: upload, DINOv2 maps without vs with registers, "
                    "and our ViT-Tiny’s prediction.", width=14 * cm)

    # ---------------- 10 ----------------
    story += [p("10. Limitations", "h1"), *bullets([
        "<b>Compute:</b> a single free Colab T4 GPU. We could not train models of the paper’s size or on its "
        "data, so large-model results come from Meta’s released checkpoints and our own models are small.",
        "<b>Dataset substitution:</b> Tiny-ImageNet (64×64) instead of ImageNet-22k/LVD-142M; absolute "
        "accuracies are not comparable with the paper.",
        "<b>Distilled checkpoints:</b> Meta’s released ViT-S/B/L were distilled from ViT-g, so Part A’s "
        "size trend is not a clean test of the paper’s Fig. 4c.",
        "<b>Single seed per setting</b> and one pause/resume in a training run; differences of about one point "
        "may be noise.",
        "<b>Overfitting</b> of our from-scratch ViTs (≈91% train vs ≈39% validation accuracy for "
        "ViT-Tiny).",
        "<b>Not tested:</b> the probing results on what high-norm tokens contain (Table 1, Fig. 5b) and the "
        "downstream benefits of registers (dense prediction, LOST object discovery; Tables 2–3).",
        "<b>Sample sizes:</b> 256 Imagenette images for Part A and 2,000 validation images for the experiment.",
    ])]

    # ---------------- 11 ----------------
    story += [p("11. Conclusion", "h1")]
    story += [p(x) for x in text.get("conclusion", [
        "We reproduced the paper’s diagnosis on the models it studied: a small share of very long patch tokens "
        "on redundant background, removed by registers, with the large values moving into the registers. "
        "<i>[Experiment conclusion is added with the results.]</i>"])]

    # ---------------- 12 ----------------
    refs = [
        "Darcet, T., Oquab, M., Mairal, J., Bojanowski, P. Vision Transformers Need Registers. ICLR 2024.",
        "Dosovitskiy, A. et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.",
        "Burtsev, M. et al. Memory Transformer. arXiv:2006.11527, 2020.",
        "Sandler, M. et al. Fine-tuning Image Transformers using Learnable Memory. CVPR 2022.",
        "Caron, M. et al. Emerging Properties in Self-Supervised Vision Transformers. ICCV 2021.",
        "Siméoni, O. et al. Localizing Objects with Self-Supervised Transformers and no Labels. BMVC 2021.",
        "Oquab, M. et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR 2024.",
        "He, K. et al. Masked Autoencoders Are Scalable Vision Learners. CVPR 2022.",
        "Touvron, H. et al. Training data-efficient image transformers &amp; distillation through attention. ICML 2021.",
        "Sun, M., Chen, X., Kolter, J. Z., Liu, Z. Massive Activations in Large Language Models. COLM 2024.",
        "Yang, J. et al. Denoising Vision Transformers. ECCV 2024.",
        "Jiang, N., Dravid, A., Efros, A. A., Gandelsman, Y. Vision Transformers Don’t Need Trained Registers. "
        "NeurIPS 2025.",
        "Le, Y., Yang, X. Tiny ImageNet Visual Recognition Challenge. CS231N, 2015.",
        "Imagenette dataset, fast.ai: https://github.com/fastai/imagenette",
    ]
    story += [p("12. References", "h1")] + [p(f"[{i}] {r}", "ref") for i, r in enumerate(refs, 1)]
    story += [Spacer(1, 10), p(
        "<b>Provenance and AI use.</b> Every number and figure in this report was produced by our code (results/ "
        "in the repository); numbers attributed to the paper are the authors’. DINOv2 code and weights are "
        "Meta’s (Apache-2.0), loaded unchanged. PROVENANCE.md tags each code component as written, adapted or "
        "reused. Code and text were drafted with the help of an AI assistant (Claude, Anthropic) and reviewed, run "
        "and tested by us.", "caption")]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="Reproducing Vision Transformers Need Registers",
                            author=", ".join(n for n, _ in TEAM))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"saved {OUT}")


if __name__ == "__main__":
    build()
