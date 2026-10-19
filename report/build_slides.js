// Builds the presentation (report/presentation.pptx) from the results in this repository.
// Run from the repo root:  node report/build_slides.js
// Needs: npm install pptxgenjs   (numbers are read from results/, so the slides always match the code)

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const RES = path.join(ROOT, "results");
const FIG = path.join(RES, "figures");
const ASSETS = path.join(__dirname, "assets");
const OUT = path.join(__dirname, "presentation.pptx");
const REPO = "github.com/recommended131225435/vit-registers-reproduction";
const TEAM = "Muhammad Hamza Uzair (30544)  ·  Shaheer Ahmed (31543)";

const THEME = {
  name: "Registers",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1C1A2B", lt1: "FFFFFF", dk2: "2A1F4D", lt2: "F3F2F8",
    accent1: "21918C", accent2: "FF7F0E", accent3: "1F77B4", accent4: "FDE725",
    accent5: "6B6880", accent6: "440154", hlink: "1F77B4", folHlink: "6B6880",
  },
};
const HEX = { noReg: "FF7F0E", reg: "1F77B4", grid: "E4E2EC", muted: "6B6880", ink: "1C1A2B", teal: "21918C" };

const readJson = (p, fallback = null) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : fallback);
const readCsv = (p) => {
  const [head, ...rows] = fs.readFileSync(p, "utf8").trim().split(/\r?\n/);
  const keys = head.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [keys[i], Number(v)])));
};
const pct = (v, d = 2) => `${(100 * v).toFixed(d)}%`;

const partA = (readJson(path.join(RES, "part_a", "summary.json"), {}).results) || {};
const g = partA.g || {};
const final0 = readJson(path.join(RES, "runs", "tiny_reg0", "final.json"));
const final4 = readJson(path.join(RES, "runs", "tiny_reg4", "final.json"));
const log0 = readCsv(path.join(RES, "runs", "tiny_reg0", "log.csv"));
const log4 = readCsv(path.join(RES, "runs", "tiny_reg4", "log.csv"));
const expDir = path.join(RES, "experiment");
const ORDER = ["tiny", "small", "base"];
// Same rule as scripts/summarise_experiment.py: one run per (size, registers), the one at MAIN_LR for its size.
const MAIN_LR = { tiny: 2.5e-4, small: 2.5e-4, base: 2.5e-4 };
const allRuns = fs.existsSync(expDir)
  ? fs.readdirSync(expDir).map((d) => readJson(path.join(expDir, d, "stats.json"))).filter((s) => s && !s.fake)
      .map((s) => ({ ...s, lr: (readJson(path.join(RES, "runs", s.run, "config.json"), {}) || {}).lr || 1e-3 }))
  : [];
const experiment = [];
const extraRuns = [];
for (const size of ORDER) {
  for (const regs of [0, 4]) {
    const group = allRuns.filter((r) => r.size === size && r.registers === regs);
    if (!group.length) continue;
    const chosen = group.find((r) => Math.abs(r.lr - MAIN_LR[size]) < 1e-9) || group[0];
    experiment.push(chosen);
    extraRuns.push(...group.filter((r) => r !== chosen));
  }
}
const text = readJson(path.join(__dirname, "text.json"), {});
const screenshot = [path.join(FIG, "deployment_screenshot.png"), path.join(ASSETS, "deployment_screenshot.png")]
  .find((p) => fs.existsSync(p));

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Reproducing Vision Transformers Need Registers";
pres.author = TEAM;
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333;

// ---------------- layouts ----------------
pres.defineSlideMaster({
  title: "Title Dark",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.1, w: 11.7, h: 1.8, fontSize: 44, bold: true,
      color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.05, w: 11.7, h: 1.6, fontSize: 18,
      color: C.background2, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "Content",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  slideNumber: { x: 12.4, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5 },
  objects: [
    { text: { text: "Vision Transformers Need Registers — reproduction", options: { x: 0.6, y: 7.0, w: 8, h: 0.3,
      fontSize: 10, color: C.accent5, margin: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 32, bold: true,
      color: C.text1, valign: "middle", margin: 0 }, text: "" } },
  ],
});

// ---------------- helpers ----------------
function content(title, section) {
  const s = pres.addSlide({ masterName: "Content", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function bullets(slide, items, opts) {
  slide.addText(items.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 },
  })), { isTextBox: true, fontSize: 16, color: C.text1, valign: "top", margin: 0, ...opts });
}
function stat(slide, value, label, x, y, w, colour) {
  slide.addText(value, { isTextBox: true, x, y, w, h: 0.95, fontSize: 48, bold: true, fontFace: THEME.headFontFace,
    color: colour, margin: 0, objectName: `stat ${label}` });
  slide.addText(label, { isTextBox: true, x, y: y + 0.95, w, h: 0.6, fontSize: 13, color: C.accent5, margin: 0,
    valign: "top" });
}
function card(slide, x, y, w, h, heading, body, colour) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: C.background2 },
    line: { color: C.background2 }, objectName: `card ${heading}` });
  slide.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.28, w: 0.36, h: 0.36, fill: { color: colour },
    line: { color: colour } });
  slide.addText(heading, { isTextBox: true, x: x + 0.75, y: y + 0.2, w: w - 0.95, h: 0.52, fontSize: 17, bold: true,
    color: C.text1, margin: 0, valign: "middle" });
  slide.addText(body, { isTextBox: true, x: x + 0.25, y: y + 0.85, w: w - 0.5, h: h - 1.0, fontSize: 14,
    color: C.text1, margin: 0, valign: "top" });
}
function picture(slide, file, x, y, maxW, maxH, name) {
  if (!file || !fs.existsSync(file)) {
    slide.addText(`[${path.basename(file || "figure")} — added when results are in]`, { isTextBox: true, x, y, w: maxW,
      h: maxH, fontSize: 14, italic: true, color: C.accent5, align: "center", valign: "middle",
      fill: { color: C.background2 } });
    return;
  }
  const png = fs.readFileSync(file);
  const iw = png.readUInt32BE(16), ih = png.readUInt32BE(20); // PNG header: width, height
  const scale = Math.min(maxW / iw, maxH / ih);
  const w = iw * scale, h = ih * scale;
  slide.addImage({ path: file, x: x + (maxW - w) / 2, y: y + (maxH - h) / 2, w, h, objectName: name });
}

// ---------------- 1. title ----------------
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "Title Dark", sectionTitle: "Introduction" });
s.addText("Vision Transformers Need Registers", { placeholder: "title" });
s.addText([
  { text: "Reproduction, experiment and deployment of Darcet et al., ICLR 2024", options: { breakLine: true } },
  { text: TEAM, options: { breakLine: true, color: THEME.colors.accent4 } },
  { text: REPO, options: { fontSize: 14 } },
], { placeholder: "body" });
s.addNotes("Hello, we are Hamza and Shaheer. We reproduced the ICLR 2024 paper Vision Transformers Need Registers, ran our own experiment on it, and deployed it as a web app. Everything is in our GitHub repository.");

// ---------------- 2. problem ----------------
s = content("Big ViTs look at empty background", "Introduction");
picture(s, path.join(ASSETS, "attention_vitg.png"), 7.3, 1.45, 5.4, 5.3, "attention maps ViT-g");
s.addText([
  { text: "photo", options: {} }, { text: "      no registers      ", options: { color: HEX.noReg, bold: true } },
  { text: "with registers", options: { color: HEX.reg, bold: true } },
], { isTextBox: true, x: 7.3, y: 6.72, w: 5.4, h: 0.3, fontSize: 12, align: "center", margin: 0, color: C.accent5 });
bullets(s, [
  "A Vision Transformer cuts an image into patches; [CLS] attention shows where it looks",
  "In large models (DINOv2, CLIP, DeiT-III) this map has bright spots on empty background",
  "These spots make the maps meaningless and hurt tasks that need clean patch features",
  "The paper asks: what are these spots, why do they appear, and how can we remove them?",
], { x: 0.6, y: 1.6, w: 6.3, h: 4.8 });
s.addNotes("This is our own measurement on Meta's DINOv2 ViT-g. In the middle column, without registers, the attention has random yellow squares in the background, often at the edges. On the right, with registers, the attention sits on the fish and the person. The paper explains these spots and fixes them.");

// ---------------- 3. key idea ----------------
s = content("The idea: give the model scratch space", "Introduction");
card(s, 0.6, 1.5, 3.9, 2.3, "Diagnosis", "About 2% of patch tokens become ~10× longer than normal. They sit on boring background and hold global, not local, information.", C.accent2);
card(s, 4.72, 1.5, 3.9, 2.3, "Explanation", "A large model needs space for global computation, so it takes over patches it considers useless.", C.accent6);
card(s, 8.84, 1.5, 3.9, 2.3, "Fix: registers", "Add 4 learnable tokens with no image content and no position. The model uses them instead; they are dropped at the output.", C.accent3);
// token sequence diagram
const seq = [["[CLS]", C.accent5], ["REG", C.accent3], ["REG", C.accent3], ["REG", C.accent3], ["REG", C.accent3],
  ["patch", C.accent1], ["patch", C.accent1], ["patch", C.accent1], ["…", C.accent1], ["patch", C.accent1]];
seq.forEach(([label, colour], i) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.9 + i * 1.18, y: 4.75, w: 1.02, h: 0.62, rectRadius: 0.08,
    fill: { color: colour }, line: { color: colour } });
  s.addText(label, { isTextBox: true, x: 0.9 + i * 1.18, y: 4.75, w: 1.02, h: 0.62, fontSize: 13, bold: true,
    color: C.background1, align: "center", valign: "middle", margin: 0 });
});
s.addText("Token sequence with registers: [CLS] is used for the prediction, the 4 registers are thrown away, patches describe the image. Cost: < 2% extra compute.",
  { isTextBox: true, x: 0.9, y: 5.6, w: 11.6, h: 0.8, fontSize: 14, color: C.accent5, margin: 0 });
s.addNotes("The paper has three steps. Diagnosis: the spots are tokens with a very large norm, about 2% of them, on redundant background. Explanation: the model reuses useless patches as scratch space for global information. Fix: add a few extra tokens called registers, which are discarded at the output, so the patches stay clean.");

// ---------------- 4. what we built ----------------
pres.addSection({ title: "Reproduction" });
s = content("What we built", "Reproduction");
card(s, 0.6, 1.5, 5.9, 2.35, "Part A: the paper's own models", "Meta's released DINOv2 S/B/L/g, each with and without registers. We measure token norms, outliers and attention maps exactly as the paper does.", C.accent1);
card(s, 6.83, 1.5, 5.9, 2.35, "Part B: our own ViT with registers", "Written from scratch in PyTorch, trained on Tiny-ImageNet (100k images, 200 classes). 24 forward-pass checks, 15 unit tests, automatic CI.", C.accent3);
card(s, 0.6, 4.1, 5.9, 2.35, "Experiment: model size", "Train ViTs of growing size with 0 and 4 registers and measure whether artifacts appear.", C.accent2);
card(s, 6.83, 4.1, 5.9, 2.35, "Deployment: FastAPI app", "Upload an image: our ViT-Tiny predicts the class, DINOv2 shows its maps without vs with registers.", C.accent6);
s.addNotes("We worked in four parts. Part A uses Meta's released models to check the paper's claims directly. Part B is our own implementation, trained on Tiny-ImageNet because ImageNet is too big for a free GPU. Then our experiment on model size, and finally the deployed app.");

// ---------------- 5. headline reproduction ----------------
s = content("Reproduced: rare, huge tokens that registers remove", "Reproduction");
stat(s, `${(g.pct_above_paper_cutoff_no_reg || 0).toFixed(2)}%`, "of ViT-g patch tokens above the paper's cutoff of 150 (paper: 2.37%)", 0.6, 1.55, 3.6, HEX.noReg);
stat(s, `${(g.pct_outliers_reg || 0).toFixed(2)}%`, "outlier tokens in the same model trained with registers", 0.6, 3.25, 3.6, HEX.reg);
stat(s, `${(g.max_over_median_no_reg || 0).toFixed(1)}×`, "largest token norm vs the median (before the final LayerNorm)", 0.6, 4.95, 3.6, C.accent6);
picture(s, path.join(FIG, "part_a_norm_histograms.png"), 4.5, 1.6, 8.3, 4.9, "norm histograms");
s.addNotes("Our headline number: 2.62% of ViT-g's tokens are above the paper's cutoff, very close to the paper's 2.37%. With registers, zero. The histogram shows the second hump of huge tokens in orange, and it disappears in blue. One detail we found: the norms must be measured before the final LayerNorm, which hides them.");

// ---------------- 6. claims table ----------------
s = content("Six claims checked against the paper", "Reproduction");
const yes = { text: "Yes", options: { bold: true, color: HEX.teal } };
const partly = { text: "Partly", options: { bold: true, color: HEX.noReg } };
const reg = g.register_norms_reg || [0];
const rows = [
  ["Rare high-norm tokens exist", "Fig 3", `${(g.pct_above_paper_cutoff_no_reg || 0).toFixed(2)}% vs 2.37%`, yes],
  ["They sit on redundant background", "Fig 5a", `neighbour similarity ${(g.cosine_outlier || 0).toFixed(3)} vs ${(g.cosine_normal || 0).toFixed(3)}`, yes],
  ["Registers remove them", "Fig 7", `${(g.pct_outliers_no_reg || 0).toFixed(2)}% → ${(g.pct_outliers_reg || 0).toFixed(2)}%`, yes],
  ["Registers clean the attention maps", "Fig 1", "same visual result", yes],
  ["The large values move into registers", "Fig 15", `one register at ${Math.max(...reg).toFixed(0)} vs ${(g.median_reg || 0).toFixed(0)}`, yes],
  ["Outliers only in large models", "Fig 4c", "only in ViT-g (released S/B/L are distilled)", partly],
];
const head = ["Paper claim", "Paper", "Our evidence", "Reproduced?"].map((t) => ({ text: t, options: { bold: true,
  color: C.background1, fill: { color: C.text2 } } }));
s.addTable([head, ...rows.map((r) => r.map((c) => (typeof c === "string" ? { text: c } : c)))], {
  x: 0.6, y: 1.55, w: 12.1, colW: [4.0, 1.2, 5.0, 1.9], fontSize: 15, color: C.text1, rowH: 0.62, valign: "middle",
  border: { type: "solid", pt: 0.5, color: HEX.grid }, fill: { color: C.background1 },
});
s.addNotes("Five of the six diagnostic claims reproduce. The size claim is partial: we only see clear outliers in ViT-g, while the paper also sees them in ViT-L. Meta's released smaller models were distilled from ViT-g, unlike the separately trained models in the paper, which likely explains it.");

// ---------------- 7. our training ----------------
s = content("Our ViT-Tiny: registers barely change accuracy", "Reproduction");
s.addChart(pres.charts.LINE, [
  { name: "0 registers", labels: log0.map((r) => String(r.epoch)), values: log0.map((r) => +(100 * r.val_acc).toFixed(2)) },
  { name: "4 registers", labels: log4.map((r) => String(r.epoch)), values: log4.map((r) => +(100 * r.val_acc).toFixed(2)) },
], {
  x: 0.5, y: 1.45, w: 8.2, h: 5.3, chartColors: [HEX.noReg, HEX.reg], lineSize: 2, lineDataSymbol: "none",
  showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt",
  showTitle: true, title: "Validation accuracy per epoch (%)", titleFontSize: 14, titleFontFace: "+mn-lt", titleColor: HEX.ink,
  catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFrequency: 5,
  valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" }, valAxisMinVal: 0,
});
stat(s, pct(final0.val_acc), "without registers (Tiny-ImageNet, 50 epochs)", 9.1, 1.6, 3.7, HEX.noReg);
stat(s, pct(final4.val_acc), "with 4 registers", 9.1, 3.25, 3.7, HEX.reg);
s.addText(`Gap ${(100 * (final4.val_acc - final0.val_acc)).toFixed(2).replace("-", "−")} points (lr 1e-3); −0.81 when retrained at lr 2.5e-4. The paper reports −0.1 to +0.5. One seed per setting.`,
  { isTextBox: true, x: 9.1, y: 4.95, w: 3.7, h: 1.6, fontSize: 14, color: C.text1, margin: 0, valign: "top" });
s.addNotes("We trained our own ViT-Tiny for 50 epochs on Tiny-ImageNet. Both versions learn the same way and end around 38 to 39 percent, about one point apart. The paper reports almost no change. With one run each we cannot tell if one point is real or noise, so we call this partly reproduced. Absolute accuracy is low because the model is tiny and the images are 64 by 64.");

// ---------------- 8. experiment design ----------------
pres.addSection({ title: "Experiment" });
s = content("Our experiment: do small ViTs get artifacts?", "Experiment");
s.addText("Do high-norm artifacts appear in small ViTs trained from scratch, or only once models are large — and do registers do anything below that size?",
  { isTextBox: true, x: 0.6, y: 1.45, w: 12.1, h: 1.0, fontSize: 20, italic: true, color: C.accent6, margin: 0, valign: "top" });
card(s, 0.6, 2.75, 3.9, 2.6, "Change", "Model size: ViT-Tiny (5.4M), ViT-Small (21.5M)" + (experiment.some((e) => e.size === "base") ? ", ViT-Base (85.4M)" : "") + ", each with 0 and 4 registers.", C.accent2);
card(s, 4.72, 2.75, 3.9, 2.6, "Keep fixed", "Data (Tiny-ImageNet), recipe with lr 2.5e-4 for every size, 50 epochs, seed, patch size. Meta's DINOv2 S/B/L/g give the large end of the axis.", C.accent1);
card(s, 8.84, 2.75, 3.9, 2.6, "Measure", "Token norms after every block, % outliers (> 3× median), how much attention the registers get, attention maps, accuracy.", C.accent3);
s.addNotes("The paper says artifacts only appear in large, long-trained models, but it shows this for one training method on huge data. We can only train small models, so we asked: what happens at the small end? We change only model size, keep everything else fixed, and measure the artifacts in our own trained models, next to Meta's large models from Part A. One lesson on the way: with the learning rate of our first runs, 1e-3, the Base models collapsed after the warm-up, so we retrained all six models at 2.5e-4, which is what the DeiT paper's rule gives for our batch size.");

// ---------------- 9-10. experiment results ----------------
s = content(text.slide_experiment_title || "Result: artifacts are a large-model effect", "Experiment");
picture(s, path.join(FIG, "experiment_scaling.png"), 0.5, 1.4, 8.6, 5.4, "experiment scaling figure");
bullets(s, text.slide_experiment_points || ["[Key findings are added when the runs finish]"], { x: 9.35, y: 1.6, w: 3.4, h: 5.1, fontSize: 15 });
s.addNotes(text.slide_experiment_notes || "Experiment results to be added.");

s = content(text.slide_registers_title || "Inside our models: are the registers used?", "Experiment");
picture(s, path.join(FIG, "experiment_layer_norms.png"), 0.5, 1.4, 6.0, 5.4, "layer norms figure");
picture(s, path.join(FIG, "experiment_accuracy_registers.png"), 6.7, 1.4, 6.1, 3.2, "accuracy and register use");
if (experiment.length) {
  const rowsE = experiment.map((e) => `ViT-${e.size[0].toUpperCase() + e.size.slice(1)}, ${e.registers} reg: `
    + `${e.last_block.pct_outliers.toFixed(2)}% outliers, peak ${Math.max(...e.layers.map((l) => l.max_over_median)).toFixed(2)}×`
    + (e.registers ? `, ${pct(e.attn_patch_to_reg, 1)} attention to registers (uniform ${pct(e.attn_uniform_share, 1)})` : ""));
  s.addText(rowsE.join("\n"), { isTextBox: true, x: 6.7, y: 4.75, w: 6.1, h: 2.0, fontSize: 12, color: C.text1, margin: 0, valign: "top" });
}
s.addNotes(text.slide_registers_notes || "Register-usage results to be added.");

// ---------------- learning-rate lesson ----------------
s = content("A lesson on the way: the learning rate", "Experiment");
picture(s, path.join(FIG, "experiment_learning_rate.png"), 0.5, 1.45, 12.3, 4.0, "learning rate figure");
bullets(s, [
  "At lr 1e-3 (dashed), ViT-Base collapses after the warm-up and ends at 8.6–10.3%; ViT-Small trains worse",
  "At lr 2.5e-4 (solid; DeiT's rule for batch 256), every size trains normally: about 39–40%",
  "We retrained all six models at 2.5e-4 before comparing, and kept the failed runs as a result",
], { x: 0.6, y: 5.6, w: 12.1, h: 1.3, fontSize: 15 });
s.addNotes("One thing that went wrong, and what we learned from it. Our first experiment runs used the learning rate of our earlier ViT-Tiny runs, 1e-3. ViT-Tiny was fine, ViT-Small trained worse, and ViT-Base completely collapsed once the learning rate reached its peak: it ended at around 9 to 10 percent accuracy. If we had measured those broken models, the size comparison would have been meaningless. So we retrained all six models at 2.5e-4, the value the DeiT rule gives for our batch size, and every size trained normally. The failed runs are still reported in our results.");

// ---------------- 11. deployment ----------------
pres.addSection({ title: "Deployment" });
s = content("Deployment: try it on any photo", "Deployment");
picture(s, screenshot, 0.5, 1.4, 7.6, 5.4, "app screenshot");
bullets(s, [
  "FastAPI app: web page, POST /predict, /health, /docs",
  "Our ViT-Tiny (0 and 4 registers) gives the top-5 classes",
  "DINOv2 without vs with registers: attention map, norm map and outlier count (size S/B/L/G)",
  "Here, DINOv2-L: without registers it stares at one sky patch; with registers it looks at the grass",
  "Bad uploads rejected (400 / 413); 5 automated tests",
], { x: 8.4, y: 1.6, w: 4.4, h: 5.1, fontSize: 15 });
s.addNotes("Now the demo. We upload a photo of sky above grass. On the left, DINOv2-Large without registers puts its attention on one random square of empty sky: that is the paper's artifact. On the right, with registers, the attention moves to the grass, the only real content. The norm maps show no outliers at this size, which matches Part A: only the giant model has clear norm outliers. Below, our own ViT-Tiny gives its top-5 classes; sky is not one of its 200 classes, so it guesses scenery like barn and alp. The same thing is available as a JSON API, with documentation at /docs.");

// ---------------- 12. conclusion ----------------
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "Title Dark", sectionTitle: "Conclusion" });
s.addText("What we learned", { placeholder: "title" });
s.addText((text.slide_conclusion || [
  "The paper's diagnosis reproduces on its own models: 2.62% outliers (paper 2.37%), gone with registers",
  "Measure norms before the final LayerNorm, or the artifacts are invisible",
  "Our experiment: [added with the results]",
  "Limits: free GPU, Tiny-ImageNet, one seed; downstream gains (segmentation, LOST) not tested",
]).map((t, i, a) => ({ text: t, options: { bullet: true, breakLine: i < a.length - 1, paraSpaceAfter: 6 } })),
{ placeholder: "body", fontSize: 17 });
s.addNotes("To sum up: the paper's diagnosis holds on its own models, and registers remove the artifacts. We learned that where you measure matters. Our experiment shows when the problem appears. The limits are our compute, the smaller dataset and single seeds. Thank you, we are happy to take questions.");

// pptxgenjs cannot write theme colours itself; apply_theme.js writes THEME into the saved deck.
const { applyTheme } = require("./apply_theme.js");
pres.writeFile({ fileName: OUT }).then(async () => {
  await applyTheme(OUT, THEME);
  console.log(`saved ${OUT}`);
});
