"use client";

import { Latex } from "@/components/ui/latex";

/* ------------------------------------------------------------------ */
/*  Small helpers                                                      */
/* ------------------------------------------------------------------ */

function Section({
  id,
  number,
  title,
  children,
}: {
  id: string;
  number: number;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-24">
      <h2 className="text-2xl font-bold tracking-tight mb-4 flex items-center gap-3">
        <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded">
          {String(number).padStart(2, "0")}
        </span>
        {title}
      </h2>
      <div className="space-y-5">{children}</div>
    </section>
  );
}

function Legend({ rows }: { rows: [string, string, string, string][] }) {
  return (
    <div className="overflow-x-auto my-4">
      <table className="w-full text-sm border border-border rounded-lg overflow-hidden">
        <thead>
          <tr className="bg-muted/50 text-left">
            <th className="px-3 py-2 font-semibold">Symbol</th>
            <th className="px-3 py-2 font-semibold">Type</th>
            <th className="px-3 py-2 font-semibold">Description</th>
            <th className="px-3 py-2 font-semibold">Value / Unit</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([sym, type, desc, val], i) => (
            <tr key={i} className="border-t border-border">
              <td className="px-3 py-2 font-mono">
                <Latex>{sym}</Latex>
              </td>
              <td className="px-3 py-2 text-muted-foreground">{type}</td>
              <td className="px-3 py-2">{desc}</td>
              <td className="px-3 py-2 text-muted-foreground font-mono">
                {val}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Eq({ children }: { children: string }) {
  return (
    <div className="rounded-xl border border-border bg-card/60 backdrop-blur px-6 py-5 my-4 overflow-x-auto">
      <Latex block>{children}</Latex>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  TOC entries                                                        */
/* ------------------------------------------------------------------ */

const TOC = [
  { id: "preprocessing", label: "Audio Preprocessing" },
  { id: "spectrogram", label: "Log-Mel Spectrogram" },
  { id: "vae", label: "VAE Architecture" },
  { id: "loss", label: "Loss Function" },
  { id: "beta", label: "Cyclical Beta Annealing" },
  { id: "pca", label: "PCA & Latent Space" },
  { id: "descriptors", label: "Perceptual Descriptors" },
  { id: "correlation", label: "Pearson Correlation" },
  { id: "gmm", label: "GMM Clustering" },
  { id: "vocoder", label: "Vocoder & Post-Processing" },
];

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

export default function MathPage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-3xl px-5 py-12 sm:py-20 space-y-14">
        {/* ---- Header ---- */}
        <header className="space-y-4">
          <h1 className="text-4xl sm:text-5xl font-black tracking-tighter bg-gradient-to-r from-violet-400 via-pink-400 to-emerald-400 bg-clip-text text-transparent">
            Mathematical Foundations
          </h1>
          <p className="text-muted-foreground leading-relaxed max-w-2xl">
            A complete walkthrough of every equation in the{" "}
            <span className="font-semibold text-foreground">Kicks</span>{" "}
            synthesis pipeline — from raw audio to latent-space exploration
            and neural vocoding.
          </p>
        </header>

        {/* ---- Notation ---- */}
        <div className="rounded-xl border border-border bg-card/60 backdrop-blur px-6 py-5 space-y-2 text-sm">
          <p className="font-semibold mb-2">Notation conventions</p>
          <ul className="list-disc list-inside space-y-1 text-muted-foreground">
            <li>
              Boldface lowercase for vectors:{" "}
              <Latex>{"\\mathbf{z}, \\boldsymbol{\\mu}, \\boldsymbol{\\sigma}, \\boldsymbol{\\varepsilon}"}</Latex>
            </li>
            <li>
              Boldface uppercase for matrices:{" "}
              <Latex>{"\\mathbf{S}, \\hat{\\mathbf{S}}, \\mathbf{W}"}</Latex>
            </li>
            <li>
              <Latex>{"\\odot"}</Latex> denotes element-wise (Hadamard) product
            </li>
            <li>
              <Latex>{"\\|\\cdot\\|_F"}</Latex> denotes the Frobenius norm
            </li>
            <li>
              Dimensions annotated explicitly, e.g.{" "}
              <Latex>{"\\mathbf{z} \\in \\mathbb{R}^d"}</Latex>
            </li>
          </ul>
        </div>

        {/* ---- Table of contents ---- */}
        <nav className="space-y-1">
          <p className="text-xs font-semibold tracking-widest uppercase text-muted-foreground mb-2">
            Contents
          </p>
          <ol className="grid grid-cols-1 sm:grid-cols-2 gap-1 text-sm">
            {TOC.map((t, i) => (
              <li key={t.id}>
                <a
                  href={`#${t.id}`}
                  className="hover:text-primary transition-colors"
                >
                  <span className="text-muted-foreground font-mono mr-2">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  {t.label}
                </a>
              </li>
            ))}
          </ol>
        </nav>

        {/* ================================================================
            SECTION 1 — Audio Preprocessing
        ================================================================ */}
        <Section id="preprocessing" number={1} title="Audio Preprocessing">
          <p className="text-muted-foreground">
            Raw kick drum samples are loaded as WAV files, converted to mono,
            resampled to 44 100 Hz, and padded or truncated to a fixed length
            of 65 536 samples (~1.49 s). Loudness is normalised to a common
            target using the <strong>LUFS</strong> (Loudness Units Full Scale)
            standard.
          </p>

          <p className="font-medium mt-2">Mono conversion</p>
          <Eq>{
            "\\mathbf{x}_{\\text{mono}}[n] = \\frac{1}{C} \\sum_{c=1}^{C} \\mathbf{x}_c[n], \\qquad n = 0, \\ldots, N-1"
          }</Eq>
          <Legend
            rows={[
              ["\\mathbf{x}_c[n]", "Vector", "Sample n of channel c", "—"],
              ["C", "Scalar", "Number of channels", "1 or 2"],
              ["N", "Scalar", "Number of audio samples", "65 536"],
            ]}
          />

          <p className="font-medium mt-2">LUFS loudness normalisation</p>
          <Eq>{
            "\\mathbf{x}_{\\text{norm}} = \\mathbf{x} \\cdot 10^{\\,(L_{\\text{target}} - L_{\\text{measured}})\\,/\\,20}"
          }</Eq>
          <Legend
            rows={[
              ["L_{\\text{target}}", "Scalar", "Target integrated loudness", "\\text{-14 LUFS}"],
              ["L_{\\text{measured}}", "Scalar", "Measured integrated loudness of input", "dB LUFS"],
              ["\\mathbf{x}", "Vector", "Input audio signal", "\\mathbb{R}^N"],
            ]}
          />
        </Section>

        {/* ================================================================
            SECTION 2 — Log-Mel Spectrogram
        ================================================================ */}
        <Section id="spectrogram" number={2} title="Log-Mel Spectrogram">
          <p className="text-muted-foreground">
            The normalised audio is transformed into a log-magnitude mel
            spectrogram using the Short-Time Fourier Transform (STFT),
            a mel filterbank, and a log-amplitude mapping. The representation
            matches the BigVGAN vocoder&apos;s expected input format.
          </p>

          <p className="font-medium mt-2">Short-Time Fourier Transform</p>
          <Eq>{
            "X[m, k] = \\sum_{n=0}^{N_{\\text{FFT}}-1} x[m \\cdot H + n] \\; w[n] \\; e^{-j\\,2\\pi kn / N_{\\text{FFT}}}"
          }</Eq>
          <Legend
            rows={[
              ["X[m, k]", "Matrix", "Complex STFT coefficient at frame m, bin k", "\\mathbb{C}"],
              ["x[n]", "Vector", "Input audio signal", "\\mathbb{R}^N"],
              ["w[n]", "Vector", "Hann window function", "\\mathbb{R}^{N_{\\text{FFT}}}"],
              ["H", "Scalar", "Hop length (frame stride)", "256 samples"],
              ["N_{\\text{FFT}}", "Scalar", "FFT window size", "1024"],
              ["m", "Scalar", "Frame (time) index", "0 \\ldots 255"],
              ["k", "Scalar", "Frequency bin index", "0 \\ldots 512"],
            ]}
          />

          <p className="font-medium mt-2">Mel filterbank projection</p>
          <Eq>{
            "\\mathbf{S}_{\\text{mel}} = \\mathbf{M} \\, |\\mathbf{X}|^2 \\in \\mathbb{R}^{F \\times T}"
          }</Eq>
          <Legend
            rows={[
              ["\\mathbf{M}", "Matrix", "Mel filterbank matrix", "\\mathbb{R}^{F \\times K}"],
              ["|\\mathbf{X}|^2", "Matrix", "Magnitude-squared STFT", "\\mathbb{R}^{K \\times T}"],
              ["F", "Scalar", "Number of mel bands", "128"],
              ["T", "Scalar", "Number of time frames", "256"],
              ["K", "Scalar", "Number of FFT bins", "513"],
            ]}
          />

          <p className="font-medium mt-2">Log-amplitude with clamp</p>
          <Eq>{
            "\\mathbf{S}_{\\text{log}} = \\ln\\!\\bigl(\\max(\\mathbf{S}_{\\text{mel}},\\; \\epsilon)\\bigr), \\qquad \\epsilon = 10^{-5}"
          }</Eq>

          <p className="font-medium mt-2">
            Fixed-bounds normalisation to [0, 1]
          </p>
          <Eq>{
            "\\hat{\\mathbf{S}} = \\frac{\\operatorname{clamp}(\\mathbf{S}_{\\text{log}},\\; s_{\\min},\\; s_{\\max}) \\;-\\; s_{\\min}}{s_{\\max} - s_{\\min}}"
          }</Eq>
          <Legend
            rows={[
              ["s_{\\min}", "Scalar", "Silence floor = ln(10^{-5})", "-11.5129"],
              ["s_{\\max}", "Scalar", "Headroom ceiling", "2.5"],
              ["\\hat{\\mathbf{S}}", "Matrix", "Normalised spectrogram (VAE input)", "[0, 1]^{1 \\times 128 \\times 256}"],
            ]}
          />
        </Section>

        {/* ================================================================
            SECTION 3 — VAE Architecture
        ================================================================ */}
        <Section id="vae" number={3} title="VAE Architecture">
          <p className="text-muted-foreground">
            A 2-D Convolutional Variational Autoencoder maps spectrograms into
            a compact latent space and back. The encoder applies four
            stride-2 convolutions to downsample the input by a factor of 16
            in each spatial dimension, then projects to a mean and
            log-variance vector. The decoder mirrors this structure using
            transposed convolutions.
          </p>

          <p className="font-medium mt-2">Encoder</p>
          <Eq>{
            "\\mathbf{h} = \\operatorname{flatten}\\!\\Big(\\operatorname{Enc}\\bigl(\\hat{\\mathbf{S}}\\bigr)\\Big) \\in \\mathbb{R}^{32\\,768}"
          }</Eq>
          <Eq>{
            "\\boldsymbol{\\mu} = \\mathbf{W}_\\mu \\, \\mathbf{h} + \\mathbf{b}_\\mu \\in \\mathbb{R}^d, \\qquad \\log \\boldsymbol{\\sigma}^2 = \\operatorname{clamp}\\!\\left(\\mathbf{W}_\\sigma \\, \\mathbf{h} + \\mathbf{b}_\\sigma,\\; -10,\\; 10\\right) \\in \\mathbb{R}^d"
          }</Eq>
          <Legend
            rows={[
              ["\\hat{\\mathbf{S}}", "Matrix", "Normalised spectrogram input", "\\mathbb{R}^{1 \\times 128 \\times 256}"],
              ["\\operatorname{Enc}", "Function", "4\\times stride-2 Conv2d + BN + ReLU", "(1{\\to}32{\\to}64{\\to}128{\\to}256)"],
              ["\\mathbf{h}", "Vector", "Flattened encoder output (256 \\times 8 \\times 16)", "\\mathbb{R}^{32\\,768}"],
              ["\\boldsymbol{\\mu}", "Vector", "Latent mean", "\\mathbb{R}^d"],
              ["\\log \\boldsymbol{\\sigma}^2", "Vector", "Latent log-variance (clamped)", "\\mathbb{R}^d"],
              ["d", "Scalar", "Latent dimensionality", "128"],
            ]}
          />

          <p className="font-medium mt-2">Reparameterisation trick</p>
          <Eq>{
            "\\mathbf{z} = \\boldsymbol{\\mu} + \\boldsymbol{\\sigma} \\odot \\boldsymbol{\\varepsilon}, \\qquad \\boldsymbol{\\varepsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I}_d)"
          }</Eq>
          <Eq>{
            "\\boldsymbol{\\sigma} = \\exp\\!\\left(\\tfrac{1}{2} \\log \\boldsymbol{\\sigma}^2\\right)"
          }</Eq>
          <Legend
            rows={[
              ["\\mathbf{z}", "Vector", "Sampled latent code", "\\mathbb{R}^d"],
              ["\\boldsymbol{\\varepsilon}", "Vector", "Standard normal noise", "\\mathbb{R}^d"],
              ["\\odot", "Operator", "Element-wise (Hadamard) product", "—"],
            ]}
          />

          <p className="font-medium mt-2">Decoder</p>
          <Eq>{
            "\\tilde{\\mathbf{S}} = \\sigma\\!\\Big(\\operatorname{Dec}\\bigl(\\operatorname{reshape}(\\mathbf{W}_z \\mathbf{z} + \\mathbf{b}_z)\\bigr)\\Big) \\in [0, 1]^{1 \\times 128 \\times 256}"
          }</Eq>
          <Legend
            rows={[
              ["\\tilde{\\mathbf{S}}", "Matrix", "Reconstructed spectrogram", "[0,1]^{1 \\times 128 \\times 256}"],
              ["\\operatorname{Dec}", "Function", "4\\times stride-2 ConvTranspose2d + BN + ReLU", "(256{\\to}128{\\to}64{\\to}32{\\to}1)"],
              ["\\sigma(\\cdot)", "Function", "Sigmoid activation", "[0,1]"],
            ]}
          />
        </Section>

        {/* ================================================================
            SECTION 4 — Loss Function
        ================================================================ */}
        <Section id="loss" number={4} title="Loss Function">
          <p className="text-muted-foreground">
            The training objective is a weighted sum of a multi-resolution
            reconstruction loss (capturing both fine transient detail and
            global spectral shape) and a KL divergence term that regularises
            the latent space.
          </p>

          <p className="font-medium mt-2">Spectral convergence loss</p>
          <Eq>{
            "\\mathcal{L}_{\\text{SC}}(\\mathbf{S}, \\hat{\\mathbf{S}}) = \\frac{\\|\\mathbf{S} - \\hat{\\mathbf{S}}\\|_F}{\\|\\mathbf{S}\\|_F + \\epsilon}"
          }</Eq>
          <Legend
            rows={[
              ["\\|\\cdot\\|_F", "Operator", "Frobenius norm: \\sqrt{\\sum_{i,j} a_{ij}^2}", "—"],
              ["\\mathbf{S}", "Matrix", "Target spectrogram", "\\mathbb{R}^{F \\times T}"],
              ["\\hat{\\mathbf{S}}", "Matrix", "Reconstructed spectrogram", "\\mathbb{R}^{F \\times T}"],
              ["\\epsilon", "Scalar", "Numerical stability constant", "10^{-8}"],
            ]}
          />

          <p className="font-medium mt-2">Frequency-weighted L1 loss</p>
          <Eq>{
            "\\mathcal{L}_{\\ell_1}(\\mathbf{S}, \\hat{\\mathbf{S}}) = \\frac{1}{FT}\\sum_{f,t} w_f \\;\\bigl|S_{f,t} - \\hat{S}_{f,t}\\bigr|"
          }</Eq>
          <Eq>{
            "w_f = 1 - \\alpha \\cdot \\frac{f}{F - 1}, \\qquad f = 0, \\ldots, F{-}1"
          }</Eq>
          <Legend
            rows={[
              ["w_f", "Vector", "Linearly decaying frequency weight at bin f", "[1-\\alpha,\\; 1]"],
              ["\\alpha", "Scalar", "Frequency-weight decay factor", "0.5"],
              ["F", "Scalar", "Number of mel bands", "128"],
              ["T", "Scalar", "Number of time frames", "256"],
            ]}
          />

          <p className="font-medium mt-2">Multi-resolution aggregation</p>
          <Eq>{
            "\\mathcal{L}_{\\text{recon}} = \\sum_{s \\in \\{1, 2, 4\\}} \\left[\\mathcal{L}_{\\text{SC}}\\!\\left(\\operatorname{pool}_s(\\mathbf{S}),\\, \\operatorname{pool}_s(\\hat{\\mathbf{S}})\\right) + \\mathcal{L}_{\\ell_1}\\!\\left(\\operatorname{pool}_s(\\mathbf{S}),\\, \\operatorname{pool}_s(\\hat{\\mathbf{S}})\\right)\\right]"
          }</Eq>
          <Legend
            rows={[
              ["\\operatorname{pool}_s", "Function", "2-D average pooling with kernel size s", "—"],
              ["s", "Scalar", "Resolution scale factor", "\\{1, 2, 4\\}"],
            ]}
          />

          <p className="font-medium mt-2">KL divergence</p>
          <Eq>{
            "D_{\\text{KL}}\\!\\left(q(\\mathbf{z}|\\mathbf{x}) \\,\\|\\, p(\\mathbf{z})\\right) = -\\frac{1}{2B}\\sum_{b=1}^{B}\\sum_{i=1}^{d} \\left(1 + \\log \\sigma_{b,i}^2 - \\mu_{b,i}^2 - \\sigma_{b,i}^2\\right)"
          }</Eq>
          <Legend
            rows={[
              ["q(\\mathbf{z}|\\mathbf{x})", "Distribution", "Approximate posterior (encoder output)", "\\mathcal{N}(\\boldsymbol{\\mu}, \\operatorname{diag}(\\boldsymbol{\\sigma}^2))"],
              ["p(\\mathbf{z})", "Distribution", "Prior", "\\mathcal{N}(\\mathbf{0}, \\mathbf{I}_d)"],
              ["B", "Scalar", "Batch size", "—"],
              ["d", "Scalar", "Latent dimensionality", "128"],
            ]}
          />

          <p className="font-medium mt-2">Total loss</p>
          <Eq>{
            "\\mathcal{L} = \\mathcal{L}_{\\text{recon}} + \\beta \\cdot D_{\\text{KL}}"
          }</Eq>
          <Legend
            rows={[
              ["\\beta", "Scalar", "KL weight (beta-VAE trade-off)", "0.01 (default)"],
            ]}
          />
        </Section>

        {/* ================================================================
            SECTION 5 — Cyclical Beta Annealing
        ================================================================ */}
        <Section id="beta" number={5} title="Cyclical Beta Annealing">
          <p className="text-muted-foreground">
            To prevent posterior collapse (where the encoder ignores the
            latent code), the KL weight <Latex>{`\\beta`}</Latex> is annealed
            cyclically. Within each cycle the weight ramps linearly from 0 to{" "}
            <Latex>{`\\beta_{\\max}`}</Latex> over the first half, then holds
            constant for the second half.
          </p>

          <Eq>{
            "T_c = \\frac{E_{\\text{anneal}}}{K}"
          }</Eq>
          <Eq>{
            "\\phi(e) = \\frac{e \\bmod T_c}{T_c}"
          }</Eq>
          <Eq>{
            "\\beta(e) = \\beta_{\\max} \\cdot \\min\\!\\bigl(1,\\; 2\\,\\phi(e)\\bigr)"
          }</Eq>
          <Legend
            rows={[
              ["e", "Scalar", "Current epoch", "0 \\ldots E-1"],
              ["E_{\\text{anneal}}", "Scalar", "Number of epochs for annealing", "200"],
              ["K", "Scalar", "Number of annealing cycles", "4"],
              ["T_c", "Scalar", "Epochs per cycle", "E_{\\text{anneal}} / K"],
              ["\\phi(e)", "Scalar", "Position within current cycle", "[0, 1)"],
              ["\\beta_{\\max}", "Scalar", "Maximum KL weight", "0.01"],
            ]}
          />
        </Section>

        {/* ================================================================
            SECTION 6 — PCA & Latent Space
        ================================================================ */}
        <Section id="pca" number={6} title="PCA & Latent Space">
          <p className="text-muted-foreground">
            After training, every sample&apos;s latent mean vector{" "}
            <Latex>{"\\boldsymbol{\\mu}"}</Latex> is extracted. Principal
            Component Analysis reduces the 128-dimensional space to{" "}
            <Latex>{"P = 4"}</Latex> components that capture the most
            variance — these become the user-facing synthesis sliders.
          </p>

          <p className="font-medium mt-2">Eigen-decomposition of covariance</p>
          <Eq>{
            "\\mathbf{C} = \\frac{1}{N-1} \\sum_{n=1}^{N} (\\boldsymbol{\\mu}_n - \\bar{\\boldsymbol{\\mu}})(\\boldsymbol{\\mu}_n - \\bar{\\boldsymbol{\\mu}})^\\top \\in \\mathbb{R}^{d \\times d}"
          }</Eq>
          <Eq>{
            "\\mathbf{C} = \\mathbf{V} \\boldsymbol{\\Lambda} \\mathbf{V}^\\top"
          }</Eq>
          <Legend
            rows={[
              ["\\mathbf{C}", "Matrix", "Sample covariance of latent means", "\\mathbb{R}^{d \\times d}"],
              ["\\bar{\\boldsymbol{\\mu}}", "Vector", "Mean of all latent vectors", "\\mathbb{R}^d"],
              ["\\mathbf{V}", "Matrix", "Eigenvectors (principal directions)", "\\mathbb{R}^{d \\times d}"],
              ["\\boldsymbol{\\Lambda}", "Matrix", "Diagonal eigenvalue matrix", "\\mathbb{R}^{d \\times d}"],
              ["N", "Scalar", "Number of training samples", "—"],
              ["d", "Scalar", "Latent dimensionality", "128"],
            ]}
          />

          <p className="font-medium mt-2">Projection to P components</p>
          <Eq>{
            "\\mathbf{p}_n = \\mathbf{V}_P^\\top (\\boldsymbol{\\mu}_n - \\bar{\\boldsymbol{\\mu}}) \\in \\mathbb{R}^P"
          }</Eq>
          <Legend
            rows={[
              ["\\mathbf{V}_P", "Matrix", "First P principal component vectors", "\\mathbb{R}^{d \\times P}"],
              ["\\mathbf{p}_n", "Vector", "PC-projected coordinates for sample n", "\\mathbb{R}^P"],
              ["P", "Scalar", "Number of retained components", "4"],
            ]}
          />

          <p className="font-medium mt-2">Explained variance ratio</p>
          <Eq>{
            "\\rho_i = \\frac{\\lambda_i}{\\sum_{j=1}^{d} \\lambda_j}, \\qquad i = 1, \\ldots, P"
          }</Eq>

          <p className="font-medium mt-2">Slider mapping (percentile-based)</p>
          <Eq>{
            "\\text{pc}_i = p_i^{(2)} + u_i \\cdot \\bigl(p_i^{(98)} - p_i^{(2)}\\bigr), \\qquad u_i \\in [0, 1]"
          }</Eq>
          <Legend
            rows={[
              ["u_i", "Scalar", "User slider value for component i", "[0, 1]"],
              ["p_i^{(2)}, p_i^{(98)}", "Scalar", "2nd and 98th percentile of projected PC_i", "—"],
            ]}
          />

          <p className="font-medium mt-2">Inverse PCA (back to latent space)</p>
          <Eq>{
            "\\mathbf{z} = \\mathbf{V}_P \\, \\mathbf{p} + \\bar{\\boldsymbol{\\mu}} \\in \\mathbb{R}^d"
          }</Eq>
        </Section>

        {/* ================================================================
            SECTION 7 — Perceptual Descriptors
        ================================================================ */}
        <Section id="descriptors" number={7} title="Perceptual Descriptors">
          <p className="text-muted-foreground">
            Five perceptual features are computed directly from each
            normalised spectrogram{" "}
            <Latex>{"\\hat{\\mathbf{S}} \\in [0,1]^{128 \\times 256}"}</Latex>.
            These are used to auto-name the PCA sliders by correlating them
            with each principal component. Frames 0–2 are the transient
            region (&lt;17 ms); frames 3+ are the body/tail.
          </p>

          <p className="font-medium mt-2">Sub-bass energy</p>
          <Eq>{
            "\\text{Sub} = \\frac{1}{12 \\cdot (T-3)} \\sum_{f=0}^{11} \\sum_{t=3}^{T-1} \\hat{S}_{f,t}"
          }</Eq>
          <Legend
            rows={[
              ["f \\in [0, 11]", "Index", "Mel bands 0–11 (\\approx 20{-}80\\text{ Hz})", "Sub-bass region"],
              ["t \\in [3, T{-}1]", "Index", "Body + tail frames (excl. transient)", "\\approx 17\\text{ ms onward}"],
            ]}
          />

          <p className="font-medium mt-2">Punch (transient-to-body ratio)</p>
          <Eq>{
            "E_{\\text{attack}} = \\frac{1}{20 \\cdot 3} \\sum_{f=10}^{29} \\sum_{t=0}^{2} \\hat{S}_{f,t}"
          }</Eq>
          <Eq>{
            "E_{\\text{body}} = \\frac{1}{20 \\cdot 27} \\sum_{f=10}^{29} \\sum_{t=3}^{29} \\hat{S}_{f,t}"
          }</Eq>
          <Eq>{
            "\\text{Punch} = \\min\\!\\left(\\frac{E_{\\text{attack}}}{E_{\\text{body}} + \\epsilon},\\; 1\\right)"
          }</Eq>
          <Legend
            rows={[
              ["f \\in [10, 29]", "Index", "Mel bands 10–29 (\\approx 80{-}250\\text{ Hz})", "Bass region"],
              ["E_{\\text{attack}}", "Scalar", "Mean energy in transient (first 3 frames)", "—"],
              ["E_{\\text{body}}", "Scalar", "Mean energy in body (frames 3–29)", "—"],
            ]}
          />

          <p className="font-medium mt-2">Click (high-frequency transient)</p>
          <Eq>{
            "\\text{Click} = \\frac{1}{60 \\cdot 3} \\sum_{f=40}^{99} \\sum_{t=0}^{2} \\hat{S}_{f,t}"
          }</Eq>
          <Legend
            rows={[
              ["f \\in [40, 99]", "Index", "Mel bands 40–99 (\\approx 1{-}8\\text{ kHz})", "High-mid / high region"],
            ]}
          />

          <p className="font-medium mt-2">Brightness</p>
          <Eq>{
            "\\text{Bright} = \\frac{\\displaystyle\\sum_{f=50}^{127} \\sum_{t=3}^{T-1} \\hat{S}_{f,t}}{\\displaystyle\\sum_{f=0}^{29} \\sum_{t=3}^{T-1} \\hat{S}_{f,t} \\;+\\; \\sum_{f=50}^{127} \\sum_{t=3}^{T-1} \\hat{S}_{f,t} \\;+\\; \\epsilon}"
          }</Eq>

          <p className="font-medium mt-2">Decay (broadband early-to-late ratio)</p>
          <p className="text-muted-foreground text-sm mb-2">
            Compares the mean energy in the tail (~174-700 ms) to the body
            (~17-174 ms) across a broad frequency range (bands 0-40,
            ~20-350 Hz). The &ldquo;1 - ratio&rdquo; formulation spreads
            naturally across [0, 1]: higher values mean tighter/faster
            decay, lower values mean longer sustain.
          </p>
          <Eq>{
            "E_{\text{body}} = \frac{1}{40 \cdot 27} \sum_{f=0}^{39} \sum_{t=3}^{29} \hat{S}_{f,t}"
          }</Eq>
          <Eq>{
            "E_{\text{tail}} = \frac{1}{40 \cdot 90} \sum_{f=0}^{39} \sum_{t=30}^{119} \hat{S}_{f,t}"
          }</Eq>
          <Eq>{
            "\text{Decay} = 1 - \operatorname{clip}\!\left(\frac{E_{\text{tail}}}{E_{\text{body}} + \epsilon},\; 0,\; 1\right)"
          }</Eq>
          <Legend
            rows={[
              ["\hat{S}_{f,t}", "Matrix", "Normalized log-mel spectrogram", "(128 \times 256)"],
              ["f \in [0, 39]", "Index", "Mel bands 0-39 (\approx 20{-}350\text{ Hz})", "Sub through low-mids"],
              ["t \in [3, 29]", "Index", "Body frames (\approx 17{-}174\text{ ms})", "Excludes transient"],
              ["t \in [30, 119]", "Index", "Tail frames (\approx 174{-}700\text{ ms})", "Sustain tail"],
            ]}
          />

        {/* ================================================================
            SECTION 8 — Pearson Correlation
        ================================================================ */}
        <Section id="correlation" number={8} title="Pearson Correlation (PC Auto-Naming)">
          <p className="text-muted-foreground">
            Each principal component is correlated with every perceptual
            descriptor. The descriptor with the highest absolute correlation
            (above threshold 0.15) names the slider. If the correlation is
            negative, the PC axis is flipped for intuitive control.
          </p>

          <Eq>{
            "r_{ij} = \\frac{\\displaystyle\\frac{1}{N}\\sum_{n=1}^{N}(p_{n,i} - \\bar{p}_i)(d_{n,j} - \\bar{d}_j)}{s_{p_i} \\cdot s_{d_j}}"
          }</Eq>
          <Eq>{
            "\\text{name}_i = \\arg\\max_{j} |r_{ij}| \\quad \\text{subject to} \\quad |r_{ij}| \\geq 0.15"
          }</Eq>
          <Legend
            rows={[
              ["r_{ij}", "Scalar", "Pearson correlation between PC_i and descriptor_j", "[-1, 1]"],
              ["p_{n,i}", "Scalar", "PC_i value for sample n", "—"],
              ["d_{n,j}", "Scalar", "Descriptor_j value for sample n", "—"],
              ["\\bar{p}_i,\\; s_{p_i}", "Scalar", "Mean and std of PC_i across all samples", "—"],
              ["\\bar{d}_j,\\; s_{d_j}", "Scalar", "Mean and std of descriptor_j across all samples", "—"],
              ["N", "Scalar", "Number of training samples", "—"],
            ]}
          />
        </Section>

        {/* ================================================================
            SECTION 9 — GMM Clustering
        ================================================================ */}
        <Section id="gmm" number={9} title="GMM Clustering">
          <p className="text-muted-foreground">
            A Gaussian Mixture Model clusters the latent vectors into
            perceptually meaningful groups. The optimal number of clusters is
            selected by minimising the Bayesian Information Criterion.
          </p>

          <p className="font-medium mt-2">Gaussian Mixture Model</p>
          <Eq>{
            "p(\\mathbf{z}) = \\sum_{k=1}^{K} \\pi_k \\;\\mathcal{N}\\!\\left(\\mathbf{z} \\;\\middle|\\; \\boldsymbol{\\mu}_k,\\, \\boldsymbol{\\Sigma}_k\\right)"
          }</Eq>
          <Legend
            rows={[
              ["K", "Scalar", "Number of mixture components", "2 \\ldots 10"],
              ["\\pi_k", "Scalar", "Mixing weight for component k", "\\sum_k \\pi_k = 1"],
              ["\\boldsymbol{\\mu}_k", "Vector", "Mean of component k", "\\mathbb{R}^d"],
              ["\\boldsymbol{\\Sigma}_k", "Matrix", "Full covariance of component k", "\\mathbb{R}^{d \\times d}"],
            ]}
          />

          <p className="font-medium mt-2">Bayesian Information Criterion</p>
          <Eq>{
            "\\text{BIC}(K) = -2 \\ln \\hat{L}(K) + m_K \\ln N"
          }</Eq>
          <Eq>{
            "K^* = \\arg\\min_K \\;\\text{BIC}(K)"
          }</Eq>
          <Legend
            rows={[
              ["\\hat{L}(K)", "Scalar", "Maximised log-likelihood of the K-component model", "—"],
              ["m_K", "Scalar", "Number of free parameters in the K-component model", "—"],
              ["N", "Scalar", "Number of data points", "—"],
              ["K^*", "Scalar", "Optimal number of clusters", "—"],
            ]}
          />
        </Section>

        {/* ================================================================
            SECTION 10 — Vocoder & Post-Processing
        ================================================================ */}
        <Section id="vocoder" number={10} title="Vocoder & Post-Processing">
          <p className="text-muted-foreground">
            The VAE decoder output is converted back to a waveform by the
            BigVGAN neural vocoder, then cleaned up with filtering and
            normalisation.
          </p>

          <p className="font-medium mt-2">Denormalisation</p>
          <Eq>{
            "\\mathbf{S}_{\\text{log}} = \\tilde{\\mathbf{S}} \\cdot (s_{\\max} - s_{\\min}) + s_{\\min}"
          }</Eq>

          <p className="font-medium mt-2">Silence gating</p>
          <Eq>{
            "S_{f,t}^{\\prime} = \\begin{cases} s_{\\min} & \\text{if } S_{f,t} < s_{\\min} + \\delta \\\\ S_{f,t} & \\text{otherwise} \\end{cases}"
          }</Eq>
          <Legend
            rows={[
              ["\\delta", "Scalar", "Gating margin above silence floor", "2.0 \\text{ nats}"],
              ["s_{\\min}", "Scalar", "Silence floor", "-11.5129"],
            ]}
          />

          <p className="font-medium mt-2">BigVGAN neural vocoder</p>
          <Eq>{
            "\\mathbf{x} = \\operatorname{BigVGAN}\\!\\left(\\mathbf{S}^{\\prime}\\right) \\in \\mathbb{R}^T"
          }</Eq>

          <p className="font-medium mt-2">Biquad filters</p>
          <p className="text-muted-foreground text-sm mb-2">
            A high-pass filter at 25 Hz removes subsonic rumble; a low-pass
            at 20 kHz removes aliasing artefacts.
          </p>
          <Eq>{
            "H_{\\text{HP}}(z) : f_c = 25\\text{ Hz}, \\qquad H_{\\text{LP}}(z) : f_c = 20\\,000\\text{ Hz}"
          }</Eq>
          <Eq>{
            "\\mathbf{x}_{\\text{filt}} = H_{\\text{LP}}\\!\\bigl(H_{\\text{HP}}(\\mathbf{x})\\bigr)"
          }</Eq>

          <p className="font-medium mt-2">Fade-out</p>
          <Eq>{
            "x_{\\text{out}}[n] = \\begin{cases} x_{\\text{filt}}[n] & n < N - L \\\\ x_{\\text{filt}}[n] \\cdot \\dfrac{N - n}{L} & n \\geq N - L \\end{cases}"
          }</Eq>
          <Legend
            rows={[
              ["L", "Scalar", "Fade length", "5000 \\text{ samples}"],
              ["N", "Scalar", "Total waveform length", "65\\,536"],
            ]}
          />

          <p className="font-medium mt-2">Peak normalisation</p>
          <Eq>{
            "\\mathbf{x}_{\\text{final}} = \\frac{\\mathbf{x}_{\\text{out}}}{\\|\\mathbf{x}_{\\text{out}}\\|_\\infty + \\epsilon}"
          }</Eq>
          <Legend
            rows={[
              ["\\|\\cdot\\|_\\infty", "Operator", "L^\\infty norm (max absolute value)", "—"],
              ["\\epsilon", "Scalar", "Numerical stability constant", "10^{-8}"],
            ]}
          />
        </Section>

        {/* ---- Footer ---- */}
        <footer className="pt-8 border-t border-border text-center text-xs text-muted-foreground/50">
          &copy; {new Date().getFullYear()} Kevin Paul Klaiber
        </footer>
      </div>
    </div>
  );
}
