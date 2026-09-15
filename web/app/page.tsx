import { ArrowRightIcon } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";

const steps = [
  {
    title: "Perceptual sliders, not knobs",
    body: "Each drum has five axes a listener actually hears — sub, punch, click, brightness, decay for a kick — fitted to a 32-dimensional latent space learned from thousands of one-shots.",
  },
  {
    title: "Rendered by a neural vocoder",
    body: "A convolutional VAE decodes a 128-band log-mel spectrogram; BigVGAN turns it into 44.1 kHz audio. Previews are instant and cached.",
  },
  {
    title: "Judged against the corpus",
    body: "Every render can be scored: robust z-scores per metric, a Mahalanobis likeness percentile, and plain-English verdicts on what sounds off.",
  },
];

export default function HomePage() {
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6">
      <section className="flex flex-col gap-6 py-20 md:py-28">
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-muted-foreground">Neural drum synthesis</p>
        <h1 className="max-w-3xl text-4xl font-semibold tracking-tight md:text-6xl">
          Kicks, snares and hi-hats, drawn from a learned space instead of a sample folder.
        </h1>
        <p className="max-w-2xl text-lg text-muted-foreground">
          Move sliders that mean something, play the result from a MIDI pad, and export 24-bit samples you own outright.
          Previews are free; exports cost one credit each.
        </p>
        <div className="flex flex-wrap gap-3">
          <Button size="lg" render={<Link href="/studio/" />}>
            Open the studio <ArrowRightIcon data-icon="inline-end" />
          </Button>
          <Button size="lg" variant="outline" render={<Link href="/analysis/" />}>
            Explore the corpus
          </Button>
        </div>
      </section>

      <section className="grid gap-8 border-t border-border py-16 md:grid-cols-3">
        {steps.map((s, i) => (
          <div key={s.title} className="flex flex-col gap-2">
            <span className="font-mono text-xs text-muted-foreground">0{i + 1}</span>
            <h2 className="text-base font-medium">{s.title}</h2>
            <p className="text-sm text-muted-foreground">{s.body}</p>
          </div>
        ))}
      </section>

      <section className="border-t border-border py-16">
        <div className="grid gap-6 md:grid-cols-[1fr_2fr] md:items-start">
          <h2 className="text-xl font-medium tracking-tight">Three instruments, one pipeline</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wider text-muted-foreground">
                <tr>
                  <th className="pb-2 pr-4 font-medium">Instrument</th>
                  <th className="pb-2 pr-4 font-medium">Axes</th>
                  <th className="pb-2 font-medium">Character</th>
                </tr>
              </thead>
              <tbody className="[&_td]:py-2 [&_td]:pr-4 [&_tr]:border-t [&_tr]:border-border">
                <tr>
                  <td className="font-mono">kick</td>
                  <td>Sub · Punch · Click · Bright · Decay</td>
                  <td className="text-muted-foreground">Sub-dominant, beater click, long decay</td>
                </tr>
                <tr>
                  <td className="font-mono">snare</td>
                  <td>Body · Crack · Snap · Bright · Decay</td>
                  <td className="text-muted-foreground">Shell tone plus wire rattle</td>
                </tr>
                <tr>
                  <td className="font-mono">hihat</td>
                  <td>Attack · Body · Sizzle · Bright · Decay</td>
                  <td className="text-muted-foreground">Unpitched, top-end, closed to open</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
}
