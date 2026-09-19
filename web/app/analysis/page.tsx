import type { Metadata } from "next";
import { CorpusAnalysis } from "@/components/analysis/corpus-analysis";

export const metadata: Metadata = {
  title: "Corpus atlas",
  description:
    "Explore the sound families, perceptual character and statistical structure of the kicks drum corpus in an interactive 3D atlas.",
};

export default function AnalysisPage() {
  return <CorpusAnalysis />;
}
