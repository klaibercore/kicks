import type { Metadata } from "next";
import { CorpusAnalysis } from "@/components/analysis/corpus-analysis";

export const metadata: Metadata = { title: "Corpus analysis" };

export default function AnalysisPage() {
  return <CorpusAnalysis />;
}
