import type { Metadata } from "next";
import { StudioLoader } from "@/components/studio/studio-loader";

export const metadata: Metadata = { title: "Studio" };

export default function StudioPage() {
  return <StudioLoader />;
}
