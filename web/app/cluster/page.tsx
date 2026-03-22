"use client";

import { useEffect, useState, Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import { ThemeToggle } from "@/components/theme-toggle";
import type { ColorMode } from "@/types/cluster";
import { DESCRIPTOR_KEYS } from "@/types/cluster";
import { useClusterData } from "@/hooks/use-cluster-data";
import { useAudioPlayer } from "@/hooks/use-audio";
import { sectionTitle, card, pillActive, pillInactive } from "@/lib/styles";
import { VarianceBar } from "@/components/cluster/variance-bar";
import { PCCard } from "@/components/cluster/pc-card";
import { CorrelationHeatmap } from "@/components/cluster/correlation-heatmap";
import { ScatterPlot } from "@/components/cluster/scatter-plot";
import { Scene3D } from "@/components/cluster/scatter-plot-3d";
import { ClusterProfileCard } from "@/components/cluster/cluster-profile";
import { SampleInspector } from "@/components/cluster/sample-inspector";
import { DescriptorDistribution } from "@/components/cluster/descriptor-distribution";
import { ColorModeSelector } from "@/components/cluster/color-mode-selector";
import { EdaSection } from "@/components/cluster/eda-section";

export default function ClusterPage() {
  const { data, loading, correlations, clusterProfiles, pcNames } =
    useClusterData();
  const {
    audioRef,
    waveform,
    isPlaying,
    playingCluster,
    loadSample,
    clearSample,
    handlePlayPause,
    handlePlayCluster,
    handleAudioEnded,
  } = useAudioPlayer();

  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>("cluster");
  const [view, setView] = useState<"analysis" | "3d">("analysis");
  const [spherize, setSpherize] = useState(false);

  useEffect(() => {
    if (selectedIdx === null || !data) {
      clearSample();
      return;
    }
    loadSample(selectedIdx);
  }, [selectedIdx, data, loadSample, clearSample]);

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex items-center gap-3">
          <div className="w-4 h-4 rounded-full border-2 border-muted-foreground/30 border-t-violet-400 animate-spin" />
          <span className="text-muted-foreground text-sm">
            Loading analysis data...
          </span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center space-y-3">
          <div className="text-foreground/80 font-medium">
            No analysis data found
          </div>
          <code className="text-xs text-muted-foreground bg-muted px-4 py-2 rounded-xl inline-block">
            kicks cluster
          </code>
        </div>
      </div>
    );
  }

  const totalVariance = data.pca_variance_explained.reduce(
    (a, b) => a + b,
    0,
  );
  const selectedSample = data.samples.find(
    (s) => s.sample_idx === selectedIdx,
  );

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border bg-background/90 backdrop-blur-xl sticky top-0 z-20">
        <div className="max-w-[1600px] mx-auto px-3 sm:px-6 py-3 sm:py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2 sm:gap-0">
          <div>
            <h1 className="text-lg sm:text-xl font-bold tracking-tight bg-gradient-to-r from-foreground to-muted-foreground bg-clip-text text-transparent">
              Corpus Analysis
            </h1>
            <div className="flex flex-wrap items-center gap-1.5 mt-1">
              <span className="text-[10px] sm:text-[11px] font-mono text-muted-foreground bg-muted px-1.5 sm:px-2 py-0.5 rounded-md">
                {data.samples.length} samples
              </span>
              <span className="text-[10px] sm:text-[11px] font-mono text-muted-foreground bg-muted px-1.5 sm:px-2 py-0.5 rounded-md">
                {data.n_clusters} clusters
              </span>
              <span className="text-[10px] sm:text-[11px] font-mono text-muted-foreground bg-muted px-1.5 sm:px-2 py-0.5 rounded-md">
                {(totalVariance * 100).toFixed(1)}% var
              </span>
              {data.pca_source === "descriptors_zscore" && (
                <span className="text-[10px] sm:text-[11px] font-mono text-muted-foreground bg-muted px-1.5 sm:px-2 py-0.5 rounded-md">
                  PCA desc
                </span>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex items-center gap-1 bg-card/50 rounded-lg sm:rounded-xl p-0.5 sm:p-1 border border-border">
              <button
                onClick={() => setView("analysis")}
                className={view === "analysis" ? pillActive : pillInactive}
              >
                <span className="hidden sm:inline">Analysis</span>
                <span className="sm:hidden text-[10px]">A</span>
              </button>
              <button
                onClick={() => setView("3d")}
                className={view === "3d" ? pillActive : pillInactive}
              >
                <span className="hidden sm:inline">3D View</span>
                <span className="sm:hidden text-[10px]">3D</span>
              </button>
            </div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {view === "3d" ? (
        <div className="h-[calc(100vh-60px)] sm:h-[calc(100vh-73px)] relative">
          <div className="absolute top-3 left-3 right-3 sm:top-4 sm:left-4 z-10 flex flex-wrap items-center gap-2">
            <ColorModeSelector
              colorMode={colorMode}
              setColorMode={setColorMode}
            />
            <button
              onClick={() => setSpherize(!spherize)}
              className={
                spherize
                  ? pillActive
                  : `${pillInactive} bg-background/40 backdrop-blur-sm border border-border`
              }
            >
              <span className="hidden sm:inline">Spherize</span>
              <span className="sm:hidden text-[10px]">Sph</span>
            </button>
          </div>
          <Canvas camera={{ position: [5, 5, 5], fov: 50 }}>
            <Suspense
              fallback={
                <Html center>
                  <div className="text-muted-foreground text-sm">
                    Loading 3D...
                  </div>
                </Html>
              }
            >
              <Scene3D
                data={data}
                selectedIdx={selectedIdx}
                setSelectedIdx={setSelectedIdx}
                colorMode={colorMode}
                spherize={spherize}
                pcNames={pcNames}
              />
            </Suspense>
          </Canvas>
        </div>
      ) : (
        <main className="max-w-[1600px] mx-auto px-3 sm:px-6 py-5 sm:py-8 space-y-6 sm:space-y-8">
          <EdaSection data={data} />

          <section>
            <VarianceBar
              variance={data.pca_variance_explained}
              pcNames={pcNames}
            />
          </section>

          <section className="space-y-3">
            <h2 className={sectionTitle}>Principal Components</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
              {data.pca_variance_explained.map((v, i) => (
                <PCCard
                  key={i}
                  pcIndex={i}
                  variance={v}
                  correlations={correlations[`pc${i + 1}`] || {}}
                  loadings={data.pca_loadings?.[`pc${i + 1}`]}
                  pcName={
                    pcNames[i] || {
                      name: `PC${i + 1}`,
                      descriptor: null,
                      correlation: 0,
                    }
                  }
                />
              ))}
            </div>
          </section>

          <section>
            <CorrelationHeatmap
              correlations={correlations}
              pcNames={pcNames}
            />
          </section>

          <section className="space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <h2 className={sectionTitle}>PC Space</h2>
              <ColorModeSelector
                colorMode={colorMode}
                setColorMode={setColorMode}
              />
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 sm:gap-4">
              <ScatterPlot
                samples={data.samples}
                xKey="pc1"
                yKey="pc2"
                xLabel={`${pcNames[0]?.name ?? "PC1"} (PC1)`}
                yLabel={`${pcNames[1]?.name ?? "PC2"} (PC2)`}
                colorMode={colorMode}
                stats={data.descriptor_stats}
                selectedIdx={selectedIdx}
                onSelect={(idx) => setSelectedIdx(idx)}
              />
              <ScatterPlot
                samples={data.samples}
                xKey="pc1"
                yKey="pc3"
                xLabel={`${pcNames[0]?.name ?? "PC1"} (PC1)`}
                yLabel={`${pcNames[2]?.name ?? "PC3"} (PC3)`}
                colorMode={colorMode}
                stats={data.descriptor_stats}
                selectedIdx={selectedIdx}
                onSelect={(idx) => setSelectedIdx(idx)}
              />
              <ScatterPlot
                samples={data.samples}
                xKey="pc2"
                yKey="pc3"
                xLabel={`${pcNames[1]?.name ?? "PC2"} (PC2)`}
                yLabel={`${pcNames[2]?.name ?? "PC3"} (PC3)`}
                colorMode={colorMode}
                stats={data.descriptor_stats}
                selectedIdx={selectedIdx}
                onSelect={(idx) => setSelectedIdx(idx)}
              />
            </div>
          </section>

          {selectedSample && (
            <section>
              <SampleInspector
                sample={selectedSample}
                waveform={waveform}
                isPlaying={isPlaying}
                onPlayPause={handlePlayPause}
                onClose={() => {
                  setSelectedIdx(null);
                  clearSample();
                }}
                pcNames={pcNames}
              />
            </section>
          )}

          <section className="space-y-3">
            <h2 className={sectionTitle}>Cluster Profiles</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
              {Object.entries(clusterProfiles)
                .sort(([a], [b]) => parseInt(a) - parseInt(b))
                .map(([k, profile]) => (
                  <ClusterProfileCard
                    key={k}
                    clusterIdx={parseInt(k)}
                    profile={profile}
                    globalStats={data.descriptor_stats}
                    onPlay={() => handlePlayCluster(parseInt(k))}
                    isPlaying={playingCluster === parseInt(k) && isPlaying}
                  />
                ))}
            </div>
          </section>

          <section className="space-y-3">
            <h2 className={sectionTitle}>Descriptor Distributions</h2>
            <div
              className={`grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 sm:gap-5 ${card} p-3 sm:p-5`}
            >
              {DESCRIPTOR_KEYS.map((dk) => (
                <DescriptorDistribution
                  key={dk}
                  samples={data.samples}
                  descriptor={dk}
                  stats={data.descriptor_stats?.[dk]}
                />
              ))}
            </div>
          </section>

          <div className="pt-6 pb-10 text-center text-[11px] text-muted-foreground/60">
            Kick Drum Corpus Analysis
          </div>
        </main>
      )}

      <audio ref={audioRef} className="hidden" onEnded={handleAudioEnded} />
    </div>
  );
}
