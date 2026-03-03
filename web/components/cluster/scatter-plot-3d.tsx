"use client";

import { useState, useMemo } from "react";
import { useTheme } from "next-themes";
import { OrbitControls, Html, useCursor } from "@react-three/drei";
import * as THREE from "three";
import type { Sample, ClusterData, ColorMode, PCName } from "@/types/cluster";
import { sampleColor, PC_COLORS } from "@/lib/colors";

function Point3D({
  coords,
  color,
  isSelected,
  onClick,
  selectionColor = "white",
}: {
  coords: { x: number; y: number; z: number };
  color: string;
  isSelected: boolean;
  onClick: () => void;
  selectionColor?: string;
}) {
  const [hovered, setHovered] = useState(false);
  useCursor(hovered);

  return (
    <group>
      {isSelected && (
        <mesh position={[coords.x, coords.y, coords.z]}>
          <sphereGeometry args={[0.1, 16, 16]} />
          <meshBasicMaterial color={selectionColor} transparent opacity={0.25} />
        </mesh>
      )}
      <mesh
        position={[coords.x, coords.y, coords.z]}
        onClick={onClick}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
        }}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[isSelected ? 0.06 : 0.035, 12, 12]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isSelected ? 0.8 : hovered ? 0.6 : 0.25}
          transparent
          opacity={0.65}
        />
      </mesh>
    </group>
  );
}

function Axes3D({ labels }: { labels: [string, string, string] }) {
  return (
    <group>
      <mesh position={[2, 0, 0]}>
        <boxGeometry args={[4, 0.02, 0.02]} />
        <meshStandardMaterial color="#f472b6" />
      </mesh>
      <mesh position={[0, 2, 0]}>
        <boxGeometry args={[0.02, 4, 0.02]} />
        <meshStandardMaterial color="#34d399" />
      </mesh>
      <mesh position={[0, 0, 2]}>
        <boxGeometry args={[0.02, 0.02, 4]} />
        <meshStandardMaterial color="#a78bfa" />
      </mesh>
      <Html position={[2.2, 0, 0]}>
        <div className="text-xs text-pink-400 font-mono font-medium">
          {labels[0]}
        </div>
      </Html>
      <Html position={[0, 2.2, 0]}>
        <div className="text-xs text-emerald-400 font-mono font-medium">
          {labels[1]}
        </div>
      </Html>
      <Html position={[0, 0, 2.2]}>
        <div className="text-xs text-violet-400 font-mono font-medium">
          {labels[2]}
        </div>
      </Html>
    </group>
  );
}

export function Scene3D({
  data,
  selectedIdx,
  setSelectedIdx,
  colorMode,
  spherize,
  pcNames,
}: {
  data: ClusterData;
  selectedIdx: number | null;
  setSelectedIdx: (idx: number | null) => void;
  colorMode: ColorMode;
  spherize: boolean;
  pcNames: PCName[];
}) {
  const { resolvedTheme } = useTheme();
  const gridColor = useMemo(
    () =>
      new THREE.Color(resolvedTheme === "dark" ? 0xffffff : 0x000000),
    [resolvedTheme],
  );
  const gridOpacity = resolvedTheme === "dark" ? 0.06 : 0.12;

  const flips: [number, number, number] = [
    pcNames[0]?.correlation < 0 ? -1 : 1,
    pcNames[1]?.correlation < 0 ? -1 : 1,
    pcNames[2]?.correlation < 0 ? -1 : 1,
  ];

  const getCoords = useMemo(() => {
    if (!spherize) {
      return (s: Sample) => ({
        x: s.pc1 * flips[0],
        y: s.pc2 * flips[1],
        z: s.pc3 * flips[2],
      });
    }
    const pcs = data.samples.map((s) => [
      s.pc1 * flips[0],
      s.pc2 * flips[1],
      s.pc3 * flips[2],
    ]);
    const means = [0, 1, 2].map(
      (i) => pcs.reduce((a, b) => a + b[i], 0) / pcs.length,
    );
    const stds = [0, 1, 2].map((i) =>
      Math.sqrt(
        pcs.reduce((a, b) => a + (b[i] - means[i]) ** 2, 0) / pcs.length,
      ),
    );
    return (s: Sample) => ({
      x: (s.pc1 * flips[0] - means[0]) / (stds[0] || 1),
      y: (s.pc2 * flips[1] - means[1]) / (stds[1] || 1),
      z: (s.pc3 * flips[2] - means[2]) / (stds[2] || 1),
    });
  }, [data.samples, spherize, flips]);

  const axisLabels: [string, string, string] = [
    pcNames[0]?.name ?? "PC1",
    pcNames[1]?.name ?? "PC2",
    pcNames[2]?.name ?? "PC3",
  ];

  const gridPos: [number, number, number] = spherize
    ? [0, 0, 0]
    : [2, 0, 2];

  const sceneBg = useMemo(
    () =>
      new THREE.Color(resolvedTheme === "dark" ? 0x08080c : 0xf5f5f5),
    [resolvedTheme],
  );

  return (
    <>
      <color attach="background" args={[sceneBg]} key={`bg-${resolvedTheme}`} />
      <ambientLight intensity={resolvedTheme === "dark" ? 0.5 : 0.7} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
      <Axes3D labels={axisLabels} />
      <gridHelper
        key={`grid-${resolvedTheme}-${spherize}`}
        args={[8, 16, gridColor, gridColor]}
        position={gridPos}
        material-transparent
        material-opacity={gridOpacity}
        material-depthWrite={false}
      />
      {data.samples.map((sample) => (
        <Point3D
          key={sample.sample_idx}
          coords={getCoords(sample)}
          color={sampleColor(sample, colorMode, data.descriptor_stats)}
          isSelected={selectedIdx === sample.sample_idx}
          onClick={() =>
            setSelectedIdx(
              selectedIdx === sample.sample_idx ? null : sample.sample_idx,
            )
          }
          selectionColor={resolvedTheme === "dark" ? "white" : "black"}
        />
      ))}
      <OrbitControls enableDamping dampingFactor={0.05} />
    </>
  );
}
