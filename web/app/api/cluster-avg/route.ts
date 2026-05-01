import { NextResponse } from "next/server";
import fs from "fs/promises";
import { existsSync } from "fs";
import path from "path";

const PROJECT_ROOT = path.resolve(process.cwd(), "..");

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const clusterRaw = searchParams.get("cluster");

    if (clusterRaw === null) {
      return NextResponse.json({ error: "Missing cluster parameter" }, { status: 400 });
    }

    const cluster = parseInt(clusterRaw, 10);
    if (isNaN(cluster)) {
      return NextResponse.json({ error: "Cluster parameter must be an integer" }, { status: 400 });
    }

    const dataPath = path.join(PROJECT_ROOT, "output", "cluster_analysis.json");

    if (!existsSync(dataPath)) {
      return NextResponse.json({ error: "Cluster data not found" }, { status: 404 });
    }

    const raw = await fs.readFile(dataPath, "utf-8");
    const data = JSON.parse(raw);

    if (!data.cluster_averages || !data.cluster_averages[cluster]) {
      return NextResponse.json({ error: "Cluster average not found" }, { status: 404 });
    }

    const avgPath = path.join(PROJECT_ROOT, "output", "samples", `cluster_avg_${cluster}.wav`);

    if (!existsSync(avgPath)) {
      return NextResponse.json({ error: "Cluster average audio not found. Run python cluster.py first" }, { status: 404 });
    }

    const buffer = await fs.readFile(avgPath);
    return new NextResponse(buffer, {
      headers: {
        "Content-Type": "audio/wav",
        "Content-Disposition": `inline; filename="cluster_${cluster}_avg.wav"`,
      },
    });
  } catch (error) {
    console.error("Error loading cluster audio:", error);
    return NextResponse.json({ error: "Failed to load cluster audio" }, { status: 500 });
  }
}
