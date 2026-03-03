import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const PROJECT_ROOT = path.resolve(process.cwd(), "..");

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const cluster = searchParams.get("cluster");
    
    if (cluster === null) {
      return NextResponse.json({ error: "Missing cluster parameter" }, { status: 400 });
    }
    
    const dataPath = path.join(PROJECT_ROOT, "output", "cluster_analysis.json");
    
    if (!fs.existsSync(dataPath)) {
      return NextResponse.json({ error: "Cluster data not found" }, { status: 404 });
    }
    
    const data = JSON.parse(fs.readFileSync(dataPath, "utf-8"));
    const avgKey = `cluster_${cluster}`;
    
    if (!data.cluster_averages || !data.cluster_averages[cluster]) {
      return NextResponse.json({ error: "Cluster average not found" }, { status: 404 });
    }
    
    const avgPath = path.join(PROJECT_ROOT, "output", "samples", `cluster_avg_${cluster}.wav`);
    
    if (!fs.existsSync(avgPath)) {
      return NextResponse.json({ error: "Cluster average audio not found. Run python cluster.py first" }, { status: 404 });
    }
    
    const buffer = fs.readFileSync(avgPath);
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
