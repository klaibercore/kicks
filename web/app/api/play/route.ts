import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const PROJECT_ROOT = path.resolve(process.cwd(), "..");

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const idx = searchParams.get("idx");
    
    if (!idx) {
      return NextResponse.json({ error: "Missing idx parameter" }, { status: 400 });
    }
    
    const dataPath = path.join(PROJECT_ROOT, "output", "cluster_analysis.json");
    
    if (!fs.existsSync(dataPath)) {
      console.log("Data file not found:", dataPath);
      return NextResponse.json({ error: "Cluster data not found" }, { status: 404 });
    }
    
    const data = JSON.parse(fs.readFileSync(dataPath, "utf-8"));
    const sample = data.samples[parseInt(idx)];
    
    if (!sample || !sample.original_path) {
      return NextResponse.json({ error: "Sample not found" }, { status: 404 });
    }
    
    const audioPath = path.join(PROJECT_ROOT, sample.original_path);
    
    if (!fs.existsSync(audioPath)) {
      console.log("Audio not found:", audioPath);
      return NextResponse.json({ error: "Audio file not found" }, { status: 404 });
    }
    
    const buffer = fs.readFileSync(audioPath);
    return new NextResponse(buffer, {
      headers: {
        "Content-Type": "audio/wav",
        "Content-Disposition": `inline; filename="${sample.filename}"`,
      },
    });
  } catch (error) {
    console.error("Error loading sample:", error);
    return NextResponse.json({ error: "Failed to load sample" }, { status: 500 });
  }
}
