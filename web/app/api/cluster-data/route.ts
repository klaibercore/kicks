import { NextResponse } from "next/server";
import fs from "fs/promises";
import { existsSync } from "fs";
import path from "path";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const page = Math.max(1, parseInt(searchParams.get("page") || "1", 10) || 1);
    const limit = Math.min(500, Math.max(1, parseInt(searchParams.get("limit") || "250", 10) || 250));

    const dataPath = path.join(process.cwd(), "..", "output", "cluster_analysis.json");

    if (!existsSync(dataPath)) {
      return NextResponse.json({ error: "Cluster data not found. Run python cluster.py first" }, { status: 404 });
    }

    const raw = await fs.readFile(dataPath, "utf-8");
    const data = JSON.parse(raw);

    // Paginate samples
    const allSamples = data.samples || [];
    const total = allSamples.length;
    const startIdx = (page - 1) * limit;
    const paginated = allSamples.slice(startIdx, startIdx + limit);

    return NextResponse.json({
      ...data,
      samples: paginated,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    return NextResponse.json({ error: "Failed to load cluster data" }, { status: 500 });
  }
}
