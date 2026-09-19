import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { clusterName, clusterTraits, descriptorUnit, histogram } from "../lib/analysis/stats.ts";

test("histograms retain negative dB values and millisecond measurements", () => {
  assert.deepEqual(histogram([-60, -40, -20, 0], 3, -60, 0).map((b) => b.count), [1, 1, 2]);
  assert.deepEqual(histogram([10, 100, 200, 300], 3, 0, 300).map((b) => b.count), [1, 1, 2]);
  assert.equal(histogram([4, 4, NaN], 4, 4, 4).reduce((sum, b) => sum + b.count, 0), 2);
});

for (const instrument of ["kick", "snare", "hihat"]) {
  test(`${instrument}: native ranges account for every sample and all family names have numerical support`, () => {
    const report = JSON.parse(readFileSync(new URL(`../public/analysis/${instrument}.json`, import.meta.url)));
    for (const key of report.descriptor_keys) {
      const { min, max } = report.descriptor_stats[key];
      const bins = histogram(report.samples.map((s) => s.descriptors[key]), 28, min, max);
      assert.equal(bins.reduce((sum, b) => sum + b.count, 0), report.samples.length);
      assert.ok(bins.filter((b) => b.count).length > 2, `${key} distribution collapsed`);
      assert.equal(descriptorUnit(key, report), key === "decay" ? "ms" : "dB");
    }
    for (const cluster of Object.keys(report.cluster_profiles).map(Number)) {
      const traits = clusterTraits(report, cluster);
      assert.ok(traits.every((t) => Number.isFinite(t.z)));
      assert.equal(clusterName(report, cluster) === "Near corpus average", traits.every((t) => Math.abs(t.z) < .25));
    }
  });
}
