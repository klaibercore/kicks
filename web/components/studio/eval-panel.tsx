"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useStudio } from "@/hooks/use-studio";

const symbolTone: Record<string, string> = {
  "✓": "text-emerald-600 dark:text-emerald-400",
  "⚠": "text-amber-600 dark:text-amber-400",
  "✗": "text-red-600 dark:text-red-400",
};

/** Corpus-referenced verdicts: is this hit inside the distribution of real ones? */
export function EvalPanel() {
  const { evaluation, evaluating, evaluate, sound } = useStudio();

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium">Realism check</h3>
          <p className="text-xs text-muted-foreground">Scores the rendered audio against the training corpus.</p>
        </div>
        <Button size="sm" variant="outline" onClick={() => void evaluate()} disabled={!sound || evaluating}>
          {evaluating ? "Scoring…" : "Evaluate"}
        </Button>
      </div>

      {evaluation?.error ? <p className="text-sm text-destructive">{evaluation.error}</p> : null}

      {evaluation && !evaluation.error ? (
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="font-mono text-base">
              {evaluation.grade}
            </Badge>
            <div className="flex-1">
              <div className="mb-1 flex justify-between text-xs text-muted-foreground">
                <span>Likeness</span>
                <span className="tabular font-mono">{evaluation.likeness_pct?.toFixed(0)}%</span>
              </div>
              <Progress value={evaluation.likeness_pct ?? 0} />
            </div>
          </div>
          <ul className="flex flex-col gap-1.5 text-sm">
            {evaluation.verdicts?.map((v) => (
              <li key={v.metric} className="flex gap-2">
                <span className={`w-4 shrink-0 font-mono ${symbolTone[v.symbol] ?? ""}`}>{v.symbol}</span>
                <span className="text-muted-foreground">{v.text}</span>
              </li>
            ))}
          </ul>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 border-t border-border pt-3 font-mono text-xs sm:grid-cols-3">
            {Object.entries(evaluation.descriptors).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <dt className="text-muted-foreground">{k}</dt>
                <dd className="tabular">{v.toFixed(3)}</dd>
              </div>
            ))}
          </dl>
        </div>
      ) : null}
    </div>
  );
}
