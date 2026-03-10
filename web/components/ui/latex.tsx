"use client";

import katex from "katex";
import "katex/dist/katex.min.css";

interface LatexProps {
  /** LaTeX math string (without delimiters) */
  children: string;
  /** Render as block (display) math. Default: false (inline). */
  block?: boolean;
  className?: string;
}

export function Latex({ children, block = false, className }: LatexProps) {
  const html = katex.renderToString(children, {
    displayMode: block,
    throwOnError: false,
    strict: false,
  });

  return block ? (
    <div
      className={className ?? "my-4 overflow-x-auto text-lg"}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  ) : (
    <span
      className={className}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
