import { config } from "@/lib/config";
import type {
  Backend,
  Evaluation,
  Health,
  InstrumentConfig,
  InstrumentsResponse,
  Sound,
  Spectrogram,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

/** Query string for a sound. Keys are sorted so equal sounds hit the API's cache. */
export function soundQuery(sound: Sound): string {
  const params = new URLSearchParams();
  params.set("instrument", sound.instrument);
  params.set("backend", sound.backend ?? "vae");
  if (sound.backend === "diffusion") {
    params.set("backend", "diffusion");
    if (sound.steps) params.set("steps", String(sound.steps));
  }
  if (sound.seed) params.set("seed", String(sound.seed));
  for (const key of Object.keys(sound.sliders).sort()) {
    params.set(key, sound.sliders[key].toFixed(3));
  }
  for (const [key, value] of Object.entries(sound.effects)) {
    if (value !== undefined && value !== null) params.set(key, String(value));
  }
  return params.toString();
}

export class KicksApi {
  constructor(
    private readonly baseUrl: string = config.apiUrl,
  ) {}

  private async request(path: string, init: RequestInit = {}): Promise<Response> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: init.headers,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail ?? detail;
      } catch {
        /* not JSON */
      }
      throw new ApiError(res.status, detail);
    }
    return res;
  }

  private async json<T>(path: string): Promise<T> {
    return (await this.request(path)).json() as Promise<T>;
  }

  health() {
    return this.json<Health>("/health");
  }

  instruments() {
    return this.json<InstrumentsResponse>("/instruments");
  }

  config(instrument: string, backend: Backend = "vae") {
    const query = `instrument=${encodeURIComponent(instrument)}&backend=${backend}`;
    return this.json<InstrumentConfig>(`/config?${query}`);
  }

  /** Preview render: raw WAV bytes. Free, cached server-side. Aborting drops a queued diffusion render. */
  async generate(sound: Sound, signal?: AbortSignal): Promise<ArrayBuffer> {
    const res = await this.request(`/generate?${soundQuery(sound)}`, { signal });
    return res.arrayBuffer();
  }

  evaluate(sound: Sound) {
    return this.json<Evaluation>(`/evaluate?${soundQuery(sound)}`);
  }

  spectrogram(sound: Sound) {
    return this.json<Spectrogram>(`/spectrogram?${soundQuery(sound)}`);
  }

  /** Download the rendered sound as a 24-bit WAV. */
  async exportSample(sound: Sound): Promise<ExportResult> {
    const res = await this.request(`/export?${soundQuery(sound)}`, {
      method: "POST",
    });
    const disposition = res.headers.get("Content-Disposition") ?? "";
    const match = /filename="([^"]+)"/.exec(disposition);
    return {
      blob: await res.blob(),
      filename: match?.[1] ?? `${sound.instrument}.wav`,
      exportId: res.headers.get("X-Kicks-Export-Id") ?? "",
    };
  }
}

export interface ExportResult {
  blob: Blob;
  filename: string;
  exportId: string;
}
