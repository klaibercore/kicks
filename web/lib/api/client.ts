import { config } from "@/lib/config";
import type {
  Evaluation,
  Health,
  InstrumentConfig,
  InstrumentsResponse,
  Me,
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

export type TokenProvider = () => Promise<string | null>;

/** Query string for a sound. Keys are sorted so equal sounds hit the API's cache. */
export function soundQuery(sound: Sound): string {
  const params = new URLSearchParams();
  params.set("instrument", sound.instrument);
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
    private readonly token: TokenProvider = async () => null,
  ) {}

  private async headers(extra: Record<string, string> = {}): Promise<HeadersInit> {
    const token = await this.token();
    return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
  }

  private async request(path: string, init: RequestInit = {}): Promise<Response> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: { ...(await this.headers()), ...(init.headers ?? {}) },
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

  config(instrument: string) {
    return this.json<InstrumentConfig>(`/config?instrument=${encodeURIComponent(instrument)}`);
  }

  me() {
    return this.json<Me>("/me");
  }

  /** Preview render: raw WAV bytes. Free, cached server-side. */
  async generate(sound: Sound): Promise<ArrayBuffer> {
    const res = await this.request(`/generate?${soundQuery(sound)}`);
    return res.arrayBuffer();
  }

  evaluate(sound: Sound) {
    return this.json<Evaluation>(`/evaluate?${soundQuery(sound)}`);
  }

  spectrogram(sound: Sound) {
    return this.json<Spectrogram>(`/spectrogram?${soundQuery(sound)}`);
  }

  /** Paid export: costs one credit, returns the file and the remaining balance. */
  async exportSample(sound: Sound, idempotencyKey: string): Promise<ExportResult> {
    const res = await this.request(`/export?${soundQuery(sound)}`, {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey },
    });
    const disposition = res.headers.get("Content-Disposition") ?? "";
    const match = /filename="([^"]+)"/.exec(disposition);
    return {
      blob: await res.blob(),
      filename: match?.[1] ?? `${sound.instrument}.wav`,
      remaining: Number(res.headers.get("X-Kicks-Credits-Remaining") ?? NaN),
      exportId: res.headers.get("X-Kicks-Export-Id") ?? idempotencyKey,
    };
  }
}

export interface ExportResult {
  blob: Blob;
  filename: string;
  remaining: number;
  exportId: string;
}
