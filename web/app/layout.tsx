import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { AuthProvider } from "@/hooks/use-auth";
import { CloudflareAnalytics } from "@/components/site/cloudflare-analytics";
import { ConsentBanner } from "@/components/site/consent-banner";
import { SiteFooter } from "@/components/site/footer";
import { SiteHeader } from "@/components/site/header";
import { ThemeProvider } from "@/components/site/theme-provider";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

// next/font self-hosts these at build time: no request to Google at runtime,
// which matters under German case law on remote font loading (LG München I,
// 3 O 17493/20).
const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "kicks — neural drum synthesis", template: "%s · kicks" },
  description:
    "Design kicks, snares and hi-hats with perceptual sliders, play them from a MIDI pad, and export licensed samples.",
  applicationName: "kicks",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#0a0a0a" },
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
  ],
  viewportFit: "cover",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <ThemeProvider>
          <AuthProvider>
            <TooltipProvider>
              <SiteHeader />
              <main className="flex-1">{children}</main>
              <SiteFooter />
              <ConsentBanner />
              <CloudflareAnalytics />
              <Toaster richColors position="bottom-right" />
            </TooltipProvider>
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
