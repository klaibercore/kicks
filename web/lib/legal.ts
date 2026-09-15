/**
 * Operator details for the Impressum and Datenschutzerklärung. Filled from the
 * environment at build time so the repository never has to contain a private
 * address. Placeholders are rendered visibly until they are set — an Impressum
 * with missing details is a § 5 DDG problem, so it must not fail quietly.
 */

const env = (key: string, placeholder: string) => {
  const value = process.env[key];
  return value && value.trim() ? value.trim() : placeholder;
};

export const operator = {
  name: env("NEXT_PUBLIC_LEGAL_NAME", "[Name des Betreibers]"),
  street: env("NEXT_PUBLIC_LEGAL_STREET", "[Straße und Hausnummer]"),
  city: env("NEXT_PUBLIC_LEGAL_CITY", "[PLZ Ort]"),
  email: env("NEXT_PUBLIC_LEGAL_EMAIL", "[E-Mail-Adresse]"),
  phone: env("NEXT_PUBLIC_LEGAL_PHONE", "[Telefonnummer]"),
  vatId: env("NEXT_PUBLIC_LEGAL_VAT_ID", "[USt-IdNr., falls vorhanden]"),
  country: "Deutschland",
} as const;

export const placeholdersPresent = Object.values(operator).some((v) => v.startsWith("["));

export const legalDocuments = {
  impressum: { href: "/legal/impressum/", title: "Impressum" },
  datenschutz: { href: "/legal/datenschutz/", title: "Datenschutzerklärung" },
  agb: { href: "/legal/agb/", title: "AGB" },
  widerruf: { href: "/legal/widerruf/", title: "Widerrufsbelehrung" },
} as const;
