import type { Metadata } from "next";
import { LegalPage } from "@/components/legal/legal-page";
import { operator } from "@/lib/legal";

export const metadata: Metadata = { title: "Impressum" };

export default function ImpressumPage() {
  return (
    <LegalPage title="Impressum">
      <h2>Angaben gemäß § 5 DDG</h2>
      <address>
        {operator.name}
        <br />
        {operator.street}
        <br />
        {operator.city}
        <br />
        {operator.country}
      </address>

      <h2>Kontakt</h2>
      <p>
        E-Mail: {operator.email}
        <br />
        Telefon: {operator.phone}
      </p>

      <h2>Umsatzsteuer-Identifikationsnummer</h2>
      <p>Umsatzsteuer-Identifikationsnummer gemäß § 27a Umsatzsteuergesetz: {operator.vatId}</p>

      <h2>Verantwortlich für den Inhalt nach § 18 Abs. 2 MStV</h2>
      <p>
        {operator.name}, {operator.street}, {operator.city}
      </p>

      <h2>Verbraucherstreitbeilegung</h2>
      <p>
        Wir sind nicht bereit und nicht verpflichtet, an Streitbeilegungsverfahren vor einer Verbraucherschlichtungsstelle
        teilzunehmen (§ 36 VSBG).
      </p>

      <h2>Haftung für Inhalte und Links</h2>
      <p>
        Als Diensteanbieter sind wir für eigene Inhalte auf diesen Seiten nach den allgemeinen Gesetzen verantwortlich. Für
        Inhalte externer Seiten, auf die wir verlinken, ist der jeweilige Anbieter verantwortlich; zum Zeitpunkt der
        Verlinkung waren keine Rechtsverstöße erkennbar.
      </p>
    </LegalPage>
  );
}
