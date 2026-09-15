import type { Metadata } from "next";
import Link from "next/link";
import { LegalPage } from "@/components/legal/legal-page";
import { legalDocuments, operator } from "@/lib/legal";

export const metadata: Metadata = { title: "AGB" };

export default function AgbPage() {
  return (
    <LegalPage title="Allgemeine Geschäftsbedingungen">
      <h2>§ 1 Geltungsbereich, Anbieter</h2>
      <p>
        Diese Bedingungen gelten für alle Verträge zwischen {operator.name}, {operator.street}, {operator.city} („Anbieter“)
        und Nutzerinnen und Nutzern („Kunde“) über die Nutzung des Dienstes „kicks“ und den Erwerb von Export-Credits.
        Der Dienst richtet sich an Verbraucher und Unternehmer. Verbraucher ist jede natürliche Person, die ein Rechtsgeschäft
        zu Zwecken abschließt, die überwiegend weder ihrer gewerblichen noch ihrer selbständigen beruflichen Tätigkeit
        zugerechnet werden können (§ 13 BGB).
      </p>

      <h2>§ 2 Leistungsbeschreibung</h2>
      <ol>
        <li>
          Der Dienst erzeugt Schlagzeugklänge (Kick, Snare, Hi-Hat) mit einem neuronalen Modell nach Vorgaben des Kunden.
          Das Anhören von Vorschauen im Studio ist für angemeldete Kunden kostenlos.
        </li>
        <li>
          Ein <strong>Export</strong> ist die Bereitstellung eines erzeugten Klangs als Audiodatei (WAV, 44,1 kHz, 24 Bit) zum
          Herunterladen einschließlich der Lizenz nach § 6. Jeder Export verbraucht einen Credit.
        </li>
        <li>
          <strong>Credits</strong> werden in Paketen erworben, unmittelbar nach Zahlung dem Konto gutgeschrieben und
          verfallen nicht. Sie sind an das Konto gebunden, nicht übertragbar und werden nicht in Geld ausgezahlt, soweit
          nicht § 9 etwas anderes bestimmt.
        </li>
        <li>
          Der Anbieter bemüht sich um eine hohe Verfügbarkeit, schuldet aber keine ununterbrochene Erreichbarkeit.
          Wartungsarbeiten werden nach Möglichkeit angekündigt.
        </li>
      </ol>

      <h2>§ 3 Konto</h2>
      <ol>
        <li>Die Nutzung setzt ein Konto voraus. Die Anmeldung erfolgt per E-Mail-Link oder über Google bzw. GitHub.</li>
        <li>Der Kunde muss volljährig und unbeschränkt geschäftsfähig sein.</li>
        <li>
          Der Kunde hält seine Zugangsdaten geheim und informiert den Anbieter über eine missbräuchliche Nutzung. Das Konto
          kann jederzeit in den Kontoeinstellungen gelöscht werden; nicht verbrauchte Credits verfallen dabei, soweit nicht
          § 9 greift.
        </li>
        <li>
          Der Anbieter kann ein Konto sperren, wenn der Kunde gegen diese Bedingungen verstößt, insbesondere den Dienst
          automatisiert oder missbräuchlich nutzt. Rechtmäßig erworbene Credits bleiben davon unberührt.
        </li>
      </ol>

      <h2>§ 4 Vertragsschluss beim Kauf von Credits</h2>
      <ol>
        <li>
          Die Darstellung der Credit-Pakete auf der Preisseite ist kein verbindliches Angebot, sondern eine Aufforderung an
          den Kunden, ein Angebot abzugeben.
        </li>
        <li>
          Der Bestellablauf: Der Kunde wählt ein Paket, bestätigt die Kenntnisnahme dieser AGB und der Widerrufsbelehrung
          sowie die Zustimmung zum sofortigen Beginn der Leistung und wird zur Bezahlseite unseres Zahlungsdienstleisters
          Stripe weitergeleitet. Dort werden Rechnungsanschrift und Zahlungsmittel erfasst und der endgültige Bruttopreis
          einschließlich der Umsatzsteuer seines Landes angezeigt. Mit Betätigen der Schaltfläche „Zahlen“ (bzw. der
          entsprechend beschrifteten Schaltfläche) gibt der Kunde ein verbindliches, zahlungspflichtiges Angebot ab.
        </li>
        <li>
          Der Vertrag kommt zustande, wenn der Anbieter die Zahlung annimmt und die Credits dem Konto gutschreibt. Der Kunde
          erhält eine Bestätigung und die Rechnung per E-Mail.
        </li>
        <li>
          Eingabefehler können bis zum Absenden der Bestellung auf der Bezahlseite korrigiert werden. Der Vertragstext wird
          vom Anbieter gespeichert; diese AGB sind jederzeit unter{" "}
          <Link href={legalDocuments.agb.href}>{legalDocuments.agb.title}</Link> abrufbar, die Bestelldaten im Konto unter
          „Billing“.
        </li>
        <li>Vertragssprache ist Deutsch. Die Benutzeroberfläche des Dienstes ist in englischer Sprache gehalten.</li>
      </ol>

      <h2>§ 5 Preise und Zahlung</h2>
      <ol>
        <li>
          Alle Preise sind Endpreise in Euro und enthalten die gesetzliche Umsatzsteuer. Der auf der Bezahlseite angezeigte
          Betrag ist maßgeblich; er berücksichtigt den Umsatzsteuersatz des Landes des Kunden. Unternehmer mit gültiger
          Umsatzsteuer-Identifikationsnummer aus anderen EU-Staaten können diese angeben (Reverse-Charge).
        </li>
        <li>
          Die Zahlung erfolgt über Stripe Payments Europe, Ltd. mit den dort angebotenen Zahlungsmitteln. Der Kaufpreis ist
          sofort fällig.
        </li>
        <li>Rechnungen werden elektronisch erteilt und sind im Konto unter „Billing“ jederzeit abrufbar.</li>
      </ol>

      <h2>§ 6 Nutzungsrechte an exportierten Klängen</h2>
      <ol>
        <li>
          Mit dem Export erhält der Kunde ein einfaches, zeitlich und räumlich unbeschränktes, unterlizenzierbares Recht, den
          exportierten Klang in eigenen Musik-, Video-, Spiele- und sonstigen Medienproduktionen zu vervielfältigen, zu
          bearbeiten, mit anderen Werken zu verbinden und die so entstandenen Produktionen kommerziell und nicht kommerziell
          zu verwerten. Eine Namensnennung ist nicht erforderlich.
        </li>
        <li>
          Nicht gestattet ist, exportierte Klänge unverändert oder nur unwesentlich verändert als solche weiterzugeben,
          insbesondere als Samples, Sample-Packs, Sound-Bibliotheken, Presets oder als Trainingsdaten für Software, die
          Klänge erzeugt.
        </li>
        <li>
          Klänge werden algorithmisch erzeugt; ein Ausschließlichkeitsrecht besteht nicht. Gleiche Reglerwerte können zu
          gleichen oder ähnlichen Klängen anderer Kunden führen.
        </li>
        <li>Vorschauen dienen ausschließlich dem Anhören im Studio; § 6 Abs. 1 gilt für sie nicht.</li>
      </ol>

      <h2>§ 7 Widerrufsrecht</h2>
      <p>
        Verbrauchern steht ein gesetzliches Widerrufsrecht zu. Einzelheiten ergeben sich aus der{" "}
        <Link href={legalDocuments.widerruf.href}>Widerrufsbelehrung</Link>. Credits sind digitale Inhalte, die nicht auf einem
        körperlichen Datenträger geliefert werden. Das Widerrufsrecht erlischt, wenn der Anbieter mit der Ausführung des
        Vertrags begonnen hat, nachdem der Kunde ausdrücklich zugestimmt hat, dass mit der Ausführung vor Ablauf der
        Widerrufsfrist begonnen wird, und seine Kenntnis davon bestätigt hat, dass er durch seine Zustimmung mit Beginn der
        Ausführung sein Widerrufsrecht verliert (§ 356 Abs. 5 BGB). Diese Zustimmung wird vor jedem Kauf gesondert
        eingeholt.
      </p>

      <h2>§ 8 Gewährleistung, Aktualisierungen</h2>
      <ol>
        <li>
          Für digitale Produkte gelten die gesetzlichen Vorschriften der §§ 327 ff. BGB. Der Anbieter stellt die Credits und
          Exporte in der in § 2 beschriebenen Beschaffenheit bereit.
        </li>
        <li>
          Der Anbieter stellt Aktualisierungen bereit, die für den Erhalt der Vertragsmäßigkeit des Dienstes erforderlich sind,
          solange der Kunde Credits besitzt, und informiert über sie in angemessener Weise.
        </li>
        <li>
          Der Klangcharakter des Modells kann sich durch Weiterentwicklung ändern. Ein bestimmter Klang zu bestimmten
          Reglerwerten ist nicht geschuldet; geschuldet ist ein Export in der beschriebenen technischen Qualität.
        </li>
      </ol>

      <h2>§ 9 Kulanzerstattung nicht genutzter Credits</h2>
      <p>
        Unabhängig vom gesetzlichen Widerrufsrecht erstattet der Anbieter auf Wunsch den anteiligen Kaufpreis für Credits
        eines Pakets, die innerhalb von 14 Tagen nach dem Kauf noch vollständig unverbraucht sind. Der Antrag ist per E-Mail
        an {operator.email} zu stellen; die Erstattung erfolgt über das ursprüngliche Zahlungsmittel.
      </p>

      <h2>§ 10 Haftung</h2>
      <ol>
        <li>
          Der Anbieter haftet unbeschränkt für Vorsatz und grobe Fahrlässigkeit, für Schäden aus der Verletzung des Lebens,
          des Körpers oder der Gesundheit sowie nach dem Produkthaftungsgesetz.
        </li>
        <li>
          Bei einfach fahrlässiger Verletzung wesentlicher Vertragspflichten (Pflichten, deren Erfüllung die ordnungsgemäße
          Durchführung des Vertrags überhaupt erst ermöglicht und auf deren Einhaltung der Kunde regelmäßig vertrauen darf) ist
          die Haftung auf den vertragstypischen, vorhersehbaren Schaden begrenzt.
        </li>
        <li>Im Übrigen ist die Haftung ausgeschlossen.</li>
      </ol>

      <h2>§ 11 Änderungen dieser Bedingungen</h2>
      <p>
        Der Anbieter kann diese Bedingungen mit Wirkung für die Zukunft ändern, soweit dies aus triftigen Gründen
        (Rechtsänderungen, Änderungen des Dienstes) erforderlich ist und den Kunden nicht unangemessen benachteiligt.
        Änderungen werden mindestens vier Wochen vor Wirksamwerden per E-Mail mitgeteilt. Bereits geschlossene Kaufverträge
        bleiben unberührt.
      </p>

      <h2>§ 12 Streitbeilegung</h2>
      <p>
        Der Anbieter ist nicht bereit und nicht verpflichtet, an Streitbeilegungsverfahren vor einer
        Verbraucherschlichtungsstelle teilzunehmen (§ 36 VSBG).
      </p>

      <h2>§ 13 Schlussbestimmungen</h2>
      <ol>
        <li>
          Es gilt das Recht der Bundesrepublik Deutschland unter Ausschluss des UN-Kaufrechts. Bei Verbrauchern gilt diese
          Rechtswahl nur, soweit dadurch nicht zwingende Verbraucherschutzvorschriften des Staates entzogen werden, in dem der
          Verbraucher seinen gewöhnlichen Aufenthalt hat.
        </li>
        <li>
          Ist der Kunde Kaufmann, juristische Person des öffentlichen Rechts oder öffentlich-rechtliches Sondervermögen, ist
          Gerichtsstand der Sitz des Anbieters.
        </li>
        <li>
          Sollte eine Bestimmung unwirksam sein, bleibt die Wirksamkeit der übrigen Bestimmungen unberührt.
        </li>
      </ol>
    </LegalPage>
  );
}
