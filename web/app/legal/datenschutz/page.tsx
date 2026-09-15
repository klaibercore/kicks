import type { Metadata } from "next";
import { LegalPage } from "@/components/legal/legal-page";
import { operator } from "@/lib/legal";

export const metadata: Metadata = { title: "Datenschutzerklärung" };

export default function DatenschutzPage() {
  return (
    <LegalPage title="Datenschutzerklärung">
      <p>
        Diese Erklärung informiert nach Art. 13 und 14 DSGVO darüber, welche personenbezogenen Daten beim Besuch dieser
        Website und bei der Nutzung des Dienstes „kicks“ verarbeitet werden, zu welchen Zwecken, auf welcher Rechtsgrundlage
        und welche Rechte Sie haben.
      </p>

      <h2>1. Verantwortlicher</h2>
      <address>
        {operator.name}
        <br />
        {operator.street}, {operator.city}, {operator.country}
        <br />
        E-Mail: {operator.email}
      </address>

      <h2>2. Grundsätze</h2>
      <p>
        Wir erheben so wenige Daten wie möglich. Die Website ist eine statische Seite ohne eigene Server-Logik; es gibt keine
        Tracking-Cookies, keine Werbung und keine Weitergabe von Daten zu Werbezwecken. Schriftarten werden von unserem eigenen
        Host geladen, nicht von Google.
      </p>

      <h2>3. Hosting (GitHub Pages)</h2>
      <p>
        Die Website wird über GitHub Pages ausgeliefert, einen Dienst der GitHub, Inc., 88 Colin P. Kelly Jr. Street, San
        Francisco, CA 94107, USA. Beim Aufruf werden technisch notwendige Daten (IP-Adresse, Zeitpunkt, aufgerufene Datei,
        Browserkennung) in Server-Logs verarbeitet. Rechtsgrundlage ist Art. 6 Abs. 1 lit. f DSGVO (berechtigtes Interesse an
        einem sicheren, funktionsfähigen Betrieb). GitHub ist unter dem EU-US Data Privacy Framework zertifiziert; Einzelheiten
        finden Sie in der{" "}
        <a href="https://docs.github.com/site-policy/privacy-policies/github-general-privacy-statement" rel="noopener noreferrer">
          Datenschutzerklärung von GitHub
        </a>
        .
      </p>

      <h2>4. Reichweitenmessung (Cloudflare Web Analytics) – nur mit Einwilligung</h2>
      <p>
        Mit Ihrer Einwilligung setzen wir Cloudflare Web Analytics der Cloudflare, Inc., 101 Townsend St, San Francisco, CA
        94107, USA, ein, um Seitenaufrufe zu zählen. Der Dienst verwendet keine Cookies und keine dauerhaften Kennungen im
        Browser; verarbeitet werden die aufgerufene Seite, der Referrer, Browsertyp und die – von Cloudflare gekürzte –
        IP-Adresse. Rechtsgrundlage ist Ihre Einwilligung (Art. 6 Abs. 1 lit. a DSGVO, § 25 Abs. 1 TDDDG). Die Einwilligung
        können Sie jederzeit über „Cookie-Einstellungen“ in der Fußzeile widerrufen; das Skript wird dann nicht mehr geladen.
        Cloudflare ist unter dem EU-US Data Privacy Framework zertifiziert.
      </p>

      <h2>5. Benutzerkonto und Anmeldung (Supabase)</h2>
      <p>
        Für die Nutzung des Studios und den Erwerb von Credits ist ein Konto erforderlich. Konten, Guthaben, gespeicherte Kits
        und Bestellungen werden bei Supabase, Inc., 970 Toa Payoh North #07-04, Singapore 318992, gespeichert; das Projekt ist
        in einer Rechenzentrumsregion in der Europäischen Union angelegt. Mit Supabase besteht ein Auftragsverarbeitungsvertrag
        nach Art. 28 DSGVO.
      </p>
      <p>
        Bei der Anmeldung per E-Mail-Link speichern wir Ihre E-Mail-Adresse. Bei der Anmeldung über Google oder GitHub
        übermittelt der jeweilige Anbieter Ihre E-Mail-Adresse, Ihren Anzeigenamen, ggf. eine Profilbild-URL und eine
        Anbieter-Kennung; wir fordern keine weiteren Berechtigungen an. Rechtsgrundlage ist Art. 6 Abs. 1 lit. b DSGVO
        (Erfüllung des Nutzungsvertrags). Für den Anmeldevorgang beim Anbieter gelten dessen Datenschutzhinweise (Google
        Ireland Ltd., Dublin; GitHub, Inc., San Francisco).
      </p>
      <p>
        Zur Aufrechterhaltung der Anmeldung wird ein Sitzungs-Token im lokalen Speicher Ihres Browsers abgelegt. Das ist für
        den von Ihnen gewünschten Dienst zwingend erforderlich (§ 25 Abs. 2 Nr. 2 TDDDG) und bedarf keiner Einwilligung.
      </p>

      <h2>6. Synthese-Dienst</h2>
      <p>
        Beim Abspielen und Exportieren von Klängen sendet Ihr Browser die Reglerwerte zusammen mit Ihrem Sitzungs-Token an
        unseren Synthese-Server. Der Server verarbeitet diese Daten, um das Audio zu berechnen; Vorschauen werden nicht
        gespeichert. Bei einem Export speichern wir Zeitpunkt, Instrument und Reglerwerte in Ihrem Konto, weil damit die
        erteilte Lizenz und der Verbrauch eines Credits nachgewiesen wird (Art. 6 Abs. 1 lit. b DSGVO). Serverseitige
        Zugriffsprotokolle mit IP-Adresse werden nach spätestens 14 Tagen gelöscht (Art. 6 Abs. 1 lit. f DSGVO).
      </p>

      <h2>7. Zahlungen (Stripe)</h2>
      <p>
        Käufe werden über Stripe Payments Europe, Ltd., 1 Grand Canal Street Lower, Dublin 2, Irland, abgewickelt. Auf der
        Bezahlseite von Stripe geben Sie Name, Rechnungsadresse, E-Mail-Adresse, ggf. Umsatzsteuer-ID und Ihre Zahlungsdaten
        ein. Zahlungsdaten (z. B. Kartennummern) erreichen uns zu keinem Zeitpunkt. Wir erhalten von Stripe die Bestellnummer,
        Beträge, Steuerbetrag, Land und den Status der Zahlung sowie die Rechnung. Rechtsgrundlage ist Art. 6 Abs. 1 lit. b
        DSGVO (Vertragsdurchführung) und Art. 6 Abs. 1 lit. c DSGVO in Verbindung mit den steuer- und handelsrechtlichen
        Aufbewahrungspflichten (§ 147 AO, § 257 HGB, § 14b UStG). Stripe ist für die Betrugsprävention und die Abwicklung
        teilweise eigenständig verantwortlich; Einzelheiten finden Sie in der{" "}
        <a href="https://stripe.com/de/privacy" rel="noopener noreferrer">
          Datenschutzerklärung von Stripe
        </a>
        . Eine Übermittlung an Stripe, Inc. in den USA ist durch das EU-US Data Privacy Framework und Standardvertragsklauseln
        abgesichert.
      </p>

      <h2>8. Nachweis von Einwilligungen und Vertragserklärungen</h2>
      <p>
        Beim Kauf protokollieren wir mit Zeitstempel und Versionsstand, dass Sie die AGB akzeptiert und dem sofortigen Beginn
        der Leistung unter Verlust des Widerrufsrechts zugestimmt haben (Art. 7 Abs. 1 DSGVO, § 356 Abs. 5 BGB, Art. 6 Abs. 1
        lit. c DSGVO). Die Analytics-Einwilligung wird nur lokal in Ihrem Browser gespeichert.
      </p>

      <h2>9. Empfänger und Drittlandübermittlung</h2>
      <p>
        Empfänger personenbezogener Daten sind die in dieser Erklärung genannten Dienstleister (GitHub, Cloudflare, Supabase,
        Stripe sowie – bei entsprechender Anmeldung – Google oder GitHub als Identitätsanbieter). Soweit Daten in die USA
        übermittelt werden, geschieht dies auf Grundlage eines Angemessenheitsbeschlusses (EU-US Data Privacy Framework, Art.
        45 DSGVO) oder der EU-Standardvertragsklauseln (Art. 46 Abs. 2 lit. c DSGVO). Eine Weitergabe zu anderen Zwecken findet
        nicht statt.
      </p>

      <h2>10. Speicherdauer</h2>
      <ul>
        <li>Kontodaten, Guthaben, Kits und Exportverlauf: bis zur Löschung Ihres Kontos.</li>
        <li>
          Bestell- und Rechnungsdaten: zehn Jahre ab Ende des Kalenderjahres der Rechnung (§ 147 AO). Bei Kontolöschung werden
          diese Datensätze von Ihrem Profil getrennt und ohne Personenbezug im gesetzlich erforderlichen Umfang aufbewahrt.
        </li>
        <li>Einwilligungsnachweise: drei Jahre nach Ende des Jahres, in dem die Einwilligung endete (§ 195 BGB).</li>
        <li>Server-Logs: höchstens 14 Tage.</li>
      </ul>

      <h2>11. Ihre Rechte</h2>
      <p>
        Sie haben das Recht auf Auskunft (Art. 15), Berichtigung (Art. 16), Löschung (Art. 17), Einschränkung der Verarbeitung
        (Art. 18), Datenübertragbarkeit (Art. 20) und Widerspruch gegen Verarbeitungen auf Grundlage von Art. 6 Abs. 1 lit. f
        DSGVO (Art. 21). Eine erteilte Einwilligung können Sie jederzeit mit Wirkung für die Zukunft widerrufen (Art. 7 Abs. 3).
        Auskunft und Datenübertragbarkeit können Sie unmittelbar in Ihrem Konto unter „Download my data“ ausüben; die Löschung
        Ihres Kontos ebenfalls dort unter „Delete account“. Für alle übrigen Anliegen wenden Sie sich an {operator.email}.
      </p>
      <p>
        Sie haben außerdem das Recht, sich bei einer Datenschutz-Aufsichtsbehörde zu beschweren (Art. 77 DSGVO), insbesondere
        bei der Behörde Ihres Wohnsitzes oder der für den Verantwortlichen zuständigen Landesbehörde.
      </p>

      <h2>12. Keine automatisierte Entscheidungsfindung</h2>
      <p>Es findet keine automatisierte Entscheidungsfindung einschließlich Profiling im Sinne von Art. 22 DSGVO statt.</p>

      <h2>13. Pflicht zur Bereitstellung</h2>
      <p>
        Die Angabe einer E-Mail-Adresse ist für ein Konto erforderlich, die Rechnungsdaten sind für einen Kauf gesetzlich
        erforderlich. Ohne diese Angaben kann der jeweilige Dienst nicht erbracht werden. Alle übrigen Angaben sind freiwillig.
      </p>

      <h2>14. Änderungen</h2>
      <p>
        Wir passen diese Erklärung an, wenn sich der Dienst oder die Rechtslage ändert. Bei wesentlichen Änderungen wird die
        Einwilligung zur Reichweitenmessung erneut abgefragt.
      </p>
    </LegalPage>
  );
}
