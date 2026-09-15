import type { Metadata } from "next";
import { LegalPage } from "@/components/legal/legal-page";
import { operator } from "@/lib/legal";

export const metadata: Metadata = { title: "Widerrufsbelehrung" };

export default function WiderrufPage() {
  return (
    <LegalPage title="Widerrufsbelehrung">
      <p>
        Verbrauchern steht bei Verträgen über digitale Inhalte, die nicht auf einem körperlichen Datenträger geliefert
        werden, das folgende Widerrufsrecht zu.
      </p>

      <h2>Widerrufsrecht</h2>
      <p>Sie haben das Recht, binnen vierzehn Tagen ohne Angabe von Gründen diesen Vertrag zu widerrufen.</p>
      <p>Die Widerrufsfrist beträgt vierzehn Tage ab dem Tag des Vertragsabschlusses.</p>
      <p>
        Um Ihr Widerrufsrecht auszuüben, müssen Sie uns ({operator.name}, {operator.street}, {operator.city}, E-Mail:{" "}
        {operator.email}) mittels einer eindeutigen Erklärung (z. B. ein mit der Post versandter Brief oder eine E-Mail) über
        Ihren Entschluss, diesen Vertrag zu widerrufen, informieren. Sie können dafür das beigefügte Muster-Widerrufsformular
        verwenden, das jedoch nicht vorgeschrieben ist.
      </p>
      <p>
        Zur Wahrung der Widerrufsfrist reicht es aus, dass Sie die Mitteilung über die Ausübung des Widerrufsrechts vor Ablauf
        der Widerrufsfrist absenden.
      </p>

      <h2>Folgen des Widerrufs</h2>
      <p>
        Wenn Sie diesen Vertrag widerrufen, haben wir Ihnen alle Zahlungen, die wir von Ihnen erhalten haben, unverzüglich und
        spätestens binnen vierzehn Tagen ab dem Tag zurückzuzahlen, an dem die Mitteilung über Ihren Widerruf dieses Vertrags
        bei uns eingegangen ist. Für diese Rückzahlung verwenden wir dasselbe Zahlungsmittel, das Sie bei der ursprünglichen
        Transaktion eingesetzt haben, es sei denn, mit Ihnen wurde ausdrücklich etwas anderes vereinbart; in keinem Fall
        werden Ihnen wegen dieser Rückzahlung Entgelte berechnet.
      </p>

      <h2>Erlöschen des Widerrufsrechts bei digitalen Inhalten</h2>
      <p>
        Das Widerrufsrecht erlischt bei einem Vertrag über die Bereitstellung nicht auf einem körperlichen Datenträger
        befindlicher digitaler Inhalte, wenn wir mit der Ausführung des Vertrags begonnen haben, nachdem Sie
      </p>
      <ol>
        <li>ausdrücklich zugestimmt haben, dass wir mit der Ausführung des Vertrags vor Ablauf der Widerrufsfrist beginnen, und</li>
        <li>
          Ihre Kenntnis davon bestätigt haben, dass Sie durch Ihre Zustimmung mit Beginn der Ausführung des Vertrags Ihr
          Widerrufsrecht verlieren (§ 356 Abs. 5 BGB).
        </li>
      </ol>
      <p>
        Diese Zustimmung holen wir vor jedem Kauf gesondert ein. Credits werden unmittelbar nach der Zahlung bereitgestellt;
        mit der Bereitstellung erlischt das Widerrufsrecht. Unabhängig davon erstatten wir nach § 9 unserer AGB auf Wunsch den
        anteiligen Kaufpreis für Credits, die innerhalb von 14 Tagen nach dem Kauf noch vollständig unverbraucht sind.
      </p>

      <h2>Muster-Widerrufsformular</h2>
      <div className="box">
        <p>(Wenn Sie den Vertrag widerrufen wollen, dann füllen Sie bitte dieses Formular aus und senden Sie es zurück.)</p>
        <p>
          An {operator.name}, {operator.street}, {operator.city}, E-Mail: {operator.email}
        </p>
        <p>
          Hiermit widerrufe(n) ich/wir (*) den von mir/uns (*) abgeschlossenen Vertrag über den Kauf der folgenden Waren (*) /
          die Erbringung der folgenden Dienstleistung (*):
        </p>
        <p>Bestellt am (*) / erhalten am (*):</p>
        <p>Name des/der Verbraucher(s):</p>
        <p>Anschrift des/der Verbraucher(s):</p>
        <p>Unterschrift des/der Verbraucher(s) (nur bei Mitteilung auf Papier):</p>
        <p>Datum:</p>
        <p>(*) Unzutreffendes streichen.</p>
      </div>
    </LegalPage>
  );
}
