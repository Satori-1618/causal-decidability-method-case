# Forschungsziel: kausale Erklärungen wirksamer unterscheiden

**Status: Entwicklungsauftrag, keine bestätigte Überlegenheit.**

## Präzises Ziel

Auf vorab zurückgehaltenen Familien ausführbarer neuronaler Rechnungen mit bekannter
Grundwahrheit soll eine festgelegte, vorhersagegeleitete Auswahl von Eingriffen bei
gleichem Messbudget **häufiger genau die richtige Menge interventionell vereinbarer
Erklärungen zurückgeben** als starke Vergleichsverfahren, ohne die vorab festgelegte
Rate falscher Ausschlüsse zu überschreiten.

Die richtige Menge wird über den **gesamten erlaubten Eingriffsvorrat** definiert.
Zwei Rechnungen, die dort ununterscheidbar sind, müssen gemeinsam stehen bleiben.
Ein schwaches ausgewähltes Experiment darf diese Menge nicht nachträglich vergrößern
und dadurch als richtige Identifikation gelten.

## Was der Nachweis leisten muss

1. **Ernsthafte Rivalen:** unterschiedliche innere Rechnungen liefern beim gewöhnlichen
   vollständigen Patch dieselbe Antwort. Erst gezielte Eingriffe können sie trennen.
2. **Fairer Vergleich:** gleiche Vorhersagen, erlaubte Zellen, Unsicherheitsinformationen,
   Messkosten und derselbe ausgelieferte Mengenklassifikator für alle Verfahren.
3. **Messbarer Zusatznutzen:** primär die gepaarte Differenz korrekt zurückgegebener
   Erklärungsmengen; zusätzlich falsche Ausschlüsse, falsche eindeutige Zuordnungen,
   verbleibende Mehrdeutigkeit und Erkennung unpassender Kandidatenräume.
4. **Keine leichte Referenz als Hauptgegner:** Full-Patch-only ist eine Negativkontrolle.
   Die Hauptreferenz maximiert mittlere quadratische Vorhersagetrennung; ein festes
   Read-Split-Design und zufällige Auswahl ergänzen den Vergleich.
5. **Unberührte Strukturen:** eine spätere Bestätigung muss andere vorab deklarierte
   Rechenzusammensetzungen enthalten. Neue Seeds allein belegen diesen Transfer nicht.

## Konkreter erster Aufbau

Der CPU-Benchmark unter [applications/design-comparison](../applications/design-comparison/)
führt kleine neuronale Rechengraphen tatsächlich aus. Rivalen sind Übertragung über
Kanal A, Kanal B, beide additiv, beide gemeinsam erforderlich sowie zwei kontextabhängige
Umschaltungen. Eine anders geschriebene, interventionell äquivalente Rechnung und eine
Rechnung außerhalb des Kandidatenraums prüfen die Grenzen des Verfahrens.

Die erste Runde entwickelt und prüft die Infrastruktur auf zwei deklarierten Familien.
Sie entscheidet noch nicht über das Forschungsziel. Numerische Abweichung, bekannte
simulierte Messunsicherheit und wissenschaftliche Erfolgsgrenze bleiben getrennt.

## Erfolg, Gleichstand und Scheitern

Für einen künftigen Bestätigungsvertrag wird als praktische Zielgröße ein Vorteil von
mindestens **5 Prozentpunkten** korrekter Mengenauflösung gegenüber der Hauptreferenz
vorgeschlagen; die simultane untere Unsicherheitsschranke muss diesen Wert überschreiten.
Die obere Schranke der Rate falscher Ausschlüsse soll zugleich höchstens **1 %** sein.
Das sind wissenschaftliche Zielentscheidungen, keine aus den Entwicklungsergebnissen
optimierten Werte. Stichprobe und erreichbare Power müssen vor einem Freeze dazu passen.

Ein Gleichstand mit einem guten etablierten Design ist ein Ergebnis über die Anwendung
und Effizienz des Verfahrens, aber kein Überlegenheitsnachweis. Ein Vorteil nur gegenüber
Full-Patch-only zeigt den Wert zusätzlicher Eingriffe. Ein Vorteil nur in diesen kleinen
Graphen ist keine Zuverlässigkeitsgarantie für Transformer.

## Reihenfolge

1. Entwicklungsvertrag, bekannte Graphen, gemeinsame Schnittstellen und Kontrollen bauen.
2. Entwicklungsvergleich ausführen, unabhängig nachprüfen und alle Ausgänge berichten.
3. Falls sinnvoll: Stichprobe planen, Regeln und strukturellen Holdout einfrieren.
4. Frisch bestätigen; anschließend auf einen offenen LLM-Fall übertragen.

**Anschluss an Geiger:** Die Kandidaten operationalisieren unterschiedliche kausale
Erklärungen und ihre Interventionsabbildungen. Untersucht wird, welche Eingriffe ihre
Vorhersagen messbar trennen. Das ergänzt die Prüfung der Treue einer vorgegebenen
Abstraktion; es ersetzt sie nicht und garantiert keine einzigartige natürliche Semantik.

**Aktueller Abschlussmaßstab:** Ein reproduzierbarer, geprüfter Entwicklungsbenchmark
mit ehrlicher Ergebnistabelle und weiterhin ungeöffnetem strukturellem Holdout.
