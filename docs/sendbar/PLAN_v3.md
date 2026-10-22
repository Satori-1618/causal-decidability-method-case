# Plan v3 — make seven cases publicly checkable and sendable

*Internal working document (German). Not part of the scientific release; Felix decides
before any merge whether it stays.*

**Version 3, 22.10.2026.** Basis: Plan v2 (SHA-256 `85ef4363…f387b3`) plus Gegenprüfung U03–U09
vom 22.10.2026. U01–U02 lagen nicht vor. Plan v1: `add5fee5…8e4a372`. Diese Datei ist
**ohne den früheren Chat ausführbar** (U07). Die Abschnitte zur Kontaktaufnahme (Empfänger,
Fragen) stehen bewusst nicht in diesem öffentlichen Repo; Felix hat sie separat.

**Freigabe (Chat, 22.10.2026):** Umsetzung freigegeben; Pushes auf den Arbeitsbranch
`claude/brave-thompson-9vr9bn` freigegeben. **Nicht** freigegeben: PR/Merge nach `main`,
Website-Änderungen, Paket-Uploads, Mails, Upstream-Issues. Vor jedem dieser Schritte wird
ein lokal geprüfter Kandidat vorgelegt (U03).

Statusspalte: `offen` · `erledigt <commit>` · `blockiert: <Grund>` · `entfällt: <Grund>`.
Fundstellen sind als Datei plus Anker angegeben; Zeilennummern aus älteren Ständen werden vor
der Änderung am aktuellen Text geprüft (U07).

---

## Maßstab

Eine nach außen genannte Behauptung ist sendbar, wenn sie über **öffentliche Links ohne
privaten Zugang** prüfbar ist und die Darstellung Original, Abweichung und eigenen Zusatz
trennt. Jede Behauptung trägt:

- **Prüfstufe:** N = Nachrechnung gespeicherter Evidenz · D = Designprüfung ohne Modell ·
  R = Modellreproduktion (derzeit für keinen Fall vollständig angeboten).
- **Status:** Exploration · Designanalyse · Bestätigung (vorab eingefroren, mit Umfang) ·
  künftiger Versuch.

Ein exploratives oder unentschiedenes Ergebnis kann den Maßstab erfüllen; es wird dadurch
nicht bestätigt.

## Abnahme (Teil A)

Erster Bildschirm: Arbeit korrekt benannt · Neues in deren Begriffen · was nicht behauptet
wird · wie man prüft. Dazu:

1. **Claim-Liste** je Fall (Quelle@Commit, Population/Metrik, Status, Prüfstufe, Prüfbefehl,
   erwartete Ausgabe).
2. **Anonymer Zugang** zu allen versprochenen Artefakten.
3. **Nachrechnung** aus einem sauberen öffentlichen Download; sichtbar, was nur protokolliert ist.
4. **Faire Zuschreibung**; keiner fremden Theorie eine selbst formulierte Vorhersage zuschreiben.
5. **Versionierte Veröffentlichung:** erst Forschungscommit und Pakete festlegen, dann die
   Website auf genau diese Dateien pinnen. Ein Tag ist nur ein Name für einen Commit.

---

## Änderungen v3 gegenüber v2

| Feedback | Änderung in v3 |
|---|---|
| U03 lokaler Kandidat zuerst | Teil D neu: Alles wird lokal vorbereitet und modellfrei geprüft. Dann wird der Kandidat (Repo, Branch, Commit, Dateien, Paketinhalt) vorgelegt. Erst danach folgt die einzeln freigegebene externe Aktion, und erst danach die anonyme Prüfung des tatsächlich veröffentlichten Stands. |
| U04 Ausgangsstände | `docs/sendbar/INTAKE.md`: bestätigte SHAs; Arbeitsbranch nur lokal, bis zum ersten Push; fehlende Eingaben. |
| U05 modellfrei | `docs/sendbar/MODEL_FREE_COMMANDS.md`: Positivliste mit Aufrufketten-Audit. Keine Gesamt-Testsuite. Neue Ergebnisse nur in getrennten Ausgabeorten. |
| U06 Abdeckung statt „8/8“ | 1.1 und 2.1.8: Erwartete Prüfliste und Laufbericht müssen übereinstimmen. Neue Prüfungen werden explizit ergänzt; die Anzahl entspricht der tatsächlichen Suite. Ungelaufene Runden zählen nicht. |
| U07 selbsttragend | jede Aufgabe mit Anker und Abnahme; 2.6.7 ausgeschrieben; alle 68 IDs unten auffindbar |
| U08 Paketumfang | 2.6.3 und 2.7.5: Manifest mit Dateien, Größen, Hashes, Quellen und abgedeckten Claims. Prüfung auf Zugangsdaten, private Inhalte, absolute Pfade, Lizenzen. Keine private Git-Historie. Konkreter Zielort. Vorlage vor dem Upload. |
| U09 Adäquanz | 1.4 neu: Jede Adäquanzregel wird an ihren eingefrorenen Entscheidungsvertrag gebunden. Keine Schwelle wird geändert. |

---

## Teil B — Entscheidungen

| ID | Entscheidung | Status |
|---|---|---|
| D1 | Versand nach Belegreife (Teil A erfüllt); Wellen nur organisatorisch. Die drei Fälle mit Atticus Geiger werden vor dem ersten Versand auf Konsistenz geprüft. | offen (Versand nicht Teil dieser Umsetzung) |
| D2 | Zero-Output-Lauf (Li/Saphra) jetzt nicht ausführen; bleibt offene Frage | erledigt (entschieden) |
| D3 | Proportionaler Makelov-Kandidat mit vollständigem Rechenweg; post hoc, explorativ | offen → 2.2.2 |
| D4 | Öffentliche Prüfpakete TaxiGPT und Causal Parrots, erst nach vervollständigtem Umfang und Prüfer | blockiert: private Eingaben fehlen |
| D5 | Taxi-v1 aus der prominenten Navigation (Vereinfachung, keine Korrektur) | offen → 2.6.4 |
| D6 | Saphra per Merge-Commit integrieren (empfohlen, keine Bedingung) | offen → Teil D |
| D7 | Originaltexte: Ergebnisse der Gegenprüfung übernehmen; offene Details nicht als erledigt markieren | offen |

## Teil C — Arbeitspakete

### Phase 0 — Claim-Listen

| ID | Aufgabe | Abnahme | Status |
|---|---|---|---|
| 0.1 | `applications/<fall>/CLAIMS.json` für Li/Saphra, Makelov, Mixing Mechanisms, CausaLab, Tracr. Erfasst wird jede Zahl und Schlussfolgerung aus Website-Karte, Fall-Lead und Root-README-Zeile. | Jeder Satz im ersten Bildschirm hat einen Eintrag. | offen |
| 0.2 | `scripts/check_claims.py`: führt den Prüfbefehl jedes N/D-Eintrags aus und vergleicht mit der erwarteten Ausgabe. Modellfrei, schreibt nichts. | Läuft im frischen Export grün; Abdeckung im Bericht | offen |
| 0.3 | Prüfmatrix der Gegenprüfung mit 0.1 abgleichen | Jeder Matrixpunkt zugeordnet | blockiert: Matrix liegt nicht vor |

### Phase 1 — Übergreifend

| ID | Aufgabe | Abnahme | Status |
|---|---|---|---|
| 1.1 | „Goodfire“ als Etikett für Mixing Mechanisms ersetzen. Betroffen: `RELEASE_NOTES.md`, `docs/RELEASE_CHECKS.md`, `scripts/verify_release.py` (Label und Variable). `docs/GOODFIRE_CROSSWALK.md` **nicht** ändern, weil die Datei per Hash gebunden ist (`applications/makelov-2311.17030/results/role_baseline_development/manifest.json`, Schlüssel `docs/GOODFIRE_CROSSWALK.md`). Stattdessen eine Klarstellung in `docs/EVIDENCE_MAP.md`. CausaLab-Fall in den Release Notes ergänzen. | `git grep -i goodfire` trifft nur CausaLab, den Dateinamen des Crosswalk und die Klarstellung. Der Verifier meldet jede erwartete Prüfung mit neuem Label. | offen |
| 1.2 | „independent“ nur für externe menschliche Prüfungen. Sonst „AI-agent audit“ oder „pre-run review (reviewer not named)“. Hash-gebundene oder eingefrorene Dateien bleiben unverändert; dafür ein Glossareintrag. | Jede Fundstelle in `.md` eingeordnet (Liste im Commit) | offen |
| 1.3 | README-Abschnitt „Connection to causal abstraction“: Website-Absatz übernehmen, Begriffszuordnung, „established practice“, Box & Hill / Atkinson & Fedorov. Verwandte Arbeiten nur mit bestätigter Identität; sonst „to be added after check“. Tracr nicht als Validierung des CA-Rahmens darstellen. | Keine Neuheitsbehauptung für den Vergleich von Rivalen; jede Quelle verlinkt | offen |
| 1.4 | Für jede verwendete Adäquanzregel: Population, unabhängige Einheit, Fehlermaß und Toleranz, geforderte Abdeckung, Unsicherheitsentscheidung. Dazu Link auf die eingefrorene Definition (U09). | An jeder Verwendung ist das Kriterium erkennbar; keine Zahl geändert | offen |
| 1.5 | Den alten Ratio-Rechner (`docs/USING_THE_METHOD.md`, Abschnitt mit dem ratio calculator) als historisch markieren. Den Faktor ~27 nur mit öffentlichem Nachrechnungsbeleg nennen, sonst kennzeichnen. | Satz am Abschnittsanfang; Claim-Status | offen |
| 1.6 | Links von englischer Doku auf deutsche Dateien mit „(German)“ kennzeichnen. Interne Notizen markieren (`docs/RESEARCH_SCOPE.md`). | kein unmarkierter EN→DE-Link in Einstiegsdokumenten | offen |
| 1.7 | Statuszeilen nur korrigieren, wenn sie falsch sind. Historisch korrekte Zeilen bleiben stehen. | Diff zeigt keine rückwirkende Umdeutung | offen |

### Phase 2 — Fälle

#### 2.1 Li, Saphra et al. (`applications/li-saphra-2507.06445`, aus `24f3e26`)

| ID | Aufgabe | Abnahme | Status |
|---|---|---|---|
| 2.1.1 | Neuer Einstieg im README nach dem Vorbild der Website-Notiz `mechanistic-interpretability/li-saphra/TECHNICAL_NOTE.md`: zuerst der Befund, Rundenchronik nach unten. Ein wörtliches Zitat aus §4.3/App. J erst nach Prüfung am Original; bis dahin nur verlinken. | erster Bildschirm ohne Prozessjargon | offen |
| 2.1.2 | Den Umfang von „confirmed“ eingrenzen: vorab eingefroren bestätigt; Transfer 7/35 adäquat | Wort steht nur mit Umfang | offen |
| 2.1.3 | Head-Typ-Kontrast: erst Gewichtung festlegen, dann records-only rechnen; Exploration | Skript plus Ausgabe; eine Gewichtung | offen |
| 2.1.4 | Umfang: nur Zweischicht-Transfer; Stufe 2 zerlegt R; separierende Mitglieder nur aus invaliden Strings; Value-Runden 001/002/004 auf `a9g0io1r`/H1, Screen003 auf sechs weiteren Heads | jede Aussage mit Quelle | offen |
| 2.1.5 | Zero-Output aus dem Lead in „Open question“; die triviale Schranke streichen | Lead ohne ungelaufenes Experiment | offen |
| 2.1.6 | Historische Statuszeilen nicht ändern (`value_followup/README.md` „confirmation not started“). Bei `VERIFICATION.md`/`PROTOCOL.md` nur ergänzen, falls der Bezug fehlt. | keine Umdeutung | offen |
| 2.1.7 | `history_005` im README als „prepared, not run“ belassen; aus Claims ausnehmen | nicht als Ergebnis gezählt | offen |
| 2.1.8 | Integration: Top-README, `docs/EVIDENCE_MAP.md`, `applications/README.md`, `scripts/verify_release.py` (Native-Followup- und Averaged004-Prüfung, Retrospektive) | erwartete Prüfliste = Laufbericht | offen |
| 2.1.9 | Provenienz: lokaler Freeze keine externe Vorabregistrierung; fehlende Off-target-Tensoren als Grenze der Nachrechnung | in Claims markiert | offen |
| 2.1.10 | Vollständige Autorenliste | blockiert bis Prüfung am Original; bis dahin „Li, Saphra et al.“ mit Link | offen |

#### 2.2 Makelov, Lange & Nanda (`applications/makelov-2311.17030`; App-README eingefroren)

| ID | Aufgabe | Abnahme | Status |
|---|---|---|---|
| 2.2.1 | `OVERVIEW.md`: Geometrie von Q1 nach vorn (gemeinsame Ausgaberichtung, \|c_N\|>\|c_R\| 128/128). „p = 1e-19“ nicht als Schlagzeile. | erster Absatz | offen |
| 2.2.2 | Records-only-Skript für den proportionalen Kandidaten. Erwartet: 0,01935 vs. 0,23834; 64/64. Post hoc, kein Mechanismus. | Skript reproduziert | offen |
| 2.2.3 | Operator in R2/3A benennen; „Original MLP patch“ in `docs/ROUND2_RESULT.md` korrigieren; 0,84 nur mit Rechenweg | jeder „patch“ eindeutig | offen |
| 2.2.4 | Paper-Begriffe (dormant/disconnected); Flip-Rate statt „shifts which name“ | — mit Zahl aus Records | offen |
| 2.2.5 | Neues nach oben; SE aus Table 1 nur mit Rechenweg | Claim-Einträge | offen |
| 2.2.6 | Wu, Geiger et al. 2024 nennen, sobald die Identität bestätigt ist | blockiert, bis Quelle bestätigt | offen |
| 2.2.7 | Provenienz je Runde: 3A-Push `50b7a8d` (2026-09-24T14:06:16Z) verlinken, der lokale Start ist eine eigene Belegart. Q1 ohne öffentlichen Zeitstempel; den Widerspruch zum eingefrorenen App-README erklären. Toleranz-Sensitivität 3A nach vorn. | Belegart je Runde | offen |
| 2.2.8 | Stale: „root README unchanged“ (`VERIFICATION.md`, `docs/MAKELOV_RELEASE_CHECKS.md`); „branch“ in `CASE_STUDY.md`; 16 % vs. 17 %; A/B in der Root-README-Zeile | jede Stelle geprüft | offen |

#### 2.3 Gur-Arieh, Geva & Geiger (`applications/gur-arieh-2510.06182`)

| ID | Aufgabe | Abnahme | Status |
|---|---|---|---|
| 2.3.1 | Setup-Tabelle oben in `CONFIRMATION_RESULT.md` (Modell, Task, n, Zelle, Layer, Position, Readout-Änderung, Auswahl, Mittel/Einzelfall). Werte nur aus Freeze und Records. | allein verständlich | offen |
| 2.3.2 | W_T-Caveat: nur die vorab deklarierte Richtung, keine Größe; an drei Stellen | keine Größenbehauptung | offen |
| 2.3.3 | „narrows its interpretation“ ohne Paper-Beleg umformulieren. Eigene Profile, keine Widerlegung. | — | offen |
| 2.3.4 | Patch-Position `[-1]` und Upstream-Commit `c53372c` offen nennen | — | offen |
| 2.3.5 | Neues nach oben; 58/79 nur mit Rechenweg | Claim-Einträge | offen |
| 2.3.6 | Index-5 höchstens deskriptiv, nach eigener Nachrechnung | keine kausale Deutung | offen |
| 2.3.7 | Stale nur, falls falsch (`README.md` „final approval pending“, `PREFLIGHT_PLAN.md`) | geprüft | offen |

#### 2.4 CausaLab MCQA (`applications/goodfire-mcqa-preflight`)

| ID | Aufgabe | Abnahme | Status |
|---|---|---|---|
| 2.4.1 | Neu positionieren: Hypothesentests von CausaLab und deren Kontrollen anerkennen; unser Beitrag ist eine Designanalyse, kein neuronaler Test. In deren Vokabular. | erster Absatz | offen |
| 2.4.2 | „expose“ → „make explicit“ (Root-README; Website-Karte im Patch); eigene Nennung des Gleichstands anerkennen | — | offen |
| 2.4.3 | Cross-Family-Full-Patch aus `iia_plotted.json` als *ihre* Messung berichten | Zahlen aus Datei | offen |
| 2.4.4 | Farb-Bedingung ausführbar machen (D) | Ausgabe zeigt trennende Familien | offen |
| 2.4.5 | Konkrete Aktion in ihren Begriffen | — | offen |
| 2.4.6 | Wiegreffe et al. und Geiger et al. (DAS) nennen. Autorschaft der Demos und Lizenz nicht unterstellen. | nur Belegtes | offen |
| 2.4.7 | Veralteten Quellhash in `results/summary.json` transparent korrigieren (mit Notiz); `run.py` soll nicht stillschweigend überschreiben | Exporte byteidentisch | offen |

#### 2.5 Causal Abstraction und Tracr

| ID | Aufgabe | Abnahme | Status |
|---|---|---|---|
| 2.5.1 | Neues für CA nach vorn (Baseline „recipient preserved“, MCQA, `no_candidate_fits`) | im CA-Abschnitt | offen |
| 2.5.2 | Tracr: festes, bekanntes Alignment; begrenzte Rivalen; keine Validierung des CA-Rahmens | in `applications/tracr/RESULTS.md` | offen |
| 2.5.3 | Alignment-Suche als Grenze (`docs/METHOD_PREFLIGHT.md`) | ein Absatz | offen |

#### 2.6 TaxiGPT: nur Website plus private Quelle

| ID | Aufgabe | Abnahme | Status |
|---|---|---|---|
| 2.6.1 | Hero: Belehrungssatz von den Autorennamen trennen; Entlastungssatz nach oben; K/V-Befund mit „gleiche Richtig/Falsch-Bewertung 457/512“. Die Ortsgleichheit 291/512 erst mit Paket; „misst Ortsdekodierung, nicht Navigation“. | Website-Patch | offen |
| 2.6.2 | Vergleich Paper ↔ lokal sichtbar; Abweichungsliste nur mit belegten Abweichungen | braucht Prüfung am Original | blockiert (Teil) |
| 2.6.3 | Prüfpaket mit Manifest (U08) | Manifest, lokale Prüfung, Vorlage | blockiert: private Eingaben |
| 2.6.4 | v1 aus der Navigation (`taxigpt-v2/index.html`, Links auf `../taxigpt/`); „Our local replication“ in v1 präzisieren | Website-Patch | offen |
| 2.6.5 | „Nur Kontexte am Ziel“ prominent nennen | Website-Patch | offen |
| 2.6.6 | Polyphonie-Absatz: „Their TaxiGPT example …“ eindeutig machen; Linktext „Manuscript“ | Website-Patch | offen |
| 2.6.7 | Bedingte Einzelprüfungen, jeweils nur ändern, wenn das Problem besteht und belegt ist: (a) „position-subspace“ (Ortsunterraum) vs. Token-Position benennen; (b) Autorennamen auf der Index-Karte; (c) unklare Evidenznotizen in `taxigpt/EVIDENCE.md`; (d) tatsächlich verwendete `data-rate`-Attribute ohne Bindung; (e) Dimension der Zufallskontrolle; (f) gepaarte Routenbefunde (Nicht-Monotonie) nur aus dem Paket | jede Teilprüfung mit Ergebnis | offen |
| 2.6.8 | Verhaltensmaß ist ein künftiger Versuch; eigener Plan | — | entfällt hier |

#### 2.7 Causal Parrots: nur Website plus private Quelle

| ID | Aufgabe | Abnahme | Status |
|---|---|---|---|
| 2.7.1 | Titel umformulieren; CCF-Box sichtbar | Website-Patch | offen |
| 2.7.2 | Offengelegte eigene Operationalisierungen mit Zusatzannahmen, nachträglich; keine CCF-Prognose zuschreiben | Website-Patch | offen |
| 2.7.3 | Frage an die Autoren als Text auf der Seite (welche Zusatzannahmen bräuchte CCF?) | Website-Patch | offen |
| 2.7.4 | CIs der Zellmittel, Aufteilung nach Graph-Antwort, Accuracy aus dem Vorzeichen der Log-Odds | blockiert: Einzeldaten privat | blockiert |
| 2.7.5 | Prüfpaket mit Manifest (U08), vorhandenes Schema (Log-Odds) | blockiert: private Eingaben | blockiert |
| 2.7.6 | Labelmodell und Prüfquote aus der Quelle; Chat-Template-Begründung; Stimulus-Quelle auf der Seite | Stimulus-Quelle: Website-Patch; Rest blockiert | offen (Teil) |
| 2.7.7 | Widersprüche („other models“; `7b726a4`/`e25538a`; „audited“) und SVG | Website-Patch (Text); SVG offen | offen |
| 2.7.8 | Karte auf der Startseite unter „Exploratory“, bis 2.7.5 erledigt ist | Website-Patch | offen |

## Teil D — Ablauf (U03)

1. Alle Änderungen lokal auf `claude/brave-thompson-9vr9bn` (Saphra per Merge-Commit integriert). Zwischenstände werden auf den Arbeitsbranch gepusht (freigegeben).
2. Modellfreie Prüfung in einem **frischen lokalen Export** des Kandidaten-Commits: Positivliste, `verify_release.py`, `check_claims.py`, `check_frozen_files.py`, Linkcheck. Die erwartete Prüfliste wird mit dem Laufbericht verglichen.
3. Prüfung durch frische Agenten gegen Teil A.
4. **Kandidat vorlegen:** Repo, Branch, Commit, geänderte Dateien, Prüfergebnisse, offene Punkte; dazu der Website-Patch als Diff.
5. Erst nach ausdrücklicher Freigabe: PR/Merge nach `main` → anonymen Download dieses Stands prüfen → Website-Pins auf genau diesen Commit → Website-Kandidat vorlegen → einzeln freigeben.
6. Prüfpakete (U08) sobald die privaten Eingaben vorliegen: Manifest, lokale Prüfung, Vorlage, Upload nach Freigabe, anonyme Prüfung.
7. Mails nur einzeln, nie über eine Repository-Freigabe.
