# H3 Mapping Comparison: Deterministic vs Learned (15 Examples)

**Selection rule:** Only ATT&CK techniques mapped in **both** deterministic and learned files (both columns populated).

**Source files (H3 experiment):- Deterministic: `data/mappings/deterministic_attack_defense_lookup.csv` (ransomware-focused, 173 pairs, 46 techniques)
- Learned: `data/mappings/learned_mapping.csv` (190 pairs, 47 techniques)
- Shared techniques available: 46

## D3FEND control legend (symbols used in this table)

Each control ID is a MITRE D3FEND countermeasure shorthand. Full legend for controls in this table:

| Control ID | Meaning |
|------------|---------|
| D3-AI | Asset Inventory |
| D3-AMED | Access Mediation |
| D3-BAC | Backup Access Control |
| D3-BDR | Backup Data Recovery |
| D3-CERO | Certificate Rotation |
| D3-CHN | Connected Honeynet |
| D3-CIA | Container Image Analysis |
| D3-CR | Command Restriction |
| D3-DA | Dynamic Analysis |
| D3-DCE | Dead Code Elimination |
| D3-DE | Decoy Environment |
| D3-DEM | Data Exchange Mapping |
| D3-DENCR | Disk Encryption |
| D3-DF | Decoy File |
| D3-DI | Data Inventory |
| D3-DKE | Disk Erasure |
| D3-DKF | Disk Formatting |
| D3-DKP | Disk Partitioning |
| D3-DO | Decoy Object |
| D3-DQSA | Database Query String Analysis |
| D3-DST | Decoy Session Token |
| D3-EDR | Endpoint Detection and Response |
| D3-ER | Email Removal |
| D3-FA | File Analysis |
| D3-FEMC | Firmware Embedded Monitoring Code |
| D3-FMVV | File Metadata Value Verification |
| D3-HBPI | Hardware-based Process Isolation |
| D3-IPCTA | IPC Traffic Analysis |
| D3-NFP | Network Filtering Policy |
| D3-NRAM | Network Resource Access Mediation |
| D3-ODM | Operational Dependency Mapping |
| D3-OMM | Operating Mode Monitoring |
| D3-PCSV | Process Code Segment Verification |
| D3-PE | Process Eviction |
| D3-PLA | Process Lineage Analysis |
| D3-PM | Platform Monitoring |
| D3-PSA | Process Spawn Analysis |
| D3-PSEP | Process Segment Execution Prevention |
| D3-PWA | Password Authentication |
| D3-RA | Restore Access |
| D3-RFAM | Remote File Access Mediation |
| D3-RTSD | Remote Terminal Session Detection |
| D3-SFA | System File Analysis |
| D3-SIEM | Security Information and Event Management |
| D3-SVCDM | Service Dependency Mapping |
| D3-TBA | Token-based Authentication |
| D3-TBI | TPM Boot Integrity |
| D3-TL | Trusted Library |
| D3-UBA | User Behavior Analysis |
| D3-VPM | Virtual Private Network |

See also: `mapping_control_legend.csv` and `mapping_comparison_examples_labeled.csv` (easy to copy into Excel).

| # | ATT&CK ID | Technique | Det # | Lrn # | Deterministic controls | Learned controls | Overlap | Match |
|---|-----------|-----------|------:|------:|------------------------|-------------------|---------|-------|
| 1 | T1486 | Data Encrypted for Impact | 7 | 5 | D3-AI, D3-DE, D3-DO, D3-FA, D3-PM, D3-RA… | D3-BAC, D3-BDR, D3-CHN, D3-DI, D3-SFA | — | DISJOINT |
| 2 | T1490 | Inhibit System Recovery | 3 | 5 | D3-AI, D3-PM, D3-RA | D3-BAC, D3-BDR, D3-CR, D3-DCE, D3-PCSV | — | DISJOINT |
| 3 | T1485 | Data Destruction | 3 | 4 | D3-DO, D3-FA, D3-RA | D3-DEM, D3-DI, D3-DKE, D3-DQSA | — | DISJOINT |
| 4 | T1487 | Disk Structure Wipe | 2 | 4 | D3-PM, D3-RA | D3-DENCR, D3-DKE, D3-DKF, D3-DKP | — | DISJOINT |
| 5 | T1488 | Disk Content Wipe | 2 | 4 | D3-FA, D3-RA | D3-DENCR, D3-DKE, D3-DKF, D3-DKP | — | DISJOINT |
| 6 | T1055 | Process Injection | 4 | 4 | D3-FA, D3-PE, D3-PM, D3-UBA | D3-HBPI, D3-IPCTA, D3-PLA, D3-PSEP | — | DISJOINT |
| 7 | T1070 | Indicator Removal | 4 | 5 | D3-AI, D3-FA, D3-PM, D3-UBA | D3-EDR, D3-PSA, D3-SIEM, D3-TBI, D3-TL | — | DISJOINT |
| 8 | T1021 | Remote Services | 4 | 5 | D3-AI, D3-AMED, D3-PM, D3-UBA | D3-NFP, D3-OMM, D3-RFAM, D3-RTSD, D3-VPM | — | DISJOINT |
| 9 | T1055.011 | Extra Window Memory Injection | 4 | 4 | D3-FA, D3-PE, D3-PM, D3-UBA | D3-HBPI, D3-PCSV, D3-PLA, D3-PSEP | — | DISJOINT |
| 10 | T1070.006 | Timestomp | 4 | 4 | D3-AI, D3-FA, D3-PM, D3-UBA | D3-DA, D3-FMVV, D3-RFAM, D3-SFA | — | DISJOINT |
| 11 | T1485.001 | Lifecycle-Triggered Deletion | 3 | 4 | D3-DO, D3-FA, D3-RA | D3-CIA, D3-DF, D3-DST, D3-ER | — | DISJOINT |
| 12 | T1489 | Service Stop | 3 | 4 | D3-AI, D3-PM, D3-UBA | D3-NRAM, D3-ODM, D3-OMM, D3-SVCDM | — | DISJOINT |
| 13 | T1021.005 | VNC | 4 | 4 | D3-AI, D3-AMED, D3-PM, D3-UBA | D3-CHN, D3-PWA, D3-RFAM, D3-TBA | — | DISJOINT |
| 14 | T1070.002 | Clear Linux or Mac System Logs | 4 | 4 | D3-AI, D3-FA, D3-PM, D3-UBA | D3-DCE, D3-FEMC, D3-PCSV, D3-SFA | — | DISJOINT |
| 15 | T1041 | Exfiltration Over C2 Channel | 3 | 4 | D3-AI, D3-PM, D3-UBA | D3-CERO, D3-DEM, D3-DI, D3-NRAM | — | DISJOINT |

## Side-by-side with control meanings (copy-friendly)

| # | ATT&CK ID | Deterministic (with meanings) | Learned (with meanings) | Match |
|---|-----------|--------------------------------|-------------------------|-------|
| 1 | T1486 | D3-AI (Asset Inventory); D3-DE (Decoy Environment); D3-DO (Decoy Object); D3-FA (File Analysis); D3-PM (Platform Monitoring); D3-RA (Restore Access); D3-UBA (User Behavior Analysis) | D3-BAC (Backup Access Control); D3-BDR (Backup Data Recovery); D3-CHN (Connected Honeynet); D3-DI (Data Inventory); D3-SFA (System File Analysis) | DISJOINT |
| 2 | T1490 | D3-AI (Asset Inventory); D3-PM (Platform Monitoring); D3-RA (Restore Access) | D3-BAC (Backup Access Control); D3-BDR (Backup Data Recovery); D3-CR (Command Restriction); D3-DCE (Dead Code Elimination); D3-PCSV (Process Code Segment Verification) | DISJOINT |
| 3 | T1485 | D3-DO (Decoy Object); D3-FA (File Analysis); D3-RA (Restore Access) | D3-DEM (Data Exchange Mapping); D3-DI (Data Inventory); D3-DKE (Disk Erasure); D3-DQSA (Database Query String Analysis) | DISJOINT |
| 4 | T1487 | D3-PM (Platform Monitoring); D3-RA (Restore Access) | D3-DENCR (Disk Encryption); D3-DKE (Disk Erasure); D3-DKF (Disk Formatting); D3-DKP (Disk Partitioning) | DISJOINT |
| 5 | T1488 | D3-FA (File Analysis); D3-RA (Restore Access) | D3-DENCR (Disk Encryption); D3-DKE (Disk Erasure); D3-DKF (Disk Formatting); D3-DKP (Disk Partitioning) | DISJOINT |
| 6 | T1055 | D3-FA (File Analysis); D3-PE (Process Eviction); D3-PM (Platform Monitoring); D3-UBA (User Behavior Analysis) | D3-HBPI (Hardware-based Process Isolation); D3-IPCTA (IPC Traffic Analysis); D3-PLA (Process Lineage Analysis); D3-PSEP (Process Segment Execution Prevention) | DISJOINT |
| 7 | T1070 | D3-AI (Asset Inventory); D3-FA (File Analysis); D3-PM (Platform Monitoring); D3-UBA (User Behavior Analysis) | D3-EDR (Endpoint Detection and Response); D3-PSA (Process Spawn Analysis); D3-SIEM (Security Information and Event Management); D3-TBI (TPM Boot Integrity); D3-TL (Trusted Library) | DISJOINT |
| 8 | T1021 | D3-AI (Asset Inventory); D3-AMED (Access Mediation); D3-PM (Platform Monitoring); D3-UBA (User Behavior Analysis) | D3-NFP (Network Filtering Policy); D3-OMM (Operating Mode Monitoring); D3-RFAM (Remote File Access Mediation); D3-RTSD (Remote Terminal Session Detection); D3-VPM (Virtual Private Network) | DISJOINT |
| 9 | T1055.011 | D3-FA (File Analysis); D3-PE (Process Eviction); D3-PM (Platform Monitoring); D3-UBA (User Behavior Analysis) | D3-HBPI (Hardware-based Process Isolation); D3-PCSV (Process Code Segment Verification); D3-PLA (Process Lineage Analysis); D3-PSEP (Process Segment Execution Prevention) | DISJOINT |
| 10 | T1070.006 | D3-AI (Asset Inventory); D3-FA (File Analysis); D3-PM (Platform Monitoring); D3-UBA (User Behavior Analysis) | D3-DA (Dynamic Analysis); D3-FMVV (File Metadata Value Verification); D3-RFAM (Remote File Access Mediation); D3-SFA (System File Analysis) | DISJOINT |
| 11 | T1485.001 | D3-DO (Decoy Object); D3-FA (File Analysis); D3-RA (Restore Access) | D3-CIA (Container Image Analysis); D3-DF (Decoy File); D3-DST (Decoy Session Token); D3-ER (Email Removal) | DISJOINT |
| 12 | T1489 | D3-AI (Asset Inventory); D3-PM (Platform Monitoring); D3-UBA (User Behavior Analysis) | D3-NRAM (Network Resource Access Mediation); D3-ODM (Operational Dependency Mapping); D3-OMM (Operating Mode Monitoring); D3-SVCDM (Service Dependency Mapping) | DISJOINT |
| 13 | T1021.005 | D3-AI (Asset Inventory); D3-AMED (Access Mediation); D3-PM (Platform Monitoring); D3-UBA (User Behavior Analysis) | D3-CHN (Connected Honeynet); D3-PWA (Password Authentication); D3-RFAM (Remote File Access Mediation); D3-TBA (Token-based Authentication) | DISJOINT |
| 14 | T1070.002 | D3-AI (Asset Inventory); D3-FA (File Analysis); D3-PM (Platform Monitoring); D3-UBA (User Behavior Analysis) | D3-DCE (Dead Code Elimination); D3-FEMC (Firmware Embedded Monitoring Code); D3-PCSV (Process Code Segment Verification); D3-SFA (System File Analysis) | DISJOINT |
| 15 | T1041 | D3-AI (Asset Inventory); D3-PM (Platform Monitoring); D3-UBA (User Behavior Analysis) | D3-CERO (Certificate Rotation); D3-DEM (Data Exchange Mapping); D3-DI (Data Inventory); D3-NRAM (Network Resource Access Mediation) | DISJOINT |

## Key takeaway

- EXACT match: 0/15
- PARTIAL overlap: 0/15
- DISJOINT (no shared controls): 15/15

Deterministic mappings prioritize ransomware-relevant D3FEND controls (e.g., D3-RA Restore Access). Learned mappings assign broader heuristic controls with little or no overlap — consistent with H3 DAC findings (deterministic 100%, learned 0%).