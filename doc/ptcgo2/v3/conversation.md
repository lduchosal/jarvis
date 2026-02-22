Résumé essentiel de la conversation (v3)

1) Point de départ
Le drift HP global était bloqué autour de 7.5-7.9 (seuil cible < 5.0). Le problème semblait d'abord venir de Pass.

2) Diagnostic clé
Le bug principal identifié côté training: la loss Pass supervisait la mauvaise tête (delta_obs) alors que l'inférence C# lit hp_head. La conversion Torch -> ONNX n'est pas la cause (parity check OK).

3) Correction Pass
Après correction (loss Pass sur hp_head, puis ajustements), Pass est devenu quasi parfait en benchmark (HP divergence on Pass ~0). Conclusion: Pass n'était plus le goulot principal.

4) Nouveau goulot
Le drift résiduel venait surtout des attaques (erreur HP défenseur). Le levier efficace a été le passage à une loss attaque plus adaptée (L1), avec entraînement long.

5) Résultat v3.9
Le checkpoint final a fortement réduit le drift HP (jusqu'à ~1.64), mais Done F1 a chuté (~0.789). Problème: critère de sélection de checkpoint basé sur val_loss globale, mal corrélé au benchmark final.

6) Rebalancing done (v3.10-F1)
Spécification appliquée: augmenter le poids done et utiliser une sélection de checkpoint orientée métriques benchmark. Résultat v3.10-F1 (best/final) = PASS complet et stable:
- Drift/game ~2.68 (<5.0)
- Done F1 ~0.959 (>0.95)
- HP MAE ~0.063 (<1.0)
- Pass HP ~0.0023 (<0.02)
- Reward MAE ~0.036 (<0.05)
- KO miss 0

7) État actuel
Le modèle est validé fonctionnellement sur les métriques cibles. Seul sujet restant hors qualité modèle: crash runtime en fin de benchmark (mutex lock failed), à traiter séparément.
