# OpenAI preference500 supplementary evaluation

Train847 / calibration211 / test500, grouped by normalized prompt. Three fixed seeds7/42/2026:313/500,319/500,321/500; mean accuracy63.5333%.

Seed42 calibration: short15/20 and current30/40 tie at146/211; long60/80 gives144/211. Predeclared tie rule selects fewer epochs. Short configuration is frozen for all seeds. No test outcome changes lambda or the selected configuration.

Independent score/loss/lambda checks pass. Flipping all500 final-test labels and refitting gives the byte-identical selected checkpoint; accuracy changes313->187, confirming test-label separation.

Important:400 test pairs were training rows and100 were calibration/holdout rows in the older project. They are held out from these new fits, not unseen across the entire project history. Their identity is not verified against the authors'500 pairs. Do not compare63.53% with paper65% as matched reproduction error.

This supplementary protocol does not replace the main1245/313 protocol or the completed core results. No new API calls. Real OpenAI1536D base vectors are reused.
