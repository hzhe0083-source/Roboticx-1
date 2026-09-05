# AI Generation Prompts

The first AI drafts were generated from natural-language descriptions only; the user's Transformer figure was not supplied to the generator. Later correction passes referenced only the preceding AI draft so the composition stayed stable while individual connections were repaired.

## VA Internal Structure

Create an original, minimal horizontal paper-method diagram on white. Show `VA Inputs` feeding a distinct `WQ → Q` path into `Shared Attention`; place `Key / Value Projections` above with separate `WK → K` and `WV → V` paths sourced from Vision, Visual Memory, Action, Language, and a projected delayed WM message. Continue through `Residual Add → Pre-Norm + FFN → Residual Add → VA Outputs`, with one lower and one upper residual bypass. Use flat Transformer-like pastel pink, peach, yellow-green, cyan, and restrained blue; readable sans-serif text; no copied textbook layout, icons, decoration, shadows, gradients, or watermark.

## VA–WM Interaction

Create an original, minimal horizontal paper-method diagram on white. Put `Shared Snapshot Sᵢ₋₁` above, current `Vision–Action (VA)ᵢ` left, current `World Memory (WM)ᵢ` right, two old-snapshot message cards in the center, and `Commit Sᵢ` below. The orange old-VA-token card points only into current WM; the blue delayed-WM-message card points only into the K,V region of current VA. Current VA and WM outputs point only to Commit. Absolutely no same-stage VA↔WM shortcut and no current `Zᵢ→VAᵢ` feedback. Use flat Transformer-like pastels, large readable text, clear arrows, no copied layout, icons, decoration, shadows, gradients, or watermark.

The VA draft gained its second residual Add/bypass. The interaction draft was then refined locally to expose the snapshot and commit fields, split VA evidence/action paths, add the post-predictor belief update, and close the single blue chain `derived WM message → WM K,V → Shared Attention` without copying the user's reference layout.

## Final Minimal AI Overview

Generated from a natural-language architecture description only: a single shared snapshot row, two large VA/WM blocks, orange `V→Evidence · A→Predictor`, blue delayed `Z→stop-grad→Kᴡ,Vᴡ`, and a single atomic Commit row. Two surgical edits referenced only the generated draft: connect the blue publication line, then replace the WM summary with `Evidence → B̃ᵢ → Predict Zᵢ → Bᵢ`. The supplied Transformer image was never passed to the generator.
